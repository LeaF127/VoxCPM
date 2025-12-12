import tempfile
from dataclasses import dataclass
from typing import Callable, Generator, Iterable, List, Optional, Tuple, Dict

import numpy as np
import soundfile as sf

from voxcpm.core import VoxCPM

try:
    from funasr import AutoModel
except Exception:  # pragma: no cover - funasr is optional at import time
    AutoModel = None  # type: ignore


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


@dataclass
class ASRResult:
    text: str
    is_final: bool
    segment_index: int


class AudioChunker:
    """
    简单的音频切块器，按照给定窗口/步长从 wav 读取数据。
    """

    def __init__(self, wav_path: str, chunk_ms: int = 800, hop_ms: int = 400, target_sr: int = 16000):
        self.wav_path = wav_path
        self.chunk_ms = chunk_ms
        self.hop_ms = hop_ms
        self.target_sr = target_sr

    def stream(self) -> Generator[np.ndarray, None, None]:
        audio, sr = sf.read(self.wav_path, dtype="float32", always_2d=False)
        if sr != self.target_sr:
            raise ValueError(f"expected sample rate {self.target_sr}, got {sr}")
        audio = _ensure_mono(audio)
        chunk_size = int(self.target_sr * self.chunk_ms / 1000)
        hop_size = int(self.target_sr * self.hop_ms / 1000)
        if chunk_size <= 0 or hop_size <= 0:
            raise ValueError("chunk_ms and hop_ms must be positive")

        offset = 0
        total = audio.shape[0]
        while offset < total:
            end = min(offset + chunk_size, total)
            chunk = audio[offset:end]
            # pad last chunk to fixed length for downstream components
            if chunk.shape[0] < chunk_size:
                pad = np.zeros(chunk_size - chunk.shape[0], dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)
            yield chunk
            offset += hop_size


class EnergyVAD:
    """
    轻量级能量 VAD，用于判定当前 chunk 是否包含语音。
    """

    def __init__(self, energy_threshold: float = 0.005):
        self.energy_threshold = energy_threshold

    def is_speech(self, chunk: np.ndarray) -> bool:
        rms = np.sqrt(np.mean(np.square(chunk.astype(np.float32)))) if chunk.size else 0.0
        return rms > self.energy_threshold


class StreamingASR:
    """
    基于 chunk 的增量 ASR：在检测到静音后提交分段调用 ASR。
    """

    def __init__(
        self,
        asr_model: "AutoModel",
        sample_rate: int = 16000,
        language: str = "auto",
        min_speech_chunks: int = 1,
        min_silence_chunks: int = 2,
        emit_partial: bool = False,
    ):
        if asr_model is None:
            raise ValueError("funasr AutoModel 未安装或未提供")
        self.asr_model = asr_model
        self.sample_rate = sample_rate
        self.language = language
        self.min_speech_chunks = min_speech_chunks
        self.min_silence_chunks = min_silence_chunks
        self.emit_partial = emit_partial
        self._buffer: List[np.ndarray] = []
        self._silence_count = 0
        self._segment_index = 0

    def _flush(self) -> Optional[ASRResult]:
        if len(self._buffer) < self.min_speech_chunks:
            self._buffer.clear()
            return None
        audio = np.concatenate(self._buffer, axis=0)
        self._buffer.clear()
        self._silence_count = 0
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, audio, self.sample_rate)
            res = self.asr_model.generate(input=tmp.name, language=self.language, use_itn=True)
        text = ""
        if res and isinstance(res, list):
            text = res[0].get("text", "")
            # SenseVoice 会返回 "xxx|>transcript"
            if "|>" in text:
                text = text.split("|>")[-1]
        result = ASRResult(text=text.strip(), is_final=True, segment_index=self._segment_index)
        self._segment_index += 1
        return result

    def transcribe(self, chunks: Iterable[np.ndarray], vad: Optional[EnergyVAD] = None) -> Generator[ASRResult, None, None]:
        for chunk in chunks:
            is_speech = vad.is_speech(chunk) if vad is not None else True
            if is_speech:
                self._buffer.append(chunk)
                self._silence_count = 0
                if self.emit_partial and len(self._buffer) >= self.min_speech_chunks:
                    partial = self._flush()
                    if partial:
                        partial.is_final = False
                        yield partial
            else:
                self._silence_count += 1
                if self._silence_count >= self.min_silence_chunks and len(self._buffer) > 0:
                    result = self._flush()
                    if result:
                        yield result
        # flush tail
        tail = self._flush()
        if tail:
            yield tail


def default_translate(text: str, target_lang: str = "auto") -> str:
    """
    占位翻译函数，当前直接回传输入。
    预留接入外部/自定义翻译模型的入口。
    """
    _ = target_lang  # 保留接口
    return text


def run_streaming_translation(
    audio_path: str,
    tts: VoxCPM,
    asr_model: "AutoModel",
    *,
    language: str = "auto",
    target_lang: str = "auto",
    prompt_wav: Optional[str] = None,
    prompt_text: Optional[str] = None,
    chunk_ms: int = 800,
    hop_ms: int = 400,
    energy_threshold: float = 0.005,
    min_silence_chunks: int = 2,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    translate_fn: Callable[[str, str], str] = default_translate,
    merge_prompt_cache: bool = True,
) -> Generator[Dict[str, object], None, None]:
    """
    端到端流式语音翻译：ASR -> (占位)翻译 -> VoxCPM 流式合成。

    Yields:
        dict:
            type: "asr" / "tts"
            text: 识别或翻译后的文本
            is_final: 是否为最终分段
            segment_index: 分段编号
            audio: (可选) float32 ndarray，单声道 16k PCM
    """
    chunker = AudioChunker(audio_path, chunk_ms=chunk_ms, hop_ms=hop_ms, target_sr=tts.tts_model.sample_rate)
    vad = EnergyVAD(energy_threshold=energy_threshold)
    asr = StreamingASR(
        asr_model=asr_model,
        sample_rate=tts.tts_model.sample_rate,
        language=language,
        min_speech_chunks=1,
        min_silence_chunks=min_silence_chunks,
    )

    prompt_cache = None
    if prompt_wav and prompt_text:
        prompt_cache = tts.tts_model.build_prompt_cache(prompt_text=prompt_text, prompt_wav_path=prompt_wav)

    for asr_res in asr.transcribe(chunker.stream(), vad=vad):
        yield {
            "type": "asr",
            "text": asr_res.text,
            "is_final": asr_res.is_final,
            "segment_index": asr_res.segment_index,
        }
        translated = translate_fn(asr_res.text, target_lang)
        tts_stream = tts.tts_model._generate_with_prompt_cache(
            target_text=translated,
            prompt_cache=prompt_cache,
            streaming=True,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        for audio_chunk, text_token, pred_audio_feat in tts_stream:
            pcm = audio_chunk.squeeze(0).cpu().numpy()
            yield {
                "type": "tts",
                "text": translated,
                "is_final": False,
                "segment_index": asr_res.segment_index,
                "audio": pcm,
            }
            if merge_prompt_cache:
                prompt_cache = tts.tts_model.merge_prompt_cache(prompt_cache, text_token, pred_audio_feat)
        yield {
            "type": "tts",
            "text": translated,
            "is_final": True,
            "segment_index": asr_res.segment_index,
        }




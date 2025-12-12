#!/usr/bin/env python3
"""
Streamed speech translation pipeline (ASR -> MT placeholder -> VoxCPM TTS).

Example:
    python -m voxcpm.cli.stream_translate \\
        --wav ./examples/example.wav \\
        --model-path ./models/VoxCPM-0.5B \\
        --prompt-wav ./examples/example.wav \\
        --prompt-text "这是一个参考文本" \\
        --chunk-ms 800 --hop-ms 400
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.stream import run_streaming_translation

try:
    from funasr import AutoModel
except Exception:
    AutoModel = None  # type: ignore


def _validate_wav(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"wav 文件不存在: {path}")
    return p


def _load_asr(device: str) -> "AutoModel":
    if AutoModel is None:
        raise RuntimeError("未安装 funasr，无法加载流式 ASR（pip install funasr）")
    return AutoModel(
        model="iic/SenseVoiceSmall",
        disable_update=True,
        log_level="INFO",
        device=device,
    )


def _load_voxcpm(args) -> VoxCPM:
    if args.model_path:
        return VoxCPM(
            voxcpm_model_path=args.model_path,
            enable_denoiser=not args.no_denoiser,
            optimize=not args.no_optimize,
        )
    hf_id = args.hf_model_id or os.environ.get("HF_REPO_ID", "openbmb/VoxCPM-0.5B")
    return VoxCPM.from_pretrained(
        hf_model_id=hf_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        load_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
    )


def _concat_audio(buffers: List[np.ndarray]) -> np.ndarray:
    if not buffers:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(buffers, axis=0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="VoxCPM 流式语音翻译 CLI")
    parser.add_argument("--wav", required=True, help="输入 16k PCM wav 文件路径")
    parser.add_argument("--language", default="auto", help="ASR 语言，默认 auto")
    parser.add_argument("--target-lang", default="auto", help="占位翻译目标语种")
    parser.add_argument("--model-path", default=None, help="本地 VoxCPM 模型目录")
    parser.add_argument("--hf-model-id", default=None, help="HuggingFace repo id，若未提供 model-path 时使用")
    parser.add_argument("--cache-dir", default=None, help="模型缓存目录")
    parser.add_argument("--local-files-only", action="store_true", help="仅使用本地模型文件")
    parser.add_argument("--prompt-wav", default=None, help="提示音频路径，可选")
    parser.add_argument("--prompt-text", default=None, help="提示音频对应文本，可选")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="VoxCPM CFG 值")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="扩散步数")
    parser.add_argument("--chunk-ms", type=int, default=800, help="ASR/流式切块长度 ms")
    parser.add_argument("--hop-ms", type=int, default=400, help="ASR/流式步长 ms")
    parser.add_argument("--energy-threshold", type=float, default=0.005, help="能量 VAD 阈值")
    parser.add_argument("--min-silence-chunks", type=int, default=2, help="静音帧数达到后触发分段")
    parser.add_argument("--no-denoiser", action="store_true", help="禁用提示音频降噪")
    parser.add_argument("--no-optimize", action="store_true", help="禁用 torch.compile 优化")
    parser.add_argument("--output", default=None, help="输出合成 wav 保存路径（可选）")
    parser.add_argument("--no-play", action="store_true", help="不尝试实时播放")

    args = parser.parse_args()

    wav_path = _validate_wav(args.wav)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 使用设备: {device}")

    asr_model = _load_asr(device)
    print("[INFO] 已加载增量 ASR (SenseVoiceSmall)")

    tts = _load_voxcpm(args)
    print("[INFO] 已加载 VoxCPM")

    audio_buffers: List[np.ndarray] = []
    try:
        for event in run_streaming_translation(
            audio_path=str(wav_path),
            tts=tts,
            asr_model=asr_model,
            language=args.language,
            target_lang=args.target_lang,
            prompt_wav=args.prompt_wav,
            prompt_text=args.prompt_text,
            chunk_ms=args.chunk_ms,
            hop_ms=args.hop_ms,
            energy_threshold=args.energy_threshold,
            min_silence_chunks=args.min_silence_chunks,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        ):
            if event["type"] == "asr":
                status = "FINAL" if event.get("is_final") else "PARTIAL"
                print(f"[ASR-{status}] {event.get('text','')}")
            elif event["type"] == "tts" and "audio" in event:
                audio_chunk = event["audio"]
                audio_buffers.append(audio_chunk)
                if not args.no_play:
                    try:
                        import sounddevice as sd  # type: ignore
                        sd.play(audio_chunk, tts.tts_model.sample_rate, blocking=False)
                    except Exception:
                        # 回退为打印长度
                        print(f"[TTS] chunk samples={len(audio_chunk)}")
        final_audio = _concat_audio(audio_buffers)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.output, final_audio, tts.tts_model.sample_rate)
            print(f"[INFO] 已保存到 {args.output}, 时长 {len(final_audio)/tts.tts_model.sample_rate:.2f}s")
    except KeyboardInterrupt:
        print("中断，退出。")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()




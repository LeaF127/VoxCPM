"""
端到端 S2ST 流程编排服务。

整合 ASR、翻译和 TTS 的完整流程。
"""
import asyncio
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

import numpy as np
import soundfile as sf

from ..api.models import StreamingRequest, SegmentResult, StreamingResponseModel
from ..api.dependencies import select_tts_text
from ..clients.asr_websocket import ASRWebSocketClient
from voxcpm.utils import get_test_logger, load_audio, segment_audio_by_timestamp, concatenate_audio
from ..utils.timing_stats import TimingStats, TimingStatsCollector

_logger = get_test_logger(__file__)


class S2STService:
    """
    端到端 S2ST 流程编排服务。

    负责协调 ASR、翻译和 TTS 各个模块，完成从音频到音频的翻译流程。
    """

    def __init__(self, tts_model):
        """
        初始化 S2ST 服务。

        Args:
            tts_model: 已加载的 TTS 模型实例
        """
        self.tts_model = tts_model
        self.logger = _logger

    async def process_streaming(
        self,
        audio_path: Path,
        translator: ASRWebSocketClient,
        request: StreamingRequest,
        event_queue: asyncio.Queue,
    ) -> None:
        """
        流式处理：逐句识别+翻译+TTS。

        Args:
            audio_path: 输入音频文件路径
            translator: ASR 翻译器实例
            request: 请求参数
            event_queue: 事件队列，用于发送 SSE 事件
        """
        request_id = str(uuid.uuid4())[:8]
        segment_files: List[Path] = []
        segment_output_dir = Path(f"s2st_demo/output/segments/{request_id}")
        segment_output_dir.mkdir(parents=True, exist_ok=True)
        segment_index = 0
        tts_sample_rate = getattr(self.tts_model, "sample_rate", 16000)
        temp_prompt_files: List[Path] = []
        last_intermediate_message: Optional[Dict[str, Any]] = None
        processing_complete = False

        # 加载原始音频用于切分
        target_sr = 16000
        original_audio, original_sr = load_audio(audio_path, target_sr=target_sr, mono=True)

        async def on_final_callback(parsed: Dict[str, Any]) -> None:
            """当接收到一句完整的话时触发"""
            nonlocal segment_index, last_intermediate_message

            result_text = parsed.get("result_text", "").strip()
            trans_text = parsed.get("trans_text", "").strip()
            raw_data = parsed.get("raw", {})

            begin_ms = int(raw_data.get("Begin", 0) or 0)
            end_ms = int(raw_data.get("End", 0) or 0)

            if parsed.get("is_stream_finished") and begin_ms == 0 and end_ms == 0:
                if last_intermediate_message:
                    last_raw = last_intermediate_message.get("raw", {})
                    begin_ms = int(last_raw.get("Begin", 0) or 0)
                    end_ms = int(last_raw.get("End", 0) or 0)

            tts_text = select_tts_text(result_text, trans_text, request.tts_text_source)
            if not tts_text:
                await event_queue.put(
                    StreamingResponseModel(
                        event_type="progress", data={"message": f"[句 {segment_index + 1}] 跳过合成（文本为空）"}
                    )
                )
                return

            # 根据时间戳切分原始音频作为 prompt_wav
            prompt_wav_path = None
            prompt_text_for_tts = None

            if begin_ms >= 0 and end_ms > begin_ms and result_text:
                try:
                    audio_segment = segment_audio_by_timestamp(
                        original_audio,
                        sample_rate=target_sr,
                        begin_ms=begin_ms,
                        end_ms=end_ms,
                        target_sr=tts_sample_rate,
                    )

                    temp_prompt_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=segment_output_dir)
                    temp_prompt_file.close()
                    temp_prompt_path = Path(temp_prompt_file.name)
                    sf.write(str(temp_prompt_path), audio_segment, tts_sample_rate)
                    temp_prompt_files.append(temp_prompt_path)

                    prompt_wav_path = str(temp_prompt_path)
                    prompt_text_for_tts = result_text
                except Exception as e:
                    self.logger.warning("[%s] 切分音频失败: %s", request_id, e)

            if not prompt_wav_path and request.prompt_wav_path:
                prompt_wav_path = request.prompt_wav_path
                prompt_text_for_tts = request.prompt_text or result_text

            # 合成语音
            segment_file = segment_output_dir / f"segment_{segment_index:03d}.wav"
            try:
                audio = self.tts_model.generate(
                    text=tts_text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text_for_tts,
                    cfg_value=request.cfg_value,
                    inference_timesteps=request.inference_timesteps,
                    normalize=request.normalize,
                    denoise=request.denoise and prompt_wav_path is not None,
                )

                sf.write(str(segment_file), audio, tts_sample_rate)
                segment_files.append(segment_file)
                duration = len(audio) / float(tts_sample_rate)

                # 发送单句完成事件
                segment_result = SegmentResult(
                    segment_index=segment_index,
                    asr_text=result_text,
                    trans_text=trans_text,
                    tts_text=tts_text,
                    begin_ms=begin_ms,
                    end_ms=end_ms,
                    segment_file=f"segments/{request_id}/{segment_file.name}",
                    duration=duration,
                )
                await event_queue.put(
                    StreamingResponseModel(event_type="segment", data=segment_result.model_dump())
                )
                segment_index += 1
            except Exception as e:
                self.logger.error("[%s] [句 %d] 合成失败: %s", request_id, segment_index + 1, e, exc_info=True)
                await event_queue.put(
                    StreamingResponseModel(
                        event_type="error", data={"message": f"[句 {segment_index + 1}] 合成失败: {e}"}
                    )
                )

        async def on_intermediate_callback(parsed: Dict[str, Any]) -> None:
            """中间结果回调"""
            nonlocal last_intermediate_message
            last_intermediate_message = parsed

        # 在后台任务中执行流式识别+翻译
        async def process_audio():
            nonlocal processing_complete
            try:
                await translator.transcribe_streaming(
                    audio_path=str(audio_path),
                    on_intermediate=on_intermediate_callback,
                    on_final=on_final_callback,
                )

                if not segment_files:
                    await event_queue.put(
                        StreamingResponseModel(event_type="error", data={"message": "未生成任何音频片段"})
                    )
                    processing_complete = True
                    return

                # 拼接所有片段
                try:
                    audio_segments: List[np.ndarray] = []
                    sample_rates: List[int] = []
                    for seg_file in segment_files:
                        audio, sr = sf.read(str(seg_file))
                        audio_segments.append(audio)
                        sample_rates.append(sr)

                    final_audio, final_sr = concatenate_audio(
                        audio_segments, sample_rates, target_sr=tts_sample_rate
                    )
                    final_output_path = Path(f"s2st_demo/output/{request_id}_final.wav")
                    final_output_path.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(final_output_path), final_audio, final_sr)

                    duration = len(final_audio) / float(final_sr)
                    await event_queue.put(
                        StreamingResponseModel(
                            event_type="complete",
                            data={
                                "output_file": str(final_output_path),
                                "duration": duration,
                                "segment_count": len(segment_files),
                            },
                        )
                    )
                except Exception as e:
                    self.logger.error("[%s] 拼接音频失败: %s", request_id, e, exc_info=True)
                    await event_queue.put(
                        StreamingResponseModel(
                            event_type="error", data={"message": f"拼接音频失败: {e}"}
                        )
                    )
                finally:
                    # 清理临时文件
                    try:
                        if audio_path.exists():
                            audio_path.unlink()
                        for temp_file in temp_prompt_files:
                            if temp_file.exists():
                                temp_file.unlink()
                    except Exception as e:
                        self.logger.warning("[%s] 清理临时文件失败: %s", request_id, e)
                    processing_complete = True
            except Exception as e:
                self.logger.error("[%s] 流式处理失败: %s", request_id, e, exc_info=True)
                await event_queue.put(
                    StreamingResponseModel(event_type="error", data={"message": f"流式处理失败: {e}"})
                )
                processing_complete = True

        # 启动后台处理任务
        asyncio.create_task(process_audio())

    async def process_non_streaming(
        self,
        audio_path: Path,
        translator: ASRWebSocketClient,
        tts_text_source: str,
        output_path: Path,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        normalize: bool = True,
        denoise: bool = False,
    ) -> Optional[Path]:
        """
        非流式处理：等待整段音频识别+翻译完成后，一次性合成。

        Args:
            audio_path: 输入音频文件路径
            translator: ASR 翻译器实例
            tts_text_source: TTS 文本来源（"trans" 或 "asr"）
            output_path: 输出音频文件路径
            prompt_wav: 参考音频路径
            prompt_text: 参考音频对应文本
            cfg_value: CFG 值
            inference_timesteps: 推扩散步数
            normalize: 是否文本正则化
            denoise: 是否降噪

        Returns:
            输出音频文件路径，失败返回 None
        """
        self.logger.info("开始非流式识别+翻译流程")

        asr_result = await translator.run(audio_path=str(audio_path), streaming=False)
        if not asr_result:
            self.logger.error("未获得识别/翻译结果")
            return None

        result_text = asr_result.get("result_text", "")
        trans_text = asr_result.get("trans_text", "")
        self.logger.info("ASR 结果: %s", result_text)
        self.logger.info("翻译结果: %s", trans_text)

        tts_text = select_tts_text(result_text, trans_text, tts_text_source)
        if not tts_text:
            return None

        self.logger.info("即将进入 VoxCPM 合成阶段")

        audio = self.tts_model.generate(
            text=tts_text,
            prompt_wav_path=prompt_wav,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=normalize,
            denoise=denoise and prompt_wav is not None,
        )

        sample_rate = getattr(self.tts_model, "sample_rate", 16000)
        sf.write(str(output_path), audio, sample_rate)

        duration = len(audio) / float(sample_rate)
        self.logger.info("端到端流程完成，最终输出音频: %s, 时长: %.2f 秒", output_path, duration)

        return output_path

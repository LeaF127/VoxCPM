import argparse
import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, AsyncIterator

import numpy as np
import soundfile as sf
import torch
import torchaudio

from voxcpm.utils import get_test_logger, load_audio, resample_audio, segment_audio_by_timestamp, concatenate_audio

if os.environ.get("PYTHONPATH", "") == "":
    os.environ["PYTHONPATH"] = "./s2st_demo"
else:
    os.environ["PYTHONPATH"] = "./s2st_demo:" + os.environ["PYTHONPATH"]
    
from asr_translate import ASRTranslator
from tts import load_voxcpm, synthesize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端：识别 + 翻译 + VoxCPM 合成 Demo")

    # 输入音频
    parser.add_argument(
        "--audio",
        default="s2st_demo/input/example.wav",
        help="待识别的音频路径（建议 16kHz 单声道 WAV）",
    )

    # ASR / 翻译服务相关参数（沿用 test_asr_non_stream 的默认约定）
    parser.add_argument("--ws-url", default=os.getenv("WS_URL", "ws://175.24.179.12:9301/dotcwsasr"))
    parser.add_argument("--user-id", default=os.getenv("USER_ID", "y123456"))
    parser.add_argument("--token", default=os.getenv("TOKEN", "token12345-1730889600"))
    parser.add_argument("--from-lang", default=os.getenv("FROM_LAN", "cn"))
    parser.add_argument("--to-lang", default=os.getenv("TO_LAN", "en"))
    parser.add_argument("--role", default=os.getenv("ROLE", "0"))
    parser.add_argument("--lan-id", default=os.getenv("LAN_ID", "0"))
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("SAMPLE_RATE", 16000)))
    parser.add_argument("--bit-rate", type=int, default=int(os.getenv("BIT_RATE", 16)))
    parser.add_argument("--interval", type=float, default=float(os.getenv("INTERVAL", 0.1)))

    # VoxCPM 模型加载相关
    parser.add_argument("--model-path", default=os.getenv("VOXCPM_MODEL_PATH"), help="本地 VoxCPM 模型目录")
    parser.add_argument(
        "--hf-model-id",
        default=os.getenv("VOXCPM_HF_ID", "openbmb/VoxCPM-0.5B"),
        help="Hugging Face 模型 ID（未提供 model-path 时使用）",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("VOXCPM_CACHE_DIR"),
        help="VoxCPM 模型缓存目录（可选）",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="仅使用本地 VoxCPM 文件（不访问网络）",
    )
    parser.add_argument("--no-denoiser", action="store_true", help="禁用 VoxCPM 降噪模型")
    parser.add_argument("--no-optimize", action="store_true", help="禁用 torch.compile 优化")

    # TTS 相关参数
    parser.add_argument("--prompt-wav", default=None, help="VoxCPM 参考音频路径（可选）")
    parser.add_argument("--prompt-text", default=None, help="参考音频对应文本（可选）")
    parser.add_argument("--cfg-value", type=float, default=2.0, help="VoxCPM CFG 指数，默认 2.0")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="VoxCPM 扩散步数，默认 10")
    parser.add_argument("--normalize", action="store_true", help="启用文本正则化")
    parser.add_argument("--denoise", action="store_true", help="对提示音频进行降噪")

    parser.add_argument(
        "--tts-text-source",
        choices=["trans", "asr"],
        default="trans",
        help="用于合成的文本来源：trans=翻译结果，asr=原始识别文本",
    )
    parser.add_argument(
        "--output",
        default="s2st_demo/output/tts_from_asr.wav",
        help="最终合成 WAV 输出路径",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="启用流式模式：逐句识别+翻译+TTS（而非等待整段音频处理完成）",
    )
    parser.add_argument(
        "--segment-output-dir",
        default="s2st_demo/output/segments",
        help="流式模式下，每句合成的音频片段保存目录",
    )
    parser.add_argument(
        "--input-mode",
        choices=["file", "mic"],
        default="file",
        help="输入模式：file=从文件读取，mic=从麦克风实时输入",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="音频输入设备 ID（麦克风模式，None 表示使用默认设备）",
    )

    return parser.parse_args()


def _select_tts_text(
    result_text: str,
    trans_text: str,
    source: str,
    logger,
) -> Optional[str]:
    """
    根据配置选择用于 TTS 的文本。
    """
    asr_text = (result_text or "").strip()
    mt_text = (trans_text or "").strip()

    if source == "trans":
        if mt_text:
            logger.info("使用翻译结果作为合成文本")
            return mt_text
        if asr_text:
            logger.info("翻译结果为空，回退到原始识别文本")
            return asr_text
    else:
        if asr_text:
            logger.info("使用原始识别文本作为合成文本")
            return asr_text
        if mt_text:
            logger.info("识别文本为空，回退到翻译结果")
            return mt_text

    logger.error("ASR/翻译结果均为空，无法进行合成")
    return None


async def run_non_streaming_pipeline(
    audio_path: Path,
    translator: ASRTranslator,
    tts,
    args: argparse.Namespace,
    logger,
) -> Optional[Path]:
    """
    非流式流程：等待整段音频识别+翻译完成后，一次性合成。
    """
    logger.info("开始非流式识别+翻译流程")

    asr_result = await translator.run(audio_path=str(audio_path), streaming=False)
    if not asr_result:
        logger.error("未获得识别/翻译结果")
        return None

    result_text = asr_result.get("result_text", "")
    trans_text = asr_result.get("trans_text", "")
    logger.info("ASR 结果: %s", result_text)
    logger.info("翻译结果: %s", trans_text)

    tts_text = _select_tts_text(result_text, trans_text, args.tts_text_source, logger)
    if not tts_text:
        return None

    logger.info("即将进入 VoxCPM 合成阶段")

    out_path = synthesize(
        tts=tts,
        text=tts_text,
        output_path=args.output,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise,
    )

    logger.info("端到端流程完成，最终输出音频: %s", out_path)
    return out_path


async def run_streaming_pipeline(
    audio_path: Path,
    translator: ASRTranslator,
    tts,
    args: argparse.Namespace,
    logger,
) -> Optional[Path]:
    """
    流式流程：逐句识别+翻译+TTS，每句完整后立即合成并保存为单独文件。
    整体结束后拼接所有片段为完整文件。
    
    利用 ASR 返回的时间戳切分原始音频作为 prompt_wav。
    """
    logger.info("开始流式识别+翻译+TTS 流程")

    segment_output_dir = Path(args.segment_output_dir)
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始音频用于切分，并重采样为16k
    logger.info("加载原始音频文件: %s", audio_path)
    target_sr = 16000  # 原始音频重采样目标采样率
    original_audio, original_sr = load_audio(audio_path, target_sr=target_sr, mono=True)
    logger.info("原始音频已重采样为: %d Hz, 时长: %.2f 秒", original_sr, len(original_audio) / original_sr)

    segment_files: List[Path] = []
    segment_index = 0
    sample_rate = getattr(tts.tts_model, "sample_rate", 16000)
    temp_prompt_files: List[Path] = []  # 用于清理临时文件
    last_intermediate_message: Optional[Dict[str, Any]] = None  # 记录上一条中间消息（errCode=3）

    async def on_final_callback(parsed: Dict[str, Any]) -> None:
        """当接收到一句完整的话时触发"""
        nonlocal segment_index, last_intermediate_message

        result_text = parsed.get("result_text", "").strip()
        trans_text = parsed.get("trans_text", "").strip()
        raw_data = parsed.get("raw", {}) # 这里是把原始消息保存下来，方便后续使用

        # 提取时间戳（单位：毫秒），确保转换为整数
        begin_ms = int(raw_data.get("Begin", 0) or 0)
        end_ms = int(raw_data.get("End", 0) or 0)

        # 临时方案：如果当前是流结束消息（errCode=10）且时间戳为0，使用上一条中间消息的时间戳
        if parsed.get("is_stream_finished") and begin_ms == 0 and end_ms == 0:
            if last_intermediate_message:
                last_raw = last_intermediate_message.get("raw", {})
                begin_ms = int(last_raw.get("Begin", 0) or 0)
                end_ms = int(last_raw.get("End", 0) or 0)
                logger.info("[句 %d] 使用上一条消息的时间戳（临时方案）: %d ms - %d ms", segment_index + 1, begin_ms, end_ms)
            else:
                logger.warning("[句 %d] 流结束消息时间戳为0，且无上一条消息可用", segment_index + 1)

        logger.info("[句 %d] ASR: %s", segment_index + 1, result_text)
        logger.info("[句 %d] 翻译: %s", segment_index + 1, trans_text)
        logger.info("[句 %d] 时间戳: %d ms - %d ms", segment_index + 1, begin_ms, end_ms)

        tts_text = _select_tts_text(result_text, trans_text, args.tts_text_source, logger)
        if not tts_text:
            logger.warning("[句 %d] 跳过合成（文本为空）", segment_index + 1)
            return

        # 根据时间戳切分原始音频作为 prompt_wav
        prompt_wav_path = None
        prompt_text_for_tts = None

        if begin_ms >= 0 and end_ms > begin_ms and result_text:
            # 使用音频工具切分音频片段（原始音频已重采样为16k，这里需要重采样到目标采样率）
            audio_segment = segment_audio_by_timestamp(
                original_audio,
                sample_rate=target_sr,
                begin_ms=begin_ms,
                end_ms=end_ms,
                target_sr=sample_rate,
            )

            # 保存为临时文件
            temp_prompt_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav", dir=segment_output_dir
            )
            temp_prompt_file.close()
            temp_prompt_path = Path(temp_prompt_file.name)
            sf.write(str(temp_prompt_path), audio_segment, sample_rate)
            temp_prompt_files.append(temp_prompt_path)

            prompt_wav_path = str(temp_prompt_path)
            prompt_text_for_tts = result_text  # 使用 ASR 识别结果作为 prompt_text

            logger.info(
                "[句 %d] 已切分音频片段作为 prompt (%.2f 秒)",
                segment_index + 1,
                len(audio_segment) / float(sample_rate),
            )
        else:
            # 如果没有有效时间戳，使用用户提供的 prompt（如果存在）
            if args.prompt_wav:
                prompt_wav_path = args.prompt_wav
                prompt_text_for_tts = args.prompt_text or result_text
                logger.info("[句 %d] 使用用户提供的 prompt_wav", segment_index + 1)
            else:
                logger.info("[句 %d] 无 prompt_wav，使用默认合成", segment_index + 1)

        segment_file = segment_output_dir / f"segment_{segment_index:03d}.wav"
        logger.info("[句 %d] 开始合成，保存到: %s", segment_index + 1, segment_file)

        try:
            audio = tts.generate(
                text=tts_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text_for_tts,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise and prompt_wav_path is not None,
            )

            sf.write(str(segment_file), audio, sample_rate)
            segment_files.append(segment_file)
            duration = len(audio) / float(sample_rate)
            logger.info("[句 %d] 合成完成，时长 %.2f 秒", segment_index + 1, duration)
            segment_index += 1
        except Exception as e:
            logger.error("[句 %d] 合成失败: %s", segment_index + 1, e, exc_info=True)

    async def on_intermediate_callback(parsed: Dict[str, Any]) -> None:
        """中间结果（可选，用于调试）"""
        nonlocal last_intermediate_message
        result_text = parsed.get("result_text", "").strip()
        trans_text = parsed.get("trans_text", "").strip()
        if result_text or trans_text:
            logger.debug("[中间结果] ASR: %s, 翻译: %s", result_text, trans_text)
        # 记录上一条中间消息，用于流结束时的时间戳回退
        last_intermediate_message = parsed

    # 执行流式识别+翻译
    await translator.transcribe_streaming(
        audio_path=str(audio_path),
        on_intermediate=on_intermediate_callback,
        on_final=on_final_callback,
    )

    if not segment_files:
        logger.error("未生成任何音频片段")
        return None

    logger.info("流式识别完成，共 %d 个片段，开始拼接", len(segment_files))

    # 拼接所有片段
    audio_segments: List[np.ndarray] = []
    sample_rates: List[int] = []
    for seg_file in segment_files:
        audio, sr = sf.read(str(seg_file))
        audio_segments.append(audio)
        sample_rates.append(sr)

    # 使用音频工具拼接，统一采样率
    final_audio, final_sr = concatenate_audio(audio_segments, sample_rates, target_sr=sample_rate)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), final_audio, final_sr)

    duration = len(final_audio) / float(final_sr)
    logger.info("拼接完成，最终输出: %s，总时长 %.2f 秒", out_path, duration)
    logger.info("片段文件保存在: %s", segment_output_dir)

    # 清理临时 prompt 文件
    for temp_file in temp_prompt_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug("已删除临时文件: %s", temp_file)
        except Exception as e:
            logger.warning("删除临时文件失败 %s: %s", temp_file, e)

    return out_path


async def audio_stream_generator(
    sample_rate: int,
    chunk_duration: float,
    bit_rate: int = 16,
    device: Optional[int] = None,
    logger=None,
) -> AsyncIterator[bytes]:
    """
    从麦克风实时捕获音频并生成音频流。

    Args:
        sample_rate: 采样率
        chunk_duration: 每次读取的音频时长（秒）
        bit_rate: 位深度（默认 16）
        device: 音频设备 ID，None 表示使用默认设备
        logger: 日志记录器

    Yields:
        音频数据的字节块（与文件读取格式一致）

    Raises:
        ImportError: 如果 sounddevice 未安装或 PortAudio 库未找到
    """
    # 延迟导入 sounddevice，避免在模块级别导入时因 PortAudio 缺失而报错
    try:
        import sounddevice as sd
    except ImportError as e:
        raise ImportError(
            "需要安装 sounddevice 库：pip install sounddevice\n"
            "如果已安装但仍报错，可能需要安装 PortAudio 系统库：\n"
            "  - Windows: 通常 sounddevice 会自动处理\n"
            "  - Linux: sudo apt-get install portaudio19-dev (Debian/Ubuntu) 或 sudo yum install portaudio-devel (RHEL/CentOS)\n"
            "  - macOS: brew install portaudio"
        ) from e
    except OSError as e:
        if "PortAudio" in str(e):
            raise OSError(
                "PortAudio 库未找到。请安装 PortAudio 系统库：\n"
                "  - Windows: 通常 sounddevice 会自动处理，如仍有问题请检查安装\n"
                "  - Linux: sudo apt-get install portaudio19-dev (Debian/Ubuntu) 或 sudo yum install portaudio-devel (RHEL/CentOS)\n"
                "  - macOS: brew install portaudio\n"
                "安装后可能需要重新安装 sounddevice: pip install --force-reinstall sounddevice"
            ) from e
        raise

    # 计算每次读取的样本数和字节数（与 ASRTranslator 的 chunk_size 保持一致）
    bytes_per_second = sample_rate * bit_rate // 8
    chunk_size_bytes = int(bytes_per_second * chunk_duration)
    chunk_size_samples = chunk_size_bytes // (bit_rate // 8)  # 每个样本的字节数

    dtype = np.int16 if bit_rate == 16 else np.int32

    logger.info("开始从麦克风捕获音频，采样率: %d Hz, 位深度: %d bit, 设备: %s", sample_rate, bit_rate, device or "默认")

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype=dtype, device=device) as stream:
            logger.info("麦克风已就绪，开始录音（按 Ctrl+C 停止）...")
            while True:
                audio_chunk, overflowed = stream.read(chunk_size_samples)
                if overflowed:
                    logger.warning("音频缓冲区溢出")
                # 转换为字节，确保格式与文件读取一致
                audio_bytes = audio_chunk.tobytes()
                yield audio_bytes
    except KeyboardInterrupt:
        logger.info("收到停止信号，结束录音")
    except Exception as e:
        logger.error("录音过程中出错: %s", e, exc_info=True)
        raise


async def run_mic_streaming_pipeline(
    translator: ASRTranslator,
    tts,
    args: argparse.Namespace,
    logger,
) -> None:
    """
    麦克风实时输入流程：从麦克风实时捕获音频，逐句识别+翻译+TTS。
    """
    logger.info("开始麦克风实时输入流程")

    segment_output_dir = Path(args.segment_output_dir)
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    segment_files: List[Path] = []
    segment_index = 0
    sample_rate = getattr(tts.tts_model, "sample_rate", 16000)
    temp_prompt_files: List[Path] = []

    async def on_final_callback(parsed: Dict[str, Any]) -> None:
        """当接收到一句完整的话时触发"""
        nonlocal segment_index

        result_text = parsed.get("result_text", "").strip()
        trans_text = parsed.get("trans_text", "").strip()
        raw_data = parsed.get("raw", {})

        # 提取时间戳（单位：毫秒）
        begin_ms = int(raw_data.get("Begin", 0) or 0)
        end_ms = int(raw_data.get("End", 0) or 0)

        logger.info("[句 %d] ASR: %s", segment_index + 1, result_text)
        logger.info("[句 %d] 翻译: %s", segment_index + 1, trans_text)
        logger.info("[句 %d] 时间戳: %d ms - %d ms", segment_index + 1, begin_ms, end_ms)

        tts_text = _select_tts_text(result_text, trans_text, args.tts_text_source, logger)
        if not tts_text:
            logger.warning("[句 %d] 跳过合成（文本为空）", segment_index + 1)
            return

        # 对于麦克风输入，如果没有时间戳，使用用户提供的 prompt（如果存在）
        prompt_wav_path = None
        prompt_text_for_tts = None

        if args.prompt_wav:
            prompt_wav_path = args.prompt_wav
            prompt_text_for_tts = args.prompt_text or result_text
            logger.info("[句 %d] 使用用户提供的 prompt_wav", segment_index + 1)
        else:
            logger.info("[句 %d] 无 prompt_wav，使用默认合成", segment_index + 1)

        segment_file = segment_output_dir / f"segment_{segment_index:03d}.wav"
        logger.info("[句 %d] 开始合成，保存到: %s", segment_index + 1, segment_file)

        try:
            audio = tts.generate(
                text=tts_text,
                prompt_wav_path=prompt_wav_path,
                prompt_text=prompt_text_for_tts,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise and prompt_wav_path is not None,
            )

            sf.write(str(segment_file), audio, sample_rate)
            segment_files.append(segment_file)
            duration = len(audio) / float(sample_rate)
            logger.info("[句 %d] 合成完成，时长 %.2f 秒", segment_index + 1, duration)
            segment_index += 1
        except Exception as e:
            logger.error("[句 %d] 合成失败: %s", segment_index + 1, e, exc_info=True)

    async def on_intermediate_callback(parsed: Dict[str, Any]) -> None:
        """中间结果（可选，用于调试）"""
        result_text = parsed.get("result_text", "").strip()
        trans_text = parsed.get("trans_text", "").strip()
        if result_text or trans_text:
            logger.debug("[中间结果] ASR: %s, 翻译: %s", result_text, trans_text)

    # 创建音频流生成器
    audio_stream = audio_stream_generator(
        sample_rate=args.sample_rate,
        chunk_duration=args.interval,
        bit_rate=args.bit_rate,
        device=args.device,
        logger=logger,
    )

    # 执行流式识别+翻译
    try:
        await translator.transcribe_streaming(
            audio_stream=audio_stream,
            on_intermediate=on_intermediate_callback,
            on_final=on_final_callback,
        )
    except KeyboardInterrupt:
        logger.info("用户中断，停止处理")

    if segment_files:
        logger.info("流式识别完成，共 %d 个片段", len(segment_files))
        logger.info("片段文件保存在: %s", segment_output_dir)
    else:
        logger.warning("未生成任何音频片段")

    # 清理临时 prompt 文件
    for temp_file in temp_prompt_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug("已删除临时文件: %s", temp_file)
        except Exception as e:
            logger.warning("删除临时文件失败 %s: %s", temp_file, e)


async def main_async(args: argparse.Namespace) -> None:
    logger = get_test_logger(__file__)

    # 初始化 ASR 翻译器
    translator = ASRTranslator(
        ws_url=args.ws_url,
        user_id=args.user_id,
        token=args.token,
        from_lang=args.from_lang,
        to_lang=args.to_lang,
        role=args.role,
        lan_id=args.lan_id,
        sample_rate=args.sample_rate,
        bit_rate=args.bit_rate,
        interval=args.interval,
        logger=logger,
    )

    # 加载 VoxCPM 模型（两种模式都需要）
    logger.info("加载 VoxCPM 模型...")
    tts = load_voxcpm(
        model_path=args.model_path,
        hf_model_id=args.hf_model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        no_denoiser=args.no_denoiser,
        no_optimize=args.no_optimize,
    )
    logger.info("VoxCPM 模型加载完成")

    # 根据输入模式选择流程
    if args.input_mode == "mic":
        # 麦克风实时输入模式
        await run_mic_streaming_pipeline(translator, tts, args, logger)
    else:
        # 文件输入模式
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error("输入音频不存在: %s", audio_path)
            return

        logger.info("输入音频: %s", audio_path)

        # 根据参数选择流式或非流式流程
        if args.streaming:
            await run_streaming_pipeline(audio_path, translator, tts, args, logger)
        else:
            await run_non_streaming_pipeline(audio_path, translator, tts, args, logger)


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()



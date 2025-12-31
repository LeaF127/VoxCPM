import argparse
import asyncio
import logging
import os
import tempfile
import time
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
from timing_stats import TimingStats, TimingStatsCollector


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
    parser.add_argument("--from-lang", default=os.getenv("FROM_LAN", "zh"))
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式：输出所有时间点和计算的指标（仅流式模式）",
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
    # 初始化时间统计，记录流程开始时间作为基准
    pipeline_start_time = time.perf_counter()
    stats = TimingStats()
    
    # 记录 T0: 用户音频采集完成（文件加载完成），使用相对时间
    stats.t0 = 0.0  # T0 作为基准时间点，设为 0
    
    # 计算输入音频时长
    try:
        audio_data, sr = load_audio(audio_path, target_sr=16000, mono=True)
        stats.input_audio_duration = len(audio_data) / float(sr)
        logger.info("输入音频时长: %.2f 秒", stats.input_audio_duration)
    except Exception as e:
        logger.warning("无法计算输入音频时长: %s", e)

    logger.info("开始非流式识别+翻译流程")

    # 将 timing_stats 和基准时间传递给 translator
    translator.timing_stats = stats
    translator.global_t0 = pipeline_start_time  # 传递全局基准时间

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

    # 记录 T3: TTS 开始合成，使用相对时间
    stats.t3 = time.perf_counter() - pipeline_start_time

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

    # 记录 T4: 最后一帧语音播放完成（TTS 合成完成），使用相对时间
    stats.t4 = time.perf_counter() - pipeline_start_time

    # 读取输出音频，计算时长
    try:
        output_audio, output_sr = sf.read(str(out_path))
        stats.output_audio_duration = len(output_audio) / float(output_sr)
        logger.info("输出音频时长: %.2f 秒", stats.output_audio_duration)
    except Exception as e:
        logger.warning("无法计算输出音频时长: %s", e)

    logger.info("端到端流程完成，最终输出音频: %s", out_path)
    
    # 输出性能报告
    logger.info("\n" + stats.format_report())
    
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
    # 初始化时间统计收集器，记录流程开始时间作为基准
    stats_collector = TimingStatsCollector()
    stats_collector.start_global() # 开始全局计时（T0）
    stats = stats_collector.start_segment() # 开始新的一段统计，并设置 T0
    translator.timing_stats = stats # 传递当前句子的时间统计
    translator.global_t0 = stats_collector.global_t0 # 传递全局基准时间

    # 初始化流式处理所需的变量
    segment_files: List[Path] = [] # 保存音频片段的列表
    segment_output_dir = Path(args.segment_output_dir)
    segment_index = 0 # 音频片段索引
    sample_rate = getattr(tts.tts_model, "sample_rate", 16000) # 采样率
    temp_prompt_files: List[Path] = []  # 用于清理临时文件
    last_intermediate_message: Optional[Dict[str, Any]] = None  # 记录上一条中间消息（errCode=3）

    # 读取音频文件
    original_audio, target_sr = load_audio(audio_path, target_sr=sample_rate, mono=True)
    
    # 定义 on_final_callback 回调函数
    async def on_final_callback(parsed: Dict[str, Any]) -> None:
        """当接收到一句完整的话时触发"""
        # 更新局部变量
        nonlocal segment_index, last_intermediate_message
        
        # 从 translator 获取当前句子的 stats（这是真正被 ASR 记录时间点的对象）
        current_stats = translator.timing_stats
        if current_stats is None:
            logger.warning("[句 %d] 当前 stats 为 None，跳过统计", segment_index + 1)
            return

        # 输出当前记录的chunks数
        logger.info("[句 %d] 当前记录的chunks数: %d", segment_index + 1, current_stats.chunks_sent)
        
        result_text = parsed.get("result_text", "").strip()
        trans_text = parsed.get("trans_text", "").strip()
        raw_data = parsed.get("raw", {}) # 这里是把原始消息保存下来，方便后续使用

        # 提取时间戳（单位：毫秒），确保转换为整数，并计算单句时长
        begin_ms = int(raw_data.get("Begin", 0) or 0)
        end_ms = int(raw_data.get("End", 0) or 0)
        # 修复计算错误：应该是 (end_ms - begin_ms) / 1000.0
        current_stats.input_audio_duration = (end_ms - begin_ms) / 1000.0 if end_ms > begin_ms else None

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
            # 完成当前段统计（即使没有合成，也要保存 ASR 的统计信息）
            stats_collector.finish_segment()
            # 开始新的一段统计
            new_stats = stats_collector.start_segment()
            translator.timing_stats = new_stats
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
            # 记录 T3: TTS 开始合成，使用相对时间（基于 global_t0）
            if stats_collector.global_t0 is not None:
                current_stats.t3 = time.perf_counter() - stats_collector.global_t0
            
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
            current_stats.output_audio_duration = duration
            
            # 记录 T4: 最后一帧语音播放完成（音频保存完成），使用相对时间（基于 global_t0）
            if stats_collector.global_t0 is not None:
                current_stats.t4 = time.perf_counter() - stats_collector.global_t0
            
            logger.info("[句 %d] 合成完成，时长 %.2f 秒", segment_index + 1, duration)
            segment_index += 1
            
            # 完成当前段统计
            stats_collector.finish_segment()
            # 开始新的一段统计
            new_stats = stats_collector.start_segment()
            translator.timing_stats = new_stats
        except Exception as e:
            logger.error("[句 %d] 合成失败: %s", segment_index + 1, e, exc_info=True)
            # 即使合成失败，也要保存当前段的统计信息
            stats_collector.finish_segment()
            # 开始新的一段统计
            new_stats = stats_collector.start_segment()
            translator.timing_stats = new_stats

    # 定义 on_intermediate_callback 回调函数
    async def on_intermediate_callback(parsed: Dict[str, Any]) -> None:
        """中间结果（可选，用于调试）"""
        nonlocal last_intermediate_message
        
        # 更新局部变量
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
    
    # 输出性能报告
    logger.info("\n" + stats_collector.format_summary_report(debug=args.debug))

    # 清理临时 prompt 文件
    for temp_file in temp_prompt_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
                logger.debug("已删除临时文件: %s", temp_file)
        except Exception as e:
            logger.warning("删除临时文件失败 %s: %s", temp_file, e)

    return out_path


async def main_async(args: argparse.Namespace) -> None:
    logger = get_test_logger(__file__)
    logger.setLevel(logging.INFO)

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
    start_time = time.perf_counter()
    tts = load_voxcpm(
        model_path=args.model_path,
        hf_model_id=args.hf_model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        no_denoiser=args.no_denoiser,
        no_optimize=args.no_optimize,
    )
    logger.info('='*80)
    logger.info("VoxCPM 模型加载完成，耗时 %.2f 秒", time.perf_counter() - start_time)
    logger.info('='*80)

    # 根据输入模式选择流程
    if args.input_mode == "mic":
        # 麦克风实时输入模式
        # await run_mic_streaming_pipeline(translator, tts, args, logger)
        logger.warning("麦克风实时输入模式暂未实现")
        pass
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



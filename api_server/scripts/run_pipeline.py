"""
端到端 S2ST Pipeline CLI 入口。

运行语音识别、翻译和合成的完整流程。
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from s2st_demo.clients.asr_websocket import ASRWebSocketClient
from s2st_demo.models.tts_wrapper import TTSModelWrapper
from s2st_demo.services.s2st_service import S2STService
from s2st_demo.utils import get_test_logger
from s2st_demo.utils.timing_stats import TimingStats, TimingStatsCollector

# 设置环境变量
if os.environ.get("PYTHONPATH", "") == "":
    os.environ["PYTHONPATH"] = "./s2st_demo"
else:
    os.environ["PYTHONPATH"] = "./s2st_demo:" + os.environ["PYTHONPATH"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="端到端：识别 + 翻译 + VoxCPM 合成 Demo")

    # 输入音频
    parser.add_argument(
        "--audio",
        default="s2st_demo/input/example.wav",
        help="待识别的音频路径（建议 16kHz 单声道 WAV）",
    )

    # ASR / 翻译服务相关参数（必须通过参数或环境变量提供）
    parser.add_argument("--ws-url", default=os.getenv("WS_URL"), help="ASR WebSocket 服务地址（可通过环境变量 WS_URL 设置）")
    parser.add_argument("--user-id", default=os.getenv("USER_ID"), help="用户 ID（可通过环境变量 USER_ID 设置）")
    parser.add_argument("--token", default=os.getenv("TOKEN"), help="认证 Token（可通过环境变量 TOKEN 设置）")
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


async def main_async(args: argparse.Namespace) -> None:
    logger = get_test_logger(__file__)
    logger.setLevel(logging.INFO)

    # 验证必填参数
    if not args.ws_url:
        logger.error("--ws-url 参数或环境变量 WS_URL 必须提供")
        return
    if not args.user_id:
        logger.error("--user-id 参数或环境变量 USER_ID 必须提供")
        return
    if not args.token:
        logger.error("--token 参数或环境变量 TOKEN 必须提供")
        return

    # 初始化 ASR 翻译器
    translator = ASRWebSocketClient(
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

    # 加载 VoxCPM 模型
    import time
    start_time = time.perf_counter()
    tts_wrapper = TTSModelWrapper.load(
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

    # 创建服务实例
    service = S2STService(tts_wrapper.tts_model)

    # 根据输入模式选择流程
    if args.input_mode == "mic":
        logger.warning("麦克风实时输入模式暂未实现")
        return
    else:
        # 文件输入模式
        audio_path = Path(args.audio)
        if not audio_path.exists():
            logger.error("输入音频不存在: %s", audio_path)
            return

        logger.info("输入音频: %s", audio_path)

        # 根据参数选择流式或非流式流程
        if args.streaming:
            logger.warning("流式模式暂未在此脚本中实现，请使用 API 或原始 run.py")
            return
        else:
            result = await service.process_non_streaming(
                audio_path=audio_path,
                translator=translator,
                tts_text_source=args.tts_text_source,
                output_path=Path(args.output),
                prompt_wav=args.prompt_wav,
                prompt_text=args.prompt_text,
                cfg_value=args.cfg_value,
                inference_timesteps=args.inference_timesteps,
                normalize=args.normalize,
                denoise=args.denoise,
            )

            if result:
                logger.info("处理完成！输出文件: %s", result)
            else:
                logger.error("处理失败")


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

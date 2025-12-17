import argparse
import asyncio
import os
from pathlib import Path
from typing import Optional

from voxcpm.utils import get_test_logger

from s2st_demo.asr_translate import ASRTranslator
from s2st_demo.tts import load_voxcpm, synthesize


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

    return parser


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


async def main_async(args: argparse.Namespace) -> None:
    logger = get_test_logger(__file__)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("输入音频不存在: %s", audio_path)
        return

    logger.info("输入音频: %s", audio_path)
    logger.info("开始非流式识别+翻译流程")

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

    asr_result = await translator.run(audio_path=str(audio_path), streaming=False)
    if not asr_result:
        logger.error("未获得识别/翻译结果")
        return

    result_text = asr_result.get("result_text", "")
    trans_text = asr_result.get("trans_text", "")
    logger.info("ASR 结果: %s", result_text)
    logger.info("翻译结果: %s", trans_text)

    tts_text = _select_tts_text(result_text, trans_text, args.tts_text_source, logger)
    if not tts_text:
        return

    logger.info("即将进入 VoxCPM 合成阶段")

    tts = load_voxcpm(
        model_path=args.model_path,
        hf_model_id=args.hf_model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        no_denoiser=args.no_denoiser,
        no_optimize=args.no_optimize,
    )

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


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()



import argparse
import asyncio
import os
from pathlib import Path

if os.environ.get("PYTHONPATH", "") == "":
    os.environ["PYTHONPATH"] = "./s2st_demo"
else:
    os.environ["PYTHONPATH"] = "./s2st_demo:" + os.environ["PYTHONPATH"]
# print(os.environ["PYTHONPATH"])

from voxcpm.utils import get_test_logger
from asr_translate import ASRTranslator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试 ASRTranslator 非流式模式")
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
    parser.add_argument("--audio", default="examples/amiya.wav", help="待识别音频路径")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    logger = get_test_logger(__file__, console_output=True)

    audio_path = Path(args.audio)
    if not audio_path.exists():
        logger.error("音频文件不存在: %s", audio_path)
        return

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

    logger.info("开始非流式识别，音频: %s", audio_path)
    result = await translator.run(audio_path=str(audio_path), streaming=False)
    if result:
        logger.info("最终识别: %s", result.get("result_text", ""))
        logger.info("翻译结果: %s", result.get("trans_text", ""))
    else:
        logger.warning("未获得最终结果")


if __name__ == "__main__":
    asyncio.run(main())


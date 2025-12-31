import asyncio
from datetime import datetime
import logging


# ========== 日志配置 ==========
log_name = f"logs/stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_name,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# 文件输出
file_handler = logging.FileHandler(log_name, encoding='utf-8')

logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# ================ 配置区域 ================
import os

# 从环境变量读取配置，如果没有则使用默认值（仅用于开发测试）
WS_URL = os.getenv("WS_URL")  # 必须通过环境变量设置
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))  # 采样率
BIT_RATE = int(os.getenv("BIT_RATE", "16"))  # 位深
INTERVAL = float(os.getenv("INTERVAL", "0.1"))  # 发送间隔（秒）

from app import VoxCPMDemo
from tts_module import tts
from s2st_demo.asr_translate import ASRTranslator

USER_ID = os.getenv("USER_ID")  # 必须通过环境变量设置
TOKEN = os.getenv("TOKEN")  # 必须通过环境变量设置，格式：token-时间戳
AUDIO_PATH = os.getenv("AUDIO_PATH", "examples/amiya.wav")
LAN_ID = os.getenv("LAN_ID", "0")
FROM_LAN = os.getenv("FROM_LAN", "cn")
TO_LAN = os.getenv("TO_LAN", "en")
ROLE = os.getenv("ROLE", "0")

# ================ 模型配置 ================
tts_model = VoxCPMDemo()


async def main():
    # 验证必填参数
    if not WS_URL:
        logger.error("环境变量 WS_URL 必须设置")
        return
    if not USER_ID:
        logger.error("环境变量 USER_ID 必须设置")
        return
    if not TOKEN:
        logger.error("环境变量 TOKEN 必须设置")
        return

    translator = ASRTranslator(
        ws_url=WS_URL,
        user_id=USER_ID,
        token=TOKEN,
        from_lang=FROM_LAN,
        to_lang=TO_LAN,
        role=ROLE,
        lan_id=LAN_ID,
        sample_rate=SAMPLE_RATE,
        bit_rate=BIT_RATE,
        interval=INTERVAL,
        logger=logger,
    )

    async def on_final(parsed):
        """最终结果回调，调用 TTS。"""
        trans_text = parsed.get("trans_text", "")
        result_text = parsed.get("result_text", "")
        logger.info("最终识别: %s, 翻译: %s", result_text, trans_text)
        if trans_text:
            tts(trans_text)

    await translator.run(audio_path=AUDIO_PATH, streaming=True, on_final=on_final)


if __name__ == "__main__":
    asyncio.run(main())

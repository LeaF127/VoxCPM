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
WS_URL = "ws://175.24.179.12:9301/dotcwsasr"   # 你的WS接口URL
SAMPLE_RATE = 16000        # 采样率
BIT_RATE = 16              # 位深
INTERVAL = 0.1             # 发送间隔（秒）

from app import VoxCPMDemo
from tts_module import tts
from s2st_demo.asr_translate import ASRTranslator

USER_ID = "y123456"
TOKEN = "token12345-1730889600"             # 格式：token-时间戳
AUDIO_PATH = "examples/amiya.wav"
LAN_ID = "0"
FROM_LAN = "cn"
TO_LAN = "en"
ROLE = "0"

# ================ 模型配置 ================
tts_model = VoxCPMDemo()


async def main():
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

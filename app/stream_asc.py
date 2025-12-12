import asyncio
from datetime import datetime
import json
import logging
from torch import Tensor
import torch
import torchaudio
import websockets
import time


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
BYTES_PER_SECOND = SAMPLE_RATE * BIT_RATE // 8
CHUNK_SIZE = int(BYTES_PER_SECOND * INTERVAL)  # 每包 3200 字节

import uuid
SESSION_ID = str(uuid.uuid4())

USER_ID = "y123456"
TOKEN = "token12345-1730889600"             # 格式：token-时间戳
AUDIO_PATH = "1.wav"           # 要发送的音频文件
CHUNK_SIZE = 3200                           # 每次发送的音频字节数 (~0.1秒)

LAN_ID = "0"
FROM_LAN = "cn"
TO_LAN = "en"
ROLE = "0"

AUDIO_PATH = "examples/amiya.wav"

# ================ 模型配置 ================
from app import VoxCPMDemo
from tts_module import tts
tts_model = VoxCPMDemo()


async def process_audio_stream(audio_stream, websocket):
    """处理音频流，返回识别和翻译结果"""
    async for chunk in audio_stream:
        # 模拟发送音频数据到服务器
        result = "识别结果"  # 替换为实际识别逻辑
        await websocket.send_text(result)

async def on_message(message):
    """处理服务器返回的识别/翻译结果"""
    logger.info(f"Server Response: {message}")
    
    try:
        data = json.loads(message)
    except:
        logger.error("返回值不是 JSON")
        return

    err_code = str(data.get("errCode", ""))

    # 3 = 中间结果，等待
    if err_code == "3":
        return

    # 0 = 最终结果，执行 TTS
    if err_code == "0" or err_code == "10":
        final_text = data.get("result", "")
        trans_text = data.get("trans", "")
        logger.info(f"[Final ASR] {final_text}, [Trans] {trans_text}")
        tts(trans_text)   # 调用 TTS 占位函数


async def on_error(error):
    logger.info(f"WS Error: {error}")


async def on_close(close_status_code, close_msg):
    logger.info(f"WS Closed: code={close_status_code}, msg={close_msg}")


async def send_audio(ws):
    """替代同步版本 run() 的异步发送线程"""
    logger.info("[WS Connected]")

    fake_frame = b"\x00" * CHUNK_SIZE
    logger.info(f"Start streaming audio... Chunk = {CHUNK_SIZE} bytes")

    try:
        start_time = time.time()
        frame_index = 0

        with open(AUDIO_PATH, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break

                frame_index += 1
                logger.debug(f"发送第{frame_index}帧数据,({len(chunk)} bytes)")

                if frame_index % 100 == 0:
                    end_time = time.time()
                    logger.info(
                        f"发送第{frame_index}帧数据,({len(chunk)*frame_index} bytes), 耗时{end_time-start_time:.2f}秒"
                    )

                await ws.send(chunk)
                await asyncio.sleep(INTERVAL)

    except Exception as e:
        logger.error(f"Streaming stopped: {e}")
        await ws.close()

    # 发送结束命令
    await ws.send("end")
    end_time = time.time()
    logger.info(f"已发送结束命令, 共耗时{end_time-start_time:.2f}秒")


async def main():

    init_msg = {
        "userid": USER_ID,
        "lanid": LAN_ID,
        "token": TOKEN,
        "sid": SESSION_ID,
        "fromlan": FROM_LAN,
        "tolan": TO_LAN,
        "role": ROLE,
    }

    TARGET_URL = WS_URL + f"?lanid={LAN_ID}&userid={USER_ID}&token={TOKEN}&fromlan={FROM_LAN}&tolan={TO_LAN}&sid={SESSION_ID}&role={ROLE}"

    logger.info(f"Connecting to {TARGET_URL}")

    async with websockets.connect(TARGET_URL) as ws:

        # 如果需要发送 init_msg，取消注释即可
        # await ws.send(json.dumps(init_msg))
        # logger.info(f"已发送初始信息: {init_msg}")

        # 启动异步音频发送任务
        send_task = asyncio.create_task(send_audio(ws))

        # 接收服务器返回
        try:
            async for message in ws:
                await on_message(message)
        except Exception as e:
            await on_error(e)

        # 等待发送结束
        await send_task

        await on_close(None, None)


if __name__ == "__main__":
    asyncio.run(main())

import time
import logging
from app import stream_asc
import torch
import torchaudio
from torch import Tensor
import asyncio
from datetime import datetime
import json
import websockets
import uuid
import gradio as gr
from typing import Union
from voxcpm.model.voxcpm import VoxCPMModel, VoxCPMConfig  # 导入模型
from funasr import AutoModel


# ========== 日志配置 ==========
log_name = f"logs/stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_name,
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
file_handler = logging.FileHandler(log_name, encoding='utf-8')
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# ================ 配置区域 ================
WS_URL = "ws://175.24.179.12:9301/dotcwsasr"
SAMPLE_RATE = 16000
BIT_RATE = 16
INTERVAL = 0.1
BYTES_PER_SECOND = SAMPLE_RATE * BIT_RATE // 8
CHUNK_SIZE = int(BYTES_PER_SECOND * INTERVAL)

SESSION_ID = str(uuid.uuid4())
USER_ID = "y123456"
TOKEN = "token12345-1730889600"
AUDIO_PATH = "examples/amiya.wav"
LAN_ID = "0"
FROM_LAN = "cn"
TO_LAN = "en"
ROLE = "0"

# ================ 模型配置 ================
# ASR model for prompt text recognition
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    disable_update=True,
    log_level='DEBUG',
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)


# 初始化 VoxCPM 模型
config_path = "models/openbmb__VoxCPM-0.5B/config.json"  # 替换为实际配置文件路径
model_path = "models/openbmb__VoxCPM-0.5B"  # 替换为实际模型路径
config = VoxCPMConfig.model_validate_json(open(config_path).read())
tts_model = VoxCPMModel.from_local(model_path)

def recognize_prompt_text(prompt_wav: str) -> str:
    if prompt_wav is None:
        return ""
    res = asr_model.generate(input=prompt_wav, language="auto", use_itn=True)
    text = res[0]["text"].split('|>')[-1]
    return text

def tts(prompt_wav: Tensor = None, 
        prompt_text: str = None, 
        trans_text: str = None, 
        patch_idx: Union[str, int] = 0, 
        save: bool = True):
    
    if prompt_wav is None:
        raise ValueError("prompt_wav 不能为空")
    if prompt_text is None:
        prompt_text = recognize_prompt_text(prompt_wav)
    if trans_text is None:
        raise ValueError("trans_text 不能为空")
    
    start_time = time.time()
    logger.info(f"[TTS] {trans_text}")
    # 调用 VoxCPMModel 生成音频
    generated_audio = tts_model.generate(
        target_text=trans_text,
        prompt_text=prompt_text,
        prompt_wav_path=prompt_wav,
    )
    sr = tts_model.sample_rate
    wav = generated_audio.numpy()

    if save:
        save_path = f"outputs/patch_{patch_idx}.wav"
        import os
        os.makedirs("outputs", exist_ok=True)
        logger.info(f"[TTS] 已保存至{save_path}")
        torchaudio.save(save_path, torch.from_numpy(wav), sample_rate=sr)
    
    end_time = time.time()
    logger.info(f"[TTS] 耗时{end_time-start_time:.2f}秒")
    return sr, wav


async def process_audio_stream(audio_stream, websocket):
    async for chunk in audio_stream:
        result = "识别结果"
        await websocket.send_text(result)

async def on_message(message):
    logger.info(f"Server Response: {message}")
    try:
        data = json.loads(message)
    except:
        logger.error("返回值不是 JSON")
        return

    err_code = str(data.get("errCode", ""))
    if err_code == "3":
        return
    if err_code == "0" or err_code == "10":
        final_text = data.get("result", "")
        trans_text = data.get("trans", "")
        logger.info(f"[Final ASR] {final_text}, [Trans] {trans_text}")
        tts(trans_text)

async def on_error(error):
    logger.info(f"WS Error: {error}")

async def on_close(close_status_code, close_msg):
    logger.info(f"WS Closed: code={close_status_code}, msg={close_msg}")

async def send_audio(ws):
    logger.info("[WS Connected]")
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
    logger.info(f"Init Message: {init_msg}")
    async with websockets.connect(TARGET_URL) as ws:
        send_task = asyncio.create_task(send_audio(ws))
        try:
            async for message in ws:
                await on_message(message)
        except Exception as e:
            await on_error(e)
        await send_task
        await on_close(None, None)

async def process_audio(audio):
    logger.info("开始处理音频文件")
    result = "识别结果: 示例文本"
    trans_result = "翻译结果: Example text"
    logger.info(f"识别结果: {result}, 翻译结果: {trans_result}")
    _, generated_audio = tts(prompt_wav=audio,trans_text=trans_result)
    return result, trans_result, generated_audio

def handle_audio(audio):
    logger.info("开始处理音频文件")
    
    
    # 自动识别prompt_text
    prompt_text = recognize_prompt_text(audio)
    
    # 使用WebSocket获取识别和翻译结果
    async def get_results():
        init_msg = {
            "userid": USER_ID,
            "lanid": LAN_ID,
            "token": TOKEN,
            "sid": str(uuid.uuid4()),
            "fromlan": FROM_LAN,
            "tolan": TO_LAN,
            "role": ROLE,
        }
        TARGET_URL = WS_URL + f"?lanid={LAN_ID}&userid={USER_ID}&token={TOKEN}&fromlan={FROM_LAN}&tolan={TO_LAN}&sid={init_msg['sid']}&role={ROLE}"
        
        async with websockets.connect(TARGET_URL) as ws:
            # 发送音频
            with open(audio, "rb") as f:
                while True:
                    chunk = f.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    await ws.send(chunk)
                    await asyncio.sleep(INTERVAL)
            await ws.send("end")
            
            # 接收结果
            final_result = ""
            final_trans = ""
            async for message in ws:
                try:
                    data = json.loads(message)
                    err_code = str(data.get("errCode", ""))
                    if err_code == "0" or err_code == "10":
                        final_result = data.get("result", "")
                        final_trans = data.get("trans", "")
                        break
                except:
                    continue
            return final_result, final_trans

    # 运行异步函数获取结果
    result, trans_result = asyncio.run(get_results())
    
    # 生成TTS音频
    sr, wav = tts(prompt_wav=audio, trans_text=trans_result, prompt_text=prompt_text)
    
    # 保存为临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_tensor = torch.from_numpy(wav)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        torchaudio.save(tmp.name, wav_tensor, sample_rate=sr)
        return result, trans_result, tmp.name


    # 运行异步函数获取结果
    result, trans_result = asyncio.run(get_results())
    
    # 生成TTS音频
    sr, wav = tts(prompt_wav=audio, trans_text=trans_result, prompt_text=prompt_text)
    
    # 保存为临时文件
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_tensor = torch.from_numpy(wav)
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)
        torchaudio.save(tmp.name, wav_tensor, sample_rate=sr)
        return result, trans_result, tmp.name




with gr.Blocks() as demo:
    gr.Markdown("# 同声传译 Demo")
    with gr.Row():
        # audio_input = gr.Audio(sources=["upload","microphone"], type="filepath", label="录音输入", streaming=True)
        audio_input = gr.Audio(sources="microphone", type="filepath", label="录音输入", streaming=True)
        with gr.Column():
            result_text = gr.Textbox(label="识别结果", interactive=False)
            trans_text = gr.Textbox(label="翻译结果", interactive=False)
    audio_output = gr.Audio(label="TTS输出", autoplay=True, streaming=True)  # 添加 autoplay=True
    submit_button = gr.Button("开始处理")
    
    # 修改回调函数，使用 gr.Progress 显示进度
    def process_and_play(audio):
        if audio is None:
            return "请先上传或录制音频", "请先上传或录制音频", None
            
        # 处理音频
        result, trans, audio_path = handle_audio(audio)
        
        # 直接返回结果，Gradio会自动更新组件
        return result, trans, audio_path

    
    submit_button.click(
        fn=process_and_play,
        inputs=[audio_input],
        outputs=[result_text, trans_text, audio_output],
        show_progress=True
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)
"""
FastAPI 服务器：将流式 S2ST pipeline 包装为 REST API。

支持：
- 文件上传（multipart/form-data）
- 流式响应（Server-Sent Events）
- 实时返回识别、翻译和合成结果
"""

import asyncio
import os
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voxcpm.utils import get_test_logger, load_audio, segment_audio_by_timestamp, concatenate_audio

if os.environ.get("PYTHONPATH", "") == "":
    os.environ["PYTHONPATH"] = "./s2st_demo"
else:
    os.environ["PYTHONPATH"] = "./s2st_demo:" + os.environ["PYTHONPATH"]

from asr_translate import ASRTranslator
from tts import load_voxcpm

# 全局模型实例（懒加载）
_tts_model: Optional[Any] = None
_translator: Optional[ASRTranslator] = None
_logger = get_test_logger(__file__)

app = FastAPI(
    title="VoxCPM S2ST API",
    description="语音识别、翻译与语音合成流式 API",
    version="0.1.0",
)

# 允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ========== 请求/响应模型 ==========


class StreamingRequest(BaseModel):
    """流式处理请求参数（JSON body）"""

    ws_url: str = Field(default="ws://175.24.179.12:9301/dotcwsasr", description="ASR WebSocket 服务地址")
    user_id: str = Field(default="y123456", description="用户 ID")
    token: str = Field(default="token12345-1730889600", description="认证 Token")
    from_lang: str = Field(default="zh", description="源语言代码")
    to_lang: str = Field(default="en", description="目标语言代码")
    role: str = Field(default="0", description="角色 ID")
    lan_id: str = Field(default="0", description="语言 ID")
    sample_rate: int = Field(default=16000, description="音频采样率")
    bit_rate: int = Field(default=16, description="音频位深度")
    interval: float = Field(default=0.1, description="音频块间隔（秒）")
    tts_text_source: str = Field(default="trans", description="TTS 文本来源：trans=翻译结果，asr=识别结果")
    cfg_value: float = Field(default=2.0, description="VoxCPM CFG 值")
    inference_timesteps: int = Field(default=10, description="VoxCPM 扩散步数")
    normalize: bool = Field(default=True, description="启用文本正则化")
    denoise: bool = Field(default=False, description="对 prompt 音频降噪")
    prompt_wav_path: Optional[str] = Field(default=None, description="参考音频路径（可选）")
    prompt_text: Optional[str] = Field(default=None, description="参考音频对应文本（可选）")


class SegmentResult(BaseModel):
    """单句处理结果"""

    segment_index: int = Field(description="片段索引（从 0 开始）")
    asr_text: str = Field(description="识别文本")
    trans_text: str = Field(description="翻译文本")
    tts_text: str = Field(description="用于合成的文本")
    begin_ms: int = Field(description="开始时间戳（毫秒）")
    end_ms: int = Field(description="结束时间戳（毫秒）")
    segment_file: Optional[str] = Field(default=None, description="片段音频文件路径（相对路径）")
    duration: Optional[float] = Field(default=None, description="片段时长（秒）")


class StreamingResponseModel(BaseModel):
    """流式响应数据模型"""

    event_type: str = Field(description="事件类型：segment=单句完成，progress=进度，error=错误，complete=全部完成")
    data: Dict[str, Any] = Field(description="事件数据")


# ========== 模型初始化 ==========


def get_tts_model(model_path: Optional[str] = None, hf_model_id: Optional[str] = None) -> Any:
    """获取或初始化 TTS 模型（单例）"""
    global _tts_model
    if _tts_model is None:
        _logger.info("初始化 VoxCPM 模型...")
        _tts_model = load_voxcpm(
            model_path=model_path or os.getenv("VOXCPM_MODEL_PATH"),
            hf_model_id=hf_model_id or os.getenv("VOXCPM_HF_ID", "openbmb/VoxCPM-0.5B"),
            cache_dir=os.getenv("VOXCPM_CACHE_DIR"),
            local_files_only=os.getenv("VOXCPM_LOCAL_FILES_ONLY", "false").lower() == "true",
            no_denoiser=os.getenv("VOXCPM_NO_DENOISER", "false").lower() == "true",
            no_optimize=os.getenv("VOXCPM_NO_OPTIMIZE", "false").lower() == "true",
        )
        _logger.info("VoxCPM 模型初始化完成")
    return _tts_model


def get_translator(request: StreamingRequest) -> ASRTranslator:
    """创建 ASR 翻译器实例"""
    return ASRTranslator(
        ws_url=request.ws_url,
        user_id=request.user_id,
        token=request.token,
        from_lang=request.from_lang,
        to_lang=request.to_lang,
        role=request.role,
        lan_id=request.lan_id,
        sample_rate=request.sample_rate,
        bit_rate=request.bit_rate,
        interval=request.interval,
        logger=_logger,
    )


# ========== 工具函数 ==========


def _select_tts_text(result_text: str, trans_text: str, source: str) -> Optional[str]:
    """根据配置选择用于 TTS 的文本"""
    asr_text = (result_text or "").strip()
    mt_text = (trans_text or "").strip()

    if source == "trans":
        return mt_text if mt_text else asr_text
    else:
        return asr_text if asr_text else mt_text


# ========== API 端点 ==========


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "VoxCPM S2ST API",
        "version": "0.1.0",
        "description": "语音识别、翻译与语音合成流式 API",
        "endpoints": {
            "/streaming": "POST - 流式处理音频文件（Server-Sent Events）",
            "/health": "GET - 健康检查",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "tts_model_loaded": _tts_model is not None}


@app.post("/streaming")
async def streaming_s2st(
    audio_file: UploadFile = File(..., description="音频文件（建议 16kHz 单声道 WAV）"),
    ws_url: str = Form(default="ws://175.24.179.12:9301/dotcwsasr"),
    user_id: str = Form(default="y123456"),
    token: str = Form(default="token12345-1730889600"),
    from_lang: str = Form(default="zh"),
    to_lang: str = Form(default="en"),
    role: str = Form(default="0"),
    lan_id: str = Form(default="0"),
    sample_rate: int = Form(default=16000),
    bit_rate: int = Form(default=16),
    interval: float = Form(default=0.1),
    tts_text_source: str = Form(default="trans"),
    cfg_value: float = Form(default=2.0),
    inference_timesteps: int = Form(default=10),
    normalize: bool = Form(default=True),
    denoise: bool = Form(default=False),
    prompt_wav_path: Optional[str] = Form(default=None),
    prompt_text: Optional[str] = Form(default=None),
):
    """
    流式 S2ST 处理：上传音频文件，实时返回识别、翻译和合成结果。

    使用 Server-Sent Events (SSE) 流式返回结果。
    每完成一句识别+翻译+TTS，立即发送事件；全部完成后发送最终拼接的音频文件路径。
    """
    request_id = str(uuid.uuid4())[:8]
    _logger.info("[%s] 收到流式处理请求，文件名: %s", request_id, audio_file.filename)

    # 创建请求对象
    request = StreamingRequest(
        ws_url=ws_url,
        user_id=user_id,
        token=token,
        from_lang=from_lang,
        to_lang=to_lang,
        role=role,
        lan_id=lan_id,
        sample_rate=sample_rate,
        bit_rate=bit_rate,
        interval=interval,
        tts_text_source=tts_text_source,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=denoise,
        prompt_wav_path=prompt_wav_path,
        prompt_text=prompt_text,
    )

    # 保存上传的音频文件
    temp_audio_dir = Path("s2st_demo/output/temp")
    temp_audio_dir.mkdir(parents=True, exist_ok=True)
    temp_audio_path = temp_audio_dir / f"{request_id}_{audio_file.filename}"

    try:
        with open(temp_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        _logger.info("[%s] 音频文件已保存: %s", request_id, temp_audio_path)
    except Exception as e:
        _logger.error("[%s] 保存音频文件失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"保存音频文件失败: {e}")

    # 初始化模型
    try:
        tts = get_tts_model(model_path="./models/openbmb__VoxCPM-0.5B/")
        translator = get_translator(request)
    except Exception as e:
        _logger.error("[%s] 模型初始化失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {e}")

    # 创建输出目录
    segment_output_dir = Path(f"s2st_demo/output/segments/{request_id}")
    segment_output_dir.mkdir(parents=True, exist_ok=True)

    # 加载原始音频用于切分
    target_sr = 16000
    try:
        original_audio, original_sr = load_audio(temp_audio_path, target_sr=target_sr, mono=True)
        _logger.info("[%s] 原始音频已加载，采样率: %d Hz, 时长: %.2f 秒", request_id, original_sr, len(original_audio) / original_sr)
    except Exception as e:
        _logger.error("[%s] 加载音频失败: %s", request_id, e)
        raise HTTPException(status_code=400, detail=f"加载音频失败: {e}")

    # 流式处理状态
    segment_files: List[Path] = []
    segment_index = 0
    tts_sample_rate = getattr(tts.tts_model, "sample_rate", 16000)
    temp_prompt_files: List[Path] = []
    last_intermediate_message: Optional[Dict[str, Any]] = None
    event_queue: asyncio.Queue = asyncio.Queue()
    processing_complete = False

    async def generate_events():
        """生成 SSE 事件流"""
        nonlocal segment_index, last_intermediate_message, processing_complete

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

            tts_text = _select_tts_text(result_text, trans_text, request.tts_text_source)
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
                    _logger.warning("[%s] 切分音频失败: %s", request_id, e)

            if not prompt_wav_path and request.prompt_wav_path:
                prompt_wav_path = request.prompt_wav_path
                prompt_text_for_tts = request.prompt_text or result_text

            # 合成语音
            segment_file = segment_output_dir / f"segment_{segment_index:03d}.wav"
            try:
                audio = tts.generate(
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
                _logger.error("[%s] [句 %d] 合成失败: %s", request_id, segment_index + 1, e, exc_info=True)
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
                    audio_path=str(temp_audio_path),
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
                    _logger.error("[%s] 拼接音频失败: %s", request_id, e, exc_info=True)
                    await event_queue.put(
                        StreamingResponseModel(
                            event_type="error", data={"message": f"拼接音频失败: {e}"}
                        )
                    )
                finally:
                    # 清理临时文件
                    try:
                        if temp_audio_path.exists():
                            temp_audio_path.unlink()
                        for temp_file in temp_prompt_files:
                            if temp_file.exists():
                                temp_file.unlink()
                    except Exception as e:
                        _logger.warning("[%s] 清理临时文件失败: %s", request_id, e)
                    processing_complete = True
            except Exception as e:
                _logger.error("[%s] 流式处理失败: %s", request_id, e, exc_info=True)
                await event_queue.put(
                    StreamingResponseModel(event_type="error", data={"message": f"流式处理失败: {e}"})
                )
                processing_complete = True

        # 启动后台处理任务
        asyncio.create_task(process_audio())

        # 从队列中读取事件并 yield
        while not processing_complete or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                yield f"data: {event.model_dump_json()}\n\n"
            except asyncio.TimeoutError:
                continue

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


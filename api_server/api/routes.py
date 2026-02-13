"""
FastAPI 路由定义。

定义所有 API 端点。
"""
import asyncio
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .models import StreamingRequest, SegmentResult, StreamingResponseModel
from .dependencies import get_tts_model, get_translator
from ..services.s2st_service import S2STService
from voxcpm.utils import get_test_logger

# 导入配置模块
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.settings import get_settings

_logger = get_test_logger(__file__)

# 加载配置
settings = get_settings()

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

# 挂载静态文件目录（用于提供音频文件 URL 访问）
static_dir = settings.output_dir
static_dir.mkdir(parents=True, exist_ok=True)
app.mount(settings.static_url_prefix, StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "VoxCPM S2ST API",
        "version": "0.1.0",
        "description": "语音识别、翻译与语音合成流式 API",
        "endpoints": {
            "/s2st": "POST - 端到端语音翻译（兼容前端）",
            "/streaming": "POST - 流式处理音频文件（Server-Sent Events，含 TTS）",
            "/asr-transcribe": "POST - 流式识别+翻译（Server-Sent Events，不含 TTS，转发 ASR API 响应）",
            "/health": "GET - 健康检查",
            "/config": "GET - 获取配置",
        },
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "config_loaded": True, "output_dir": str(settings.output_dir)}


@app.get("/config")
async def get_config():
    """
    获取前端默认配置
    从 .env 文件读取配置参数，返回给前端使用
    """
    return {
        "ws_url": settings.ws_url,
        "user_id": settings.user_id,
        "token": settings.token,
        "from_lang": settings.from_lang,
        "to_lang": settings.to_lang,
        "role": "0",
        "lan_id": "0",
        "tts_text_source": settings.tts_text_source,
        "cfg_value": settings.cfg_value,
        "inference_timesteps": settings.inference_timesteps,
    }


@app.post("/asr-transcribe")
async def streaming_asr_transcribe(
    audio_file: UploadFile = File(..., description="音频文件（建议 16kHz 单声道 WAV）"),
    ws_url: str = Form(default=None, description="ASR WebSocket 服务地址（可通过配置文件设置）"),
    user_id: str = Form(default=None, description="用户 ID（可通过配置文件设置）"),
    token: str = Form(default=None, description="认证 Token（可通过配置文件设置）"),
    from_lang: str = Form(default=None),
    to_lang: str = Form(default=None),
    role: str = Form(default="0"),
    lan_id: str = Form(default="0"),
    sample_rate: int = Form(default=16000),
    bit_rate: int = Form(default=16),
    interval: float = Form(default=0.1),
):
    """
    流式 ASR 识别+翻译：上传音频文件，实时返回识别和翻译结果（不含 TTS）。

    使用 Server-Sent Events (SSE) 流式返回 ASR WebSocket API 的原始响应。
    事件类型：
    - intermediate: 中间识别结果
    - final: 单句完成（包含识别文本、翻译文本、时间戳等）
    - complete: 流结束
    - error: 错误信息
    """
    request_id = str(uuid.uuid4())[:8]
    _logger.info("[%s] 收到流式 ASR 识别请求，文件名: %s", request_id, audio_file.filename)

    # 从配置文件读取默认值（如果请求中未提供）
    ws_url = ws_url or settings.ws_url
    user_id = user_id or settings.user_id
    token = token or settings.token
    from_lang = from_lang or settings.from_lang
    to_lang = to_lang or settings.to_lang

    # 验证必填参数
    if not ws_url:
        raise HTTPException(status_code=400, detail="ws_url 参数或配置文件 WS_URL 必须提供")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 参数或配置文件 USER_ID 必须提供")
    if not token:
        raise HTTPException(status_code=400, detail="token 参数或配置文件 TOKEN 必须提供")

    # 创建请求对象（仅用于 ASR 参数）
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
    )

    # 保存上传的音频文件
    temp_audio_path = settings.temp_dir / f"{request_id}_{audio_file.filename}"

    try:
        with open(temp_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        _logger.info("[%s] 音频文件已保存: %s", request_id, temp_audio_path)
    except Exception as e:
        _logger.error("[%s] 保存音频文件失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"保存音频文件失败: {e}")

    # 初始化 ASR 翻译器
    try:
        translator = get_translator(request)
    except Exception as e:
        _logger.error("[%s] ASR 翻译器初始化失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"ASR 翻译器初始化失败: {e}")

    # 创建事件队列
    event_queue: asyncio.Queue = asyncio.Queue()

    # 定义 SSE 事件生成器
    async def generate_events():
        """生成 SSE 事件流"""
        processing_complete = False
        while not processing_complete or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                if event.get("event_type") == "complete":
                    processing_complete = True
                # 直接转发 ASR API 的原始响应
                import json
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                continue

    # 定义回调函数
    async def on_intermediate(parsed):
        """中间结果回调"""
        await event_queue.put({
            "event_type": "intermediate",
            "err_code": parsed.get("err_code"),
            "result_text": parsed.get("result_text", ""),
            "trans_text": parsed.get("trans_text", ""),
            "raw": parsed.get("raw", {}),
        })

    async def on_final(parsed):
        """最终结果回调（单句完成或流结束）"""
        await event_queue.put({
            "event_type": "final",
            "err_code": parsed.get("err_code"),
            "result_text": parsed.get("result_text", ""),
            "trans_text": parsed.get("trans_text", ""),
            "is_stream_finished": parsed.get("is_stream_finished", False),
            "raw": parsed.get("raw", {}),
        })

    # 启动后台处理任务
    async def process_asr():
        """在后台执行 ASR 识别翻译"""
        try:
            await translator.transcribe_streaming(
                audio_path=str(temp_audio_path),
                on_intermediate=on_intermediate,
                on_final=on_final,
            )
            # 发送完成事件
            await event_queue.put({"event_type": "complete"})
        except Exception as e:
            _logger.error("[%s] ASR 处理失败: %s", request_id, e, exc_info=True)
            await event_queue.put({
                "event_type": "error",
                "message": f"ASR 处理失败: {e}"
            })
        finally:
            # 清理临时文件
            try:
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
            except Exception as e:
                _logger.warning("[%s] 清理临时文件失败: %s", request_id, e)

    asyncio.create_task(process_asr())

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/s2st")
async def simple_s2st(
    audio: UploadFile = File(..., description="音频文件"),
    from_lang: str = Form(default=None, description="源语言"),
    to_lang: str = Form(default=None, description="目标语言"),
    tts_text_source: str = Form(default=None, description="TTS文本来源"),
    cfg_value: float = Form(default=None, description="CFG值"),
    inference_timesteps: int = Form(default=None, description="推理步数"),
):
    """
    端到端语音翻译接口（兼容 s2st_demo 前端）

    请求格式：multipart/form-data
    响应格式：JSON

    返回字段：
    - audio: 翻译后的音频URL
    - src_text: 源语言识别文本（可选）
    - tgt_text: 目标语言翻译文本（可选）
    """
    request_id = str(uuid.uuid4())[:8]
    _logger.info("[%s] 收到 /s2st 请求", request_id)

    # 验证 ASR 配置
    is_valid, error_msg = settings.validate_asr_config()
    if not is_valid:
        _logger.error("[%s] ASR 配置错误: %s", request_id, error_msg)
        raise HTTPException(status_code=500, detail=f"服务器配置错误：{error_msg}")

    # 使用请求参数或配置文件默认值
    from_lang = from_lang or settings.from_lang
    to_lang = to_lang or settings.to_lang
    tts_text_source = tts_text_source or settings.tts_text_source
    cfg_value = cfg_value if cfg_value is not None else settings.cfg_value
    inference_timesteps = inference_timesteps if inference_timesteps is not None else settings.inference_timesteps

    # 保存上传的音频文件
    temp_audio_path = settings.temp_dir / f"{request_id}_{audio.filename}"

    try:
        with open(temp_audio_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        _logger.info("[%s] 音频文件已保存: %s", request_id, temp_audio_path)
    except Exception as e:
        _logger.error("[%s] 保存音频文件失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"保存音频文件失败: {e}")

    # 创建请求对象
    request = StreamingRequest(
        ws_url=settings.ws_url,
        user_id=settings.user_id,
        token=settings.token,
        from_lang=from_lang,
        to_lang=to_lang,
        tts_text_source=tts_text_source,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
    )

    # 初始化模型和服务
    try:
        tts = get_tts_model(hf_model_id=settings.voxcpm_hf_id)
        translator = get_translator(request)
        service = S2STService(tts)
    except Exception as e:
        _logger.error("[%s] 模型初始化失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {e}")

    # 输出路径
    output_path = settings.output_dir / f"{request_id}_output.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # 使用非流式处理
        result_path = asyncio.run(service.process_non_streaming(
            audio_path=temp_audio_path,
            translator=translator,
            tts_text_source=request.tts_text_source,
            output_path=output_path,
            cfg_value=request.cfg_value,
            inference_timesteps=request.inference_timesteps,
            normalize=True,
            denoise=False,
        ))

        if not result_path or not result_path.exists():
            raise HTTPException(status_code=500, detail="语音合成失败")

        # 获取 ASR 结果用于返回
        asr_result = asyncio.run(translator.run(str(temp_audio_path), streaming=False))
        src_text = asr_result.get("result_text", "") if asr_result else ""
        tgt_text = asr_result.get("trans_text", "") if asr_result else ""

        # 构建音频 URL（使用相对路径，通过 /static 访问）
        audio_url = f"{settings.static_url_prefix}/{request_id}_output.wav"

        _logger.info("[%s] 处理完成，输出: %s", request_id, result_path)

        return {
            "audio": audio_url,
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    except HTTPException:
        raise
    except Exception as e:
        _logger.error("[%s] 处理失败: %s", request_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理失败: {e}")
    finally:
        # 清理临时文件
        try:
            if temp_audio_path.exists():
                temp_audio_path.unlink()
        except Exception as e:
            _logger.warning("[%s] 清理临时文件失败: %s", request_id, e)


@app.post("/streaming")
async def streaming_s2st(
    audio_file: UploadFile = File(..., description="音频文件（建议 16kHz 单声道 WAV）"),
    ws_url: str = Form(default=None, description="ASR WebSocket 服务地址（可通过配置文件设置）"),
    user_id: str = Form(default=None, description="用户 ID（可通过配置文件设置）"),
    token: str = Form(default=None, description="认证 Token（可通过配置文件设置）"),
    from_lang: str = Form(default=None),
    to_lang: str = Form(default=None),
    role: str = Form(default="0"),
    lan_id: str = Form(default="0"),
    sample_rate: int = Form(default=16000),
    bit_rate: int = Form(default=16),
    interval: float = Form(default=0.1),
    tts_text_source: str = Form(default=None),
    cfg_value: float = Form(default=None),
    inference_timesteps: int = Form(default=None),
    normalize: bool = Form(default=True),
    denoise: bool = Form(default=False),
    prompt_wav_path: str = Form(default=None),
    prompt_text: str = Form(default=None),
):
    """
    流式 S2ST 处理：上传音频文件，实时返回识别、翻译和合成结果。

    使用 Server-Sent Events (SSE) 流式返回结果。
    每完成一句识别+翻译+TTS，立即发送事件；全部完成后发送最终拼接的音频文件路径。
    """
    request_id = str(uuid.uuid4())[:8]
    _logger.info("[%s] 收到流式处理请求，文件名: %s", request_id, audio_file.filename)

    # 从配置文件读取默认值（如果请求中未提供）
    ws_url = ws_url or settings.ws_url
    user_id = user_id or settings.user_id
    token = token or settings.token
    from_lang = from_lang or settings.from_lang
    to_lang = to_lang or settings.to_lang
    tts_text_source = tts_text_source or settings.tts_text_source
    cfg_value = cfg_value if cfg_value is not None else settings.cfg_value
    inference_timesteps = inference_timesteps if inference_timesteps is not None else settings.inference_timesteps

    # 验证必填参数
    if not ws_url:
        raise HTTPException(status_code=400, detail="ws_url 参数或配置文件 WS_URL 必须提供")
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 参数或配置文件 USER_ID 必须提供")
    if not token:
        raise HTTPException(status_code=400, detail="token 参数或配置文件 TOKEN 必须提供")

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
    temp_audio_path = settings.temp_dir / f"{request_id}_{audio_file.filename}"

    try:
        with open(temp_audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        _logger.info("[%s] 音频文件已保存: %s", request_id, temp_audio_path)
    except Exception as e:
        _logger.error("[%s] 保存音频文件失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"保存音频文件失败: {e}")

    # 初始化模型和服务
    try:
        tts = get_tts_model(hf_model_id=settings.voxcpm_hf_id)
        translator = get_translator(request)
        service = S2STService(tts)
    except Exception as e:
        _logger.error("[%s] 模型初始化失败: %s", request_id, e)
        raise HTTPException(status_code=500, detail=f"模型初始化失败: {e}")

    # 创建事件队列
    event_queue: asyncio.Queue = asyncio.Queue()

    # 启动后台处理任务
    asyncio.create_task(service.process_streaming(temp_audio_path, translator, request, event_queue))

    # 定义 SSE 事件生成器
    async def generate_events():
        processing_complete = False
        while not processing_complete or not event_queue.empty():
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                if event.event_type == "complete":
                    processing_complete = True
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
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        log_level=settings.log_level.lower()
    )

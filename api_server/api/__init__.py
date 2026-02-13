"""
S2ST API 模块。

包含 FastAPI 路由、数据模型和依赖项。
"""
from .routes import app
from .models import (
    StreamingRequest,
    SegmentResult,
    StreamingResponseModel,
)
from .dependencies import get_tts_model, get_translator, select_tts_text

__all__ = [
    "app",
    "StreamingRequest",
    "SegmentResult",
    "StreamingResponseModel",
    "get_tts_model",
    "get_translator",
    "select_tts_text",
]

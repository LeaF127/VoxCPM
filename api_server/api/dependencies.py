"""
API 依赖项。

提供模型初始化和依赖注入功能。
"""
import os
from typing import Optional, Any

from .models import StreamingRequest
from ..clients.asr_websocket import ASRWebSocketClient
from ..models.tts_wrapper import TTSModelWrapper
from voxcpm.utils import get_test_logger

# 全局模型实例（懒加载）
_tts_model: Optional[TTSModelWrapper] = None
_logger = get_test_logger(__file__)


def get_tts_model(hf_model_id: Optional[str] = None) -> Any:
    """获取或初始化 TTS 模型（单例）"""
    global _tts_model
    if _tts_model is None:
        _logger.info("初始化 VoxCPM 模型...")
        _tts_model = TTSModelWrapper.load(
            hf_model_id=hf_model_id or os.getenv("VOXCPM_HF_ID", "openbmb/VoxCPM-0.5B"),
            cache_dir=os.getenv("VOXCPM_CACHE_DIR"),
            local_files_only=os.getenv("VOXCPM_LOCAL_FILES_ONLY", "false").lower() == "true",
            no_denoiser=os.getenv("VOXCPM_NO_DENOISER", "false").lower() == "true",
            no_optimize=os.getenv("VOXCPM_NO_OPTIMIZE", "false").lower() == "true",
        )
        _logger.info("VoxCPM 模型初始化完成")
    return _tts_model.tts_model


def get_translator(request: StreamingRequest) -> ASRWebSocketClient:
    """创建 ASR 翻译器实例"""
    return ASRWebSocketClient(
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


def select_tts_text(result_text: str, trans_text: str, source: str) -> Optional[str]:
    """根据配置选择用于 TTS 的文本"""
    asr_text = (result_text or "").strip()
    mt_text = (trans_text or "").strip()

    if source == "trans":
        return mt_text if mt_text else asr_text
    else:
        return asr_text if asr_text else mt_text

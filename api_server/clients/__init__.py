"""
ASR 客户端模块。

包含 WebSocket ASR 客户端和本地 ASR 识别器。
"""
from .asr_websocket import ASRWebSocketClient
from .asr_local import ASRLocalRecognizer

__all__ = [
    "ASRWebSocketClient",
    "ASRLocalRecognizer",
]

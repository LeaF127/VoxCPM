"""
ASR 翻译模块（兼容层）。

此文件保留用于向后兼容。新代码应使用：
- from s2st_demo.clients.asr_websocket import ASRWebSocketClient
- from s2st_demo.clients.asr_local import ASRLocalRecognizer
"""
import warnings

# 导入新的类并创建别名
from s2st_demo.clients.asr_websocket import ASRWebSocketClient as ASRTranslator
from s2st_demo.clients.asr_local import ASRLocalRecognizer as ASRRecognizer

# 发出迁移警告
warnings.warn(
    "直接从 asr_translate 导入已过时。请使用: "
    "from s2st_demo.clients.asr_websocket import ASRWebSocketClient "
    "或 from s2st_demo.clients.asr_local import ASRLocalRecognizer",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["ASRTranslator", "ASRRecognizer"]

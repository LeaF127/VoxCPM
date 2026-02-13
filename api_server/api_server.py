"""
FastAPI 服务器（兼容层）。

此文件保留用于向后兼容。新代码应使用：
- from s2st_demo.api.routes import app
- from s2st_demo.api.dependencies import get_tts_model, get_translator
"""
import warnings

# 导入新的 API
from s2st_demo.api.routes import app
from s2st_demo.api.dependencies import get_tts_model, get_translator

# 发出迁移警告
warnings.warn(
    "直接从 api_server 导入已过时。请使用: "
    "from s2st_demo.api.routes import app",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["app", "get_tts_model", "get_translator"]

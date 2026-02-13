"""
TTS 模块（兼容层）。

此文件保留用于向后兼容。新代码应使用：
- from s2st_demo.models.tts_wrapper import TTSModelWrapper
"""
import warnings

# 导入新的类和函数
from s2st_demo.models.tts_wrapper import (
    TTSModelWrapper,
    load_voxcpm,
    synthesize,
)

# 发出迁移警告
warnings.warn(
    "直接从 tts 导入已过时。请使用: "
    "from s2st_demo.models.tts_wrapper import TTSModelWrapper, load_voxcpm, synthesize",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["TTSModelWrapper", "load_voxcpm", "synthesize"]

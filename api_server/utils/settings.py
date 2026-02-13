"""
配置加载模块

从 .env 文件加载配置，支持跨平台使用（Windows/Linux）
"""
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from voxcpm.utils import get_test_logger

_logger = get_test_logger(__file__)


class Settings:
    """应用配置类"""

    def __init__(self, env_file: Optional[Path] = None):
        """
        初始化配置

        Args:
            env_file: .env 文件路径，默认为项目根目录的 .env
        """
        # 确定项目根目录
        self.project_root = Path(__file__).parent.parent.parent

        # 默认 .env 文件路径
        if env_file is None:
            env_file = self.project_root / ".env"

        # 加载 .env 文件
        if env_file.exists():
            load_dotenv(env_file, override=True)
            _logger.info(f"已加载配置文件: {env_file}")
        else:
            _logger.warning(f"配置文件不存在: {env_file}，将使用环境变量或默认值")

        # 加载配置
        self._load_config()

    def _load_config(self):
        """加载所有配置项"""
        # ============================================
        # ASR WebSocket 服务配置
        # ============================================
        self.ws_url: str = os.getenv("WS_URL", "")
        self.user_id: str = os.getenv("USER_ID", "")
        self.token: str = os.getenv("TOKEN", "")

        # ============================================
        # 语言配置
        # ============================================
        self.from_lang: str = os.getenv("FROM_LANG", "zh")
        self.to_lang: str = os.getenv("TO_LANG", "en")

        # ============================================
        # TTS 配置
        # ============================================
        self.tts_text_source: str = os.getenv("TTS_TEXT_SOURCE", "trans")
        self.cfg_value: float = float(os.getenv("CFG_VALUE", "2.0"))
        self.inference_timesteps: int = int(os.getenv("INFERENCE_TIMESTEPS", "10"))

        # ============================================
        # 模型配置
        # ============================================
        self.voxcpm_model_path: Optional[str] = os.getenv("VOXCPM_MODEL_PATH") or None
        self.voxcpm_hf_id: str = os.getenv("VOXCPM_HF_ID", "openbmb/VoxCPM-0.5B")
        self.voxcpm_cache_dir: Optional[str] = os.getenv("VOXCPM_CACHE_DIR") or None
        self.voxcpm_local_files_only: bool = os.getenv("VOXCPM_LOCAL_FILES_ONLY", "false").lower() == "true"
        self.voxcpm_no_denoiser: bool = os.getenv("VOXCPM_NO_DENOISER", "false").lower() == "true"
        self.voxcpm_no_optimize: bool = os.getenv("VOXCPM_NO_OPTIMIZE", "false").lower() == "true"

        # ============================================
        # API 服务配置
        # ============================================
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "19366"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

        # ============================================
        # 音频处理配置
        # ============================================
        self.audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
        self.audio_bit_rate: int = int(os.getenv("AUDIO_BIT_RATE", "16"))

        # ============================================
        # 路径配置
        # ============================================
        self.output_dir: Path = self.project_root / os.getenv("OUTPUT_DIR", "s2st_demo/output")
        self.temp_dir: Path = self.output_dir / "temp"
        self.static_url_prefix: str = os.getenv("STATIC_URL_PREFIX", "/static")

        # ============================================
        # 前端配置
        # ============================================
        self.frontend_port: int = int(os.getenv("FRONTEND_PORT", "8080"))

        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_asr_config(self) -> tuple[bool, str]:
        """
        验证 ASR 配置是否完整

        Returns:
            (is_valid, error_message)
        """
        if not self.ws_url:
            return False, "WS_URL 未配置，请在 .env 文件中设置 WS_URL"
        if not self.user_id:
            return False, "USER_ID 未配置，请在 .env 文件中设置 USER_ID"
        if not self.token:
            return False, "TOKEN 未配置，请在 .env 文件中设置 TOKEN"
        return True, ""

    def get_model_path(self) -> str:
        """获取模型路径（优先使用本地路径）"""
        if self.voxcpm_model_path:
            # 转换为绝对路径
            model_path = Path(self.voxcpm_model_path)
            if not model_path.is_absolute():
                model_path = self.project_root / model_path
            return str(model_path)
        return self.voxcpm_model_path or ""

    def __repr__(self) -> str:
        """打印配置信息（隐藏敏感信息）"""
        safe_token = self.token[:4] + "****" if self.token else ""
        return (
            f"Settings(\n"
            f"  ws_url={self.ws_url},\n"
            f"  user_id={self.user_id},\n"
            f"  token={safe_token},\n"
            f"  from_lang={self.from_lang},\n"
            f"  to_lang={self.to_lang},\n"
            f"  tts_text_source={self.tts_text_source},\n"
            f"  cfg_value={self.cfg_value},\n"
            f"  inference_timesteps={self.inference_timesteps},\n"
            f"  model_path={self.voxcpm_model_path},\n"
            f"  api_host={self.api_host},\n"
            f"  api_port={self.api_port},\n"
            f"  output_dir={self.output_dir},\n"
            f")"
        )


# 全局配置实例（单例）
_settings: Optional[Settings] = None


def get_settings(env_file: Optional[Path] = None) -> Settings:
    """
    获取配置实例（单例模式）

    Args:
        env_file: .env 文件路径（仅在首次调用时生效）

    Returns:
        Settings 实例
    """
    global _settings
    if _settings is None:
        _settings = Settings(env_file)
        _logger.info(f"配置已加载: {_settings}")
    return _settings


def reload_settings(env_file: Optional[Path] = None) -> Settings:
    """
    重新加载配置

    Args:
        env_file: .env 文件路径

    Returns:
        新的 Settings 实例
    """
    global _settings
    _settings = Settings(env_file)
    _logger.info(f"配置已重新加载: {_settings}")
    return _settings

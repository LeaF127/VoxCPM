"""
统一配置管理模块。

集中管理所有环境变量配置，提供类型安全的配置类。
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ASRConfig:
    """ASR/翻译服务配置"""
    ws_url: str
    user_id: str
    token: str
    from_lang: str = "zh"
    to_lang: str = "en"
    role: str = "0"
    lan_id: str = "0"
    sample_rate: int = 16000
    bit_rate: int = 16
    interval: float = 0.1

    @classmethod
    def from_env(cls) -> "ASRConfig":
        """从环境变量加载 ASR 配置"""
        return cls(
            ws_url=os.getenv("WS_URL", ""),
            user_id=os.getenv("USER_ID", ""),
            token=os.getenv("TOKEN", ""),
            from_lang=os.getenv("FROM_LAN", os.getenv("FROM_LANG", "zh")),
            to_lang=os.getenv("TO_LAN", os.getenv("TO_LANG", "en")),
            role=os.getenv("ROLE", "0"),
            lan_id=os.getenv("LAN_ID", "0"),
            sample_rate=int(os.getenv("SAMPLE_RATE", "16000")),
            bit_rate=int(os.getenv("BIT_RATE", "16")),
            interval=float(os.getenv("INTERVAL", "0.1")),
        )

    def validate(self) -> None:
        """验证必填配置"""
        if not self.ws_url:
            raise ValueError("ws_url 必须通过环境变量 WS_URL 设置")
        if not self.user_id:
            raise ValueError("user_id 必须通过环境变量 USER_ID 设置")
        if not self.token:
            raise ValueError("token 必须通过环境变量 TOKEN 设置")


@dataclass
class TTSConfig:
    """TTS 模型配置"""
    model_path: Optional[str] = None
    hf_model_id: str = "openbmb/VoxCPM-0.5B"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    no_denoiser: bool = False
    no_optimize: bool = False
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize: bool = False
    denoise: bool = False

    @classmethod
    def from_env(cls) -> "TTSConfig":
        """从环境变量加载 TTS 配置"""
        return cls(
            model_path=os.getenv("VOXCPM_MODEL_PATH"),
            hf_model_id=os.getenv("VOXCPM_HF_ID", "openbmb/VoxCPM-0.5B"),
            cache_dir=os.getenv("VOXCPM_CACHE_DIR"),
            local_files_only=os.getenv("VOXCPM_LOCAL_FILES_ONLY", "false").lower() == "true",
            no_denoiser=os.getenv("VOXCPM_NO_DENOISER", "false").lower() == "true",
            no_optimize=os.getenv("VOXCPM_NO_OPTIMIZE", "false").lower() == "true",
            cfg_value=float(os.getenv("CFG_VALUE", "2.0")),
            inference_timesteps=int(os.getenv("INFERENCE_TIMESTEPS", "10")),
            normalize=os.getenv("NORMALIZE", "false").lower() == "true",
            denoise=os.getenv("DENOISE", "false").lower() == "true",
        )


@dataclass
class APIConfig:
    """API 服务配置"""
    host: str = "0.0.0.0"
    port: int = 19366
    cors_origins: list = field(default_factory=lambda: ["*"])
    tts_text_source: str = "trans"  # "trans" 或 "asr"

    @classmethod
    def from_env(cls) -> "APIConfig":
        """从环境变量加载 API 配置"""
        return cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "19366")),
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            tts_text_source=os.getenv("TTS_TEXT_SOURCE", "trans"),
        )


@dataclass
class S2STConfig:
    """端到端 S2ST 配置"""
    asr: ASRConfig = field(default_factory=ASRConfig.from_env)
    tts: TTSConfig = field(default_factory=TTSConfig.from_env)
    api: APIConfig = field(default_factory=APIConfig.from_env)
    tts_text_source: str = "trans"  # "trans" 或 "asr"
    normalize: bool = False
    denoise: bool = False

    @classmethod
    def from_env(cls) -> "S2STConfig":
        """从环境变量加载完整配置"""
        return cls(
            asr=ASRConfig.from_env(),
            tts=TTSConfig.from_env(),
            api=APIConfig.from_env(),
            tts_text_source=os.getenv("TTS_TEXT_SOURCE", "trans"),
            normalize=os.getenv("NORMALIZE", "false").lower() == "true",
            denoise=os.getenv("DENOISE", "false").lower() == "true",
        )

    def validate(self) -> None:
        """验证配置"""
        self.asr.validate()

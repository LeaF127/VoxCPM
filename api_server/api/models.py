"""
API 请求/响应模型。

定义 FastAPI 的请求和响应数据结构。
"""
from typing import Dict, Any, Optional

from pydantic import BaseModel, Field


class StreamingRequest(BaseModel):
    """流式处理请求参数（JSON body）"""

    ws_url: str = Field(..., description="ASR WebSocket 服务地址")
    user_id: str = Field(..., description="用户 ID")
    token: str = Field(..., description="认证 Token")
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

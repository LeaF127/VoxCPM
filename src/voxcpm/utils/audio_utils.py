"""
音频处理工具模块，提供音频加载、重采样、切分、拼接等功能。
"""

from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_audio(
    audio_path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    加载音频文件，支持自动重采样和单声道转换。

    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率，如果为 None 则保持原始采样率
        mono: 是否转换为单声道（默认 True）

    Returns:
        (audio_array, sample_rate): 音频数组和采样率
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    audio, sr = sf.read(str(audio_path))

    # 转换为单声道
    if mono and len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # 重采样
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return audio, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    对音频进行重采样。

    Args:
        audio: 音频数组（1D 或 2D）
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数组
    """
    if orig_sr == target_sr:
        return audio

    # 确保是 numpy 数组
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    # 转换为 torch tensor（需要添加 batch 和 channel 维度）
    is_1d = len(audio.shape) == 1
    if is_1d:
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    else:
        audio_tensor = torch.from_numpy(audio)

    # 重采样
    resampled_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)

    # 转换回 numpy
    if is_1d:
        resampled_audio = resampled_tensor.squeeze(0).numpy()
    else:
        resampled_audio = resampled_tensor.numpy()

    return resampled_audio


def segment_audio_by_timestamp(
    audio: np.ndarray,
    sample_rate: int,
    begin_ms: int,
    end_ms: int,
    target_sr: Optional[int] = None,
) -> np.ndarray:
    """
    根据时间戳切分音频片段。

    Args:
        audio: 音频数组
        sample_rate: 音频采样率
        begin_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒）
        target_sr: 目标采样率（可选，如果提供则对切分后的音频进行重采样）

    Returns:
        切分后的音频片段
    """
    # 将毫秒转换为样本索引
    begin_sample = int(begin_ms * sample_rate / 1000.0)
    end_sample = int(end_ms * sample_rate / 1000.0)

    # 边界检查
    begin_sample = max(0, min(begin_sample, len(audio)))
    end_sample = max(begin_sample + 1, min(end_sample, len(audio)))

    # 切分音频
    segment = audio[begin_sample:end_sample]

    # 重采样（如果需要）
    if target_sr is not None and sample_rate != target_sr:
        segment = resample_audio(segment, sample_rate, target_sr)

    return segment


def concatenate_audio(
    audio_segments: List[np.ndarray],
    sample_rates: Optional[List[int]] = None,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    拼接多个音频片段。

    Args:
        audio_segments: 音频片段列表
        sample_rates: 每个片段的采样率列表（可选）
        target_sr: 目标采样率（如果提供，所有片段将重采样到此采样率）

    Returns:
        (拼接后的音频, 采样率)
    """
    if not audio_segments:
        raise ValueError("音频片段列表不能为空")

    if sample_rates is None:
        # 假设所有片段采样率相同
        sample_rates = [None] * len(audio_segments)

    # 如果指定了目标采样率，对所有片段进行重采样
    if target_sr is not None:
        processed_segments = []
        for audio, sr in zip(audio_segments, sample_rates):
            if sr is not None and sr != target_sr:
                audio = resample_audio(audio, sr, target_sr)
            processed_segments.append(audio)
        final_sr = target_sr
    else:
        # 检查采样率是否一致
        unique_srs = set(sr for sr in sample_rates if sr is not None)
        if len(unique_srs) > 1:
            raise ValueError(f"音频片段采样率不一致: {unique_srs}，请指定 target_sr 进行统一重采样")
        final_sr = unique_srs.pop() if unique_srs else None
        processed_segments = audio_segments

    # 拼接
    concatenated = np.concatenate(processed_segments, axis=0)

    return concatenated, final_sr


def save_audio(
    audio: np.ndarray,
    output_path: Union[str, Path],
    sample_rate: int,
    **kwargs,
) -> Path:
    """
    保存音频文件。

    Args:
        audio: 音频数组
        output_path: 输出路径
        sample_rate: 采样率
        **kwargs: 传递给 soundfile.write 的其他参数

    Returns:
        输出路径（Path 对象）
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate, **kwargs)
    return output_path


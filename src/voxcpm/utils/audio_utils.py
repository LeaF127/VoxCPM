"""
音频处理工具模块，提供音频加载、重采样、切分、拼接等功能。
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional, List, Union

import numpy as np
import soundfile as sf

try:
    import torch
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


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
    对音频进行重采样。优先使用 ffmpeg，如果未安装则回退到 torchaudio。

    Args:
        audio: 音频数组（1D 或 2D）
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数组（float32，范围 [-1, 1]）

    Raises:
        RuntimeError: 如果 ffmpeg 和 torchaudio 都不可用
    """
    if orig_sr == target_sr:
        return audio

    # 确保是 numpy 数组
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    # 转换为单声道（如果多声道）
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # 确保数据类型为 float32（soundfile 标准格式）
    if audio.dtype != np.float32:
        # 如果是整数类型，先归一化到 [-1, 1]
        if audio.dtype in (np.int16, np.int32):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val
        else:
            audio = audio.astype(np.float32)

    # 优先尝试使用 ffmpeg
    try:
        return _resample_with_ffmpeg(audio, orig_sr, target_sr)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg 不可用，回退到 torchaudio
        if TORCHAUDIO_AVAILABLE:
            return _resample_with_torchaudio(audio, orig_sr, target_sr)
        else:
            raise RuntimeError(
                "无法进行音频重采样：ffmpeg 和 torchaudio 都不可用。\n"
                "请安装其中之一：\n"
                "  - ffmpeg: Windows/Linux/macOS 系统级安装\n"
                "  - torchaudio: pip install torchaudio"
            )


def _resample_with_ffmpeg(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    使用 ffmpeg 对音频进行重采样。

    Args:
        audio: 音频数组（1D，float32）
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数组（float32，范围 [-1, 1]）

    Raises:
        FileNotFoundError: 如果 ffmpeg 未安装
        subprocess.CalledProcessError: 如果 ffmpeg 执行失败
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
        input_path = Path(input_file.name)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
        output_path = Path(output_file.name)

    try:
        # 保存输入音频
        sf.write(str(input_path), audio, orig_sr)

        # 使用 ffmpeg 重采样：-ac 1（单声道） -ar {target_sr}（采样率） -c:a pcm_s16le（16位 PCM）
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-c:a",
            "pcm_s16le",
            "-y",  # 覆盖输出文件
            str(output_path),
        ]

        # 执行 ffmpeg，隐藏输出
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        # 读取输出音频
        resampled_audio, _ = sf.read(str(output_path), dtype=np.int16)

        # 转换为 float32 并归一化到 [-1, 1]
        resampled_audio = resampled_audio.astype(np.float32) / 32768.0

        return resampled_audio

    except FileNotFoundError:
        raise FileNotFoundError(
            "ffmpeg 未找到。请安装 ffmpeg：\n"
            "  - Windows: 从 https://ffmpeg.org/download.html 下载并添加到 PATH\n"
            "  - Linux: sudo apt-get install ffmpeg (Debian/Ubuntu) 或 sudo yum install ffmpeg (RHEL/CentOS)\n"
            "  - macOS: brew install ffmpeg"
        )
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode("utf-8", errors="ignore") if e.stderr else str(e)
        raise RuntimeError(f"ffmpeg 执行失败: {error_msg}") from e
    finally:
        # 清理临时文件
        try:
            if input_path.exists():
                input_path.unlink()
            if output_path.exists():
                output_path.unlink()
        except Exception:
            pass  # 忽略清理错误


def _resample_with_torchaudio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    使用 torchaudio 对音频进行重采样。

    Args:
        audio: 音频数组（1D，float32）
        orig_sr: 原始采样率
        target_sr: 目标采样率

    Returns:
        重采样后的音频数组（float32，范围 [-1, 1]）
    """
    if not TORCHAUDIO_AVAILABLE:
        raise RuntimeError("torchaudio 未安装，无法进行重采样")

    # 转换为 torch tensor，添加 batch 和 channel 维度
    # audio shape: (T,) -> (1, 1, T)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)

    # 重采样
    resampled_tensor = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)

    # 转换回 numpy 数组，移除维度
    # (1, 1, T) -> (T,)
    resampled_audio = resampled_tensor.squeeze(0).squeeze(0).numpy()

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


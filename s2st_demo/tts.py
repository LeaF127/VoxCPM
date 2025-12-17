import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from voxcpm.core import VoxCPM
from voxcpm.utils import get_test_logger


def load_voxcpm(
    model_path: Optional[str] = None,
    hf_model_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    no_denoiser: bool = False,
    no_optimize: bool = False,
) -> VoxCPM:
    """
    加载 VoxCPM 模型，优先使用本地路径，其次从 Hugging Face Hub 下载。
    """
    logger = get_test_logger(__file__)

    if model_path:
        logger.info("使用本地 VoxCPM 模型目录: %s", model_path)
        return VoxCPM(
            voxcpm_model_path=model_path,
            enable_denoiser=not no_denoiser,
            optimize=not no_optimize,
        )

    hf_id = hf_model_id or "openbmb/VoxCPM-0.5B"
    logger.info("从 Hugging Face Hub 加载 VoxCPM 模型: %s", hf_id)
    return VoxCPM.from_pretrained(
        hf_model_id=hf_id,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        load_denoiser=not no_denoiser,
        optimize=not no_optimize,
    )


def synthesize(
    tts: VoxCPM,
    text: str,
    output_path: str,
    prompt_wav: Optional[str] = None,
    prompt_text: Optional[str] = None,
    cfg_value: float = 2.0,
    inference_timesteps: int = 10,
    normalize: bool = True,
    denoise: bool = True,
) -> Path:
    """
    使用已加载的 VoxCPM 模型进行一次性 TTS 合成，并保存为 WAV。
    """
    logger = get_test_logger(__file__)
    text = (text or "").strip()
    if not text:
        raise ValueError("合成文本不能为空")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("开始合成语音，文本长度=%d，输出路径=%s", len(text), out_path)

    audio: np.ndarray = tts.generate(
        text=text,
        prompt_wav_path=prompt_wav,
        prompt_text=prompt_text,
        cfg_value=cfg_value,
        inference_timesteps=inference_timesteps,
        normalize=normalize,
        denoise=denoise and prompt_wav is not None,
    )

    sample_rate = getattr(tts.tts_model, "sample_rate", 16000)
    sf.write(str(out_path), audio, sample_rate)

    duration = len(audio) / float(sample_rate)
    logger.info("合成完成，时长 %.2f 秒，采样率 %d Hz", duration, sample_rate)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 VoxCPM 的简单 TTS 脚本")
    parser.add_argument("--text", "-t", help="待合成文本", required=True)
    parser.add_argument("--output", "-o", help="输出 WAV 路径", default="s2st_demo/output/tts_demo.wav")

    # 可选：提示音频（语音克隆）
    parser.add_argument("--prompt-wav", help="参考音频路径", default=None)
    parser.add_argument("--prompt-text", help="参考音频对应文本", default=None)

    # VoxCPM 模型加载相关
    parser.add_argument("--model-path", help="本地 VoxCPM 模型目录", default=None)
    parser.add_argument("--hf-model-id", help="Hugging Face 模型 ID", default="openbmb/VoxCPM-0.5B")
    parser.add_argument("--cache-dir", help="模型缓存目录", default=None)
    parser.add_argument("--local-files-only", action="store_true", help="仅使用本地模型文件（不访问网络）")
    parser.add_argument("--no-denoiser", action="store_true", help="禁用降噪模型")
    parser.add_argument("--no-optimize", action="store_true", help="禁用 torch.compile 优化")

    # 生成参数
    parser.add_argument("--cfg-value", type=float, default=2.0, help="CFG guidance scale，默认 2.0")
    parser.add_argument("--inference-timesteps", type=int, default=10, help="扩散步数，默认 10")
    parser.add_argument("--normalize", action="store_true", help="启用文本正则化")
    parser.add_argument("--denoise", action="store_true", help="对参考音频进行降噪")

    return parser


def main() -> None:
    """
    命令行入口：单次文本转语音测试。
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    tts = load_voxcpm(
        model_path=args.model_path,
        hf_model_id=args.hf_model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        no_denoiser=args.no_denoiser,
        no_optimize=args.no_optimize,
    )

    synthesize(
        tts=tts,
        text=args.text,
        output_path=args.output,
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        cfg_value=args.cfg_value,
        inference_timesteps=args.inference_timesteps,
        normalize=args.normalize,
        denoise=args.denoise,
    )


if __name__ == "__main__":
    main()



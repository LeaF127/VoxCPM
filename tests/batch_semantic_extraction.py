"""
批量提取语义 Tokens 脚本

从一批音频中提取语义 tokens，并保存到文件。
支持多种输出格式和可视化分析。

用法：
    # 处理单个音频
    python batch_semantic_extraction.py --input audios/test.wav

    # 处理目录中的所有音频
    python batch_semantic_extraction.py --input_dir audios/ --output_dir ./semantic_tokens

    # 从文件列表读取
    python batch_semantic_extraction.py --file_list audio_list.txt
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from voxcpm.core import VoxCPM
from voxcpm.utils import get_test_logger
from voxcpm.model.utils import get_dtype


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量提取音频的语义 tokens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个音频
  python batch_semantic_extraction.py --input test.wav

  # 处理整个目录
  python batch_semantic_extraction.py --input_dir ./audios --output_dir ./tokens

  # 使用文件列表
  python batch_semantic_extraction.py --file_list audio_list.txt
        """
    )

    # 输入选项
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="单个音频文件路径")
    group.add_argument("--input_dir", type=str, help="音频文件目录")
    group.add_argument("--file_list", type=str, help="包含音频路径列表的文件")

    # 输出选项
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./semantic_tokens",
        help="输出目录（默认: ./semantic_tokens）",
    )
    parser.add_argument(
        "--output_format",
        choices=["pt", "npy", "both"],
        default="both",
        help="输出格式: pt(PyTorch), npy(NumPy), both(两种格式)（默认: both）",
    )

    # VoxCPM 模型配置
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="本地 VoxCPM 模型目录",
    )
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="openbmb/VoxCPM-0.5B",
        help="Hugging Face 模型 ID（默认: openbmb/VoxCPM-0.5B）",
    )
    parser.add_argument(
        "--no_denoiser",
        action="store_true",
        help="禁用降噪模型（加速加载）",
    )
    parser.add_argument(
        "--no_optimize",
        action="store_true",
        help="禁用 torch.compile 优化（用于调试）",
    )

    # 音频处理选项
    parser.add_argument(
        "--audio_extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg"],
        help="处理的音频扩展名（默认: .wav .mp3 .flac .ogg）",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=None,
        help="最大音频时长（秒），超过则跳过（默认: 无限制）",
    )

    # 分析选项
    parser.add_argument(
        "--save_stats",
        action="store_true",
        help="保存统计信息到 JSON 文件",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="生成语义 tokens 的可视化图",
    )

    return parser.parse_args()


class SemanticTokenExtractor:
    """语义 Token 提取器"""

    def __init__(self, voxcpm: VoxCPM, logger=None):
        """
        Args:
            voxcpm: VoxCPM 模型实例
            logger: 日志记录器
        """
        self.voxcpm = voxcpm
        self.model = voxcpm.tts_model
        self.logger = logger or get_test_logger(__name__)

        # 获取模型配置
        self.patch_size = self.model.patch_size
        self.chunk_size = self.model.chunk_size
        self.sample_rate = self.model.sample_rate
        self.latent_dim = self.model.audio_vae.latent_dim
        self.hidden_dim = self.model.config.lm_config.hidden_size

        self.logger.info(f"模型配置:")
        self.logger.info(f"  - Patch size: {self.patch_size}")
        self.logger.info(f"  - Chunk size: {self.chunk_size}")
        self.logger.info(f"  - Sample rate: {self.sample_rate}")
        self.logger.info(f"  - Latent dim: {self.latent_dim}")
        self.logger.info(f"  - Hidden dim: {self.hidden_dim}")

    def extract_semantic_tokens(
        self,
        audio_path: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        从音频中提取语义 tokens

        Args:
            audio_path: 音频文件路径

        Returns:
            semantic_tokens: [T, hidden_dim] 语义 token 序列
            metadata: 包含音频和处理信息的字典
        """
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        original_duration = audio.size(1) / sr

        # 重采样到模型采样率
        if sr != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

        audio = audio.to(self.model.device)

        # 使用 AudioVAE 编码为特征
        patch_len = self.patch_size * self.chunk_size

        # 填充音频以匹配 patch_size
        if audio.size(1) % patch_len != 0:
            pad_length = patch_len - (audio.size(1) % patch_len)
            audio = torch.nn.functional.pad(audio, (0, pad_length))

        # 编码为音频特征: [B, D, T]
        with torch.no_grad():
            audio_feat = self.model.audio_vae.encode(audio, self.sample_rate)

        # 重塑为 [B, T, P, D] 格式
        B, D, T_total = audio_feat.shape
        P = self.patch_size

        # Reshape: [B, D, T] -> [B, T//P, P, D]
        audio_feat = audio_feat.view(B, D, T_total // P, P).permute(0, 2, 3, 1)

        # 移除最后一个 padding token
        audio_feat = audio_feat[:, :-1, ...]

        # 转换为模型的数据类型 (bfloat16)
        model_dtype = get_dtype(self.model.config.dtype)
        audio_feat = audio_feat.to(model_dtype)

        # 使用语义 encoder 提取语义 tokens
        with torch.no_grad():
            # feat_encoder: [B, T, P, D] -> [B, T, hidden_dim]
            feat_embed = self.model.feat_encoder(audio_feat)

            # 投影到 LM 空间
            semantic_tokens = self.model.enc_to_lm_proj(feat_embed)

        # 元数据
        metadata = {
            "audio_path": str(audio_path),
            "original_sample_rate": sr,
            "original_duration": original_duration,
            "resampled_duration": audio.size(1) / self.sample_rate,
            "num_frames": audio_feat.shape[1],
            "token_length": semantic_tokens.shape[1],
            "feature_dim": semantic_tokens.shape[2],
        }

        # 返回 [T, hidden_dim]
        return semantic_tokens.squeeze(0).cpu(), metadata

    def save_tokens(
        self,
        semantic_tokens: torch.Tensor,
        metadata: Dict,
        output_path: Path,
        output_format: str = "both",
    ):
        """保存语义 tokens 和元数据到文件"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存 tokens
        if output_format in ["pt", "both"]:
            torch.save(semantic_tokens, str(output_path.with_suffix(".pt")))

        if output_format in ["npy", "both"]:
            np.save(str(output_path.with_suffix(".npy")), semantic_tokens.numpy())

        # 保存元数据
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata_path

    def visualize_tokens(
        self,
        semantic_tokens: torch.Tensor,
        output_path: Path,
    ):
        """生成语义 tokens 的可视化图"""
        try:
            import matplotlib.pyplot as plt

            # 转换为 numpy
            tokens_np = semantic_tokens.numpy()

            # 创建图表
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))

            # 热力图
            im = axes[0].imshow(
                tokens_np.T,
                aspect="auto",
                cmap="viridis",
                interpolation="nearest",
            )
            axes[0].set_title("Semantic Tokens Heatmap")
            axes[0].set_xlabel("Time Step")
            axes[0].set_ylabel("Feature Dimension")
            plt.colorbar(im, ax=axes[0])

            # 特征统计
            mean = tokens_np.mean(axis=0)
            std = tokens_np.std(axis=0)

            axes[1].plot(mean, label="Mean", alpha=0.7)
            axes[1].fill_between(
                range(len(mean)),
                mean - std,
                mean + std,
                alpha=0.3,
                label="±1 Std",
            )
            axes[1].set_title("Feature Statistics Over Time")
            axes[1].set_xlabel("Feature Dimension")
            axes[1].set_ylabel("Value")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig(str(output_path.with_suffix(".png")), dpi=150)
            plt.close()

            self.logger.info(f"可视化图已保存: {output_path.with_suffix('.png')}")

        except ImportError:
            self.logger.warning("matplotlib 未安装，跳过可视化")


def collect_audio_files(args) -> List[Tuple[str, Path]]:
    """收集要处理的音频文件"""
    audio_files = []

    if args.input:
        # 单个文件
        audio_files.append((args.input, Path(args.output_dir) / Path(args.input).stem))

    elif args.input_dir:
        # 整个目录
        input_dir = Path(args.input_dir)
        for ext in args.audio_extensions:
            for audio_path in input_dir(f"**/*{ext}", recursive=True):
                rel_path = audio_path.relative_to(input_dir)
                output_path = Path(args.output_dir) / rel_path.stem
                audio_files.append((str(audio_path), output_path))

    elif args.file_list:
        # 从文件列表读取
        with open(args.file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    audio_files.append(
                        (line, Path(args.output_dir) / Path(line).stem)
                    )

    return audio_files


def main():
    args = parse_args()
    logger = get_test_logger(__file__)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 VoxCPM 模型
    logger.info("=" * 80)
    logger.info("正在加载 VoxCPM 模型...")
    logger.info("=" * 80)

    voxcpm = VoxCPM.from_pretrained(
        hf_model_id=args.model_path,
        load_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
    )

    logger.info("=" * 80)
    logger.info("模型加载完成！")
    logger.info("=" * 80)

    # 初始化提取器
    extractor = SemanticTokenExtractor(voxcpm, logger=logger)

    # 收集音频文件
    audio_files = collect_audio_files(args)

    if not audio_files:
        logger.error("未找到任何音频文件")
        return

    logger.info(f"找到 {len(audio_files)} 个音频文件")

    # 处理每个音频文件
    results = []

    for audio_path, output_path in tqdm(audio_files, desc="提取语义 tokens"):
        try:
            # 检查音频时长
            if args.max_duration:
                info = torchaudio.info(audio_path)
                duration = info.num_frames / info.sample_rate
                if duration > args.max_duration:
                    logger.warning(
                        f"跳过 {audio_path}（时长 {duration:.2f}s 超过限制）"
                    )
                    continue

            # 提取语义 tokens
            semantic_tokens, metadata = extractor.extract_semantic_tokens(audio_path)

            logger.info(
                f"提取成功: {audio_path} -> tokens shape: {semantic_tokens.shape}"
            )

            # 保存 tokens
            metadata_path = extractor.save_tokens(
                semantic_tokens,
                metadata,
                output_path,
                args.output_format,
            )

            # 可视化
            if args.visualize:
                extractor.visualize_tokens(semantic_tokens, output_path)

            results.append(
                {
                    "audio_path": audio_path,
                    "output_path": str(output_path),
                    "metadata_path": str(metadata_path),
                    "success": True,
                    **metadata,
                }
            )

        except Exception as e:
            logger.error(f"处理失败 {audio_path}: {e}")
            results.append(
                {
                    "audio_path": audio_path,
                    "output_path": str(output_path),
                    "success": False,
                    "error": str(e),
                }
            )

    # 保存统计信息
    if args.save_stats:
        stats_path = output_dir / "extraction_stats.json"

        stats = {
            "total_files": len(audio_files),
            "successful": sum(1 for r in results if r.get("success", False)),
            "failed": sum(1 for r in results if not r.get("success", False)),
            "results": results,
            "model_config": {
                "patch_size": extractor.patch_size,
                "chunk_size": extractor.chunk_size,
                "sample_rate": extractor.sample_rate,
                "latent_dim": extractor.latent_dim,
                "hidden_dim": extractor.hidden_dim,
            },
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        logger.info(f"统计信息已保存到: {stats_path}")

    # 打印总结
    logger.info("=" * 80)
    logger.info("处理完成！")
    logger.info(f"成功: {sum(1 for r in results if r.get('success', False))} / {len(results)}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

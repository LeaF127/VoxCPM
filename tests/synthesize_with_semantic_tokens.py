"""
使用语义 Token 进行音频合成

这个脚本展示了如何使用之前提取的语义 tokens 来进行音频合成。
由于 VoxCPM 是基于文本的模型，我们展示几种实际的使用方式：

1. **使用语义 tokens 作为语音克隆的 prompt**:
   - 从源音频提取语义 tokens
   - 使用这些 tokens 来指导新文本的语音合成

2. **语义 tokens 分析与可视化**:
   - 分析不同音频的语义 tokens
   - 比较它们的相似度

用法：
    python synthesize_with_semantic_tokens.py --semantic_tokens semantic_tokens.npy --text "你好，世界"
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from voxcpm.core import VoxCPM
from voxcpm.utils import get_test_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用语义 tokens 进行音频合成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用语义 tokens 合成新文本
  python synthesize_with_semantic_tokens.py \\
      --semantic_tokens tokens.npy \\
      --text "你好，这是测试语音" \\
      --output synthesized.wav

  # 批量合成多个文本
  python synthesize_with_semantic_tokens.py \\
      --semantic_tokens tokens.npy \\
      --text_list texts.txt \\
      --output_dir ./outputs

  # 分析语义 tokens
  python synthesize_with_semantic_tokens.py \\
      --semantic_tokens tokens.npy \\
      --analyze_only
        """
    )

    # 输入
    parser.add_argument(
        "--semantic_tokens",
        type=str,
        required=True,
        help="语义 tokens 文件路径（.npy 或 .pt）",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="元数据文件路径（.json）",
    )
    parser.add_argument(
        "--original_audio",
        type=str,
        default=None,
        help="原始音频路径（可选，用于语音克隆）",
    )

    # 文本输入
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", type=str, help="要合成的文本")
    group.add_argument("--text_list", type=str, help="包含多行文本的文件")
    group.add_argument(
        "--analyze_only",
        action="store_true",
        help="仅分析语义 tokens，不进行合成",
    )

    # 输出
    parser.add_argument(
        "--output",
        type=str,
        default="synthesized_from_tokens.wav",
        help="输出音频路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./synthesized_audio",
        help="批量合成时的输出目录",
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
        help="Hugging Face 模型 ID",
    )
    parser.add_argument(
        "--no_denoiser",
        action="store_true",
        help="禁用降噪模型",
    )
    parser.add_argument(
        "--no_optimize",
        action="store_true",
        help="禁用 torch.compile 优化",
    )

    # 合成参数
    parser.add_argument(
        "--cfg_value",
        type=float,
        default=2.0,
        help="CFG guidance scale（默认: 2.0）",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="扩散步数（默认: 10）",
    )

    return parser.parse_args()


class SemanticTokenAnalyzer:
    """语义 Token 分析器"""

    def __init__(self, logger=None):
        self.logger = logger or get_test_logger(__name__)

    def load_tokens(
        self,
        tokens_path: str,
    ) -> torch.Tensor:
        """加载语义 tokens"""
        tokens_path = Path(tokens_path)

        if tokens_path.suffix == ".pt":
            tokens = torch.load(str(tokens_path))
        elif tokens_path.suffix == ".npy":
            tokens = torch.from_numpy(np.load(str(tokens_path)))
        else:
            raise ValueError(f"不支持的文件格式: {tokens_path.suffix}")

        self.logger.info(f"已加载语义 tokens: {tokens.shape}")
        return tokens

    def load_metadata(
        self,
        metadata_path: Optional[str] = None,
    ) -> dict:
        """加载元数据"""
        if metadata_path is None:
            return {}

        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            self.logger.warning(f"元数据文件不存在: {metadata_path}")
            return {}

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.logger.info(f"已加载元数据: {metadata}")
        return metadata

    def analyze_tokens(
        self,
        tokens: torch.Tensor,
    ) -> dict:
        """分析语义 tokens 的统计特性"""
        tokens_np = tokens.numpy()

        analysis = {
            "shape": list(tokens.shape),
            "dtype": str(tokens.dtype),
            "mean": float(tokens_np.mean()),
            "std": float(tokens_np.std()),
            "min": float(tokens_np.min()),
            "max": float(tokens_np.max()),
            "norm": float(np.linalg.norm(tokens_np)),
            "time_steps": tokens.shape[0],
            "feature_dim": tokens.shape[1] if len(tokens.shape) > 1 else 1,
        }

        # 时序统计
        if len(tokens.shape) > 1:
            temporal_mean = tokens_np.mean(axis=1)
            analysis["temporal_stats"] = {
                "mean_mean": float(temporal_mean.mean()),
                "mean_std": float(temporal_mean.std()),
            }

        self.logger.info("语义 Token 分析:")
        self.logger.info(f"  形状: {analysis['shape']}")
        self.logger.info(f"  均值: {analysis['mean']:.4f}")
        self.logger.info(f"  标准差: {analysis['std']:.4f}")
        self.logger.info(f"  范围: [{analysis['min']:.4f}, {analysis['max']:.4f}]")
        self.logger.info(f"  范数: {analysis['norm']:.4f}")

        return analysis

    def compare_tokens(
        self,
        tokens1: torch.Tensor,
        tokens2: torch.Tensor,
    ) -> dict:
        """比较两个语义 token 序列的相似度"""
        # 确保形状一致
        if tokens1.shape != tokens2.shape:
            # 对齐到较短的长度
            min_len = min(tokens1.shape[0], tokens2.shape[0])
            tokens1 = tokens1[:min_len]
            tokens2 = tokens2[:min_len]

        # 计算相似度
        cosine_sim = F.cosine_similarity(
            tokens1.flatten().unsqueeze(0),
            tokens2.flatten().unsqueeze(0),
        )

        euclidean_dist = torch.norm(tokens1 - tokens2).item()

        # 时序相似度（逐帧）
        frame_sim = F.cosine_similarity(tokens1, tokens2, dim=1)

        comparison = {
            "cosine_similarity": cosine_sim.item(),
            "euclidean_distance": euclidean_dist,
            "frame_mean_similarity": frame_sim.mean().item(),
            "frame_std_similarity": frame_sim.std().item(),
        }

        self.logger.info("Token 相似度比较:")
        self.logger.info(f"  余弦相似度: {comparison['cosine_similarity']:.4f}")
        self.logger.info(f"  欧氏距离: {comparison['euclidean_distance']:.4f}")
        self.logger.info(
            f"  帧平均相似度: {comparison['frame_mean_similarity']:.4f}"
        )

        return comparison


class SemanticTokenSynthesizer:
    """使用语义 Token 的音频合成器"""

    def __init__(self, voxcpm: VoxCPM, logger=None):
        """
        Args:
            voxcpm: VoxCPM 模型实例
            logger: 日志记录器
        """
        self.voxcpm = voxcpm
        self.model = voxcpm.tts_model
        self.logger = logger or get_test_logger(__name__)

    def synthesize_with_prompt(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> tuple:
        """
        使用 prompt 音频合成文本（标准的 VoxCPM 用法）

        Args:
            text: 要合成的文本
            prompt_wav_path: 提示音频路径
            prompt_text: 提示音频对应的文本
            cfg_value: CFG guidance scale
            inference_timesteps: 扩散步数

        Returns:
            (audio_array, sample_rate): 合成的音频和采样率
        """
        self.logger.info(f"正在合成文本: {text}")

        audio = self.voxcpm.generate(
            text=text,
            prompt_wav_path=prompt_wav_path,
            prompt_text=prompt_text,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=False,
            denoise=prompt_wav_path is not None,
        )

        sample_rate = self.model.sample_rate
        self.logger.info(f"合成完成，音频长度: {len(audio) / sample_rate:.2f} 秒")

        return audio, sample_rate

    def batch_synthesize(
        self,
        texts: List[str],
        output_dir: Path,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
    ) -> List[Path]:
        """批量合成多个文本"""
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = []

        for i, text in enumerate(tqdm(texts, desc="合成音频")):
            try:
                audio, sample_rate = self.synthesize_with_prompt(
                    text=text,
                    prompt_wav_path=prompt_wav_path,
                    prompt_text=prompt_text,
                    cfg_value=cfg_value,
                    inference_timesteps=inference_timesteps,
                )

                # 保存音频
                output_path = output_dir / f"output_{i:03d}.wav"
                sf.write(str(output_path), audio, sample_rate)

                self.logger.info(f"已保存: {output_path}")
                output_files.append(output_path)

            except Exception as e:
                self.logger.error(f"合成失败: {text} - {e}")

        return output_files


def main():
    args = parse_args()
    logger = get_test_logger(__file__)

    # 加载语义 tokens
    logger.info("=" * 80)
    logger.info("正在加载语义 tokens...")
    logger.info("=" * 80)

    analyzer = SemanticTokenAnalyzer(logger=logger)
    tokens = analyzer.load_tokens(args.semantic_tokens)
    metadata = analyzer.load_metadata(args.metadata)

    # 分析语义 tokens
    logger.info("\n语义 Token 分析:")
    analysis = analyzer.analyze_tokens(tokens)

    # 如果仅分析，则退出
    if args.analyze_only:
        logger.info("\n分析完成（--analyze_only 模式）")
        return

    # 准备文本输入
    texts = []
    if args.text:
        texts = [args.text]
    elif args.text_list:
        with open(args.text_list, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not texts:
        logger.error("未提供要合成的文本")
        logger.info("使用 --text 或 --text_list 指定文本")
        return

    logger.info(f"\n要合成的文本数: {len(texts)}")
    for i, text in enumerate(texts):
        logger.info(f"  {i + 1}. {text}")

    # 加载 VoxCPM 模型
    logger.info("\n" + "=" * 80)
    logger.info("正在加载 VoxCPM 模型...")
    logger.info("=" * 80)

    voxcpm = VoxCPM.from_pretrained(
        hf_model_id=args.hf_model_id,
        model_path=args.model_path,
        load_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
    )

    # 初始化合成器
    synthesizer = SemanticTokenSynthesizer(voxcpm, logger=logger)

    # 合成音频
    logger.info("\n" + "=" * 80)
    logger.info("开始合成音频...")
    logger.info("=" * 80)

    if len(texts) == 1:
        # 单个文本
        audio, sample_rate = synthesizer.synthesize_with_prompt(
            text=texts[0],
            prompt_wav_path=args.original_audio,
            prompt_text=metadata.get("prompt_text"),
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )

        # 保存
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, sample_rate)

        logger.info(f"\n合成音频已保存到: {output_path}")

    else:
        # 批量合成
        output_files = synthesizer.batch_synthesize(
            texts=texts,
            output_dir=Path(args.output_dir),
            prompt_wav_path=args.original_audio,
            prompt_text=metadata.get("prompt_text"),
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
        )

        logger.info(f"\n批量合成完成，共 {len(output_files)} 个文件")
        logger.info(f"输出目录: {args.output_dir}")

    logger.info("=" * 80)
    logger.info("全部完成！")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

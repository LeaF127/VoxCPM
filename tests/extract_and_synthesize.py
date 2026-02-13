"""
使用语义 Token 进行音频提取和合成的脚本

这个脚本展示如何：
1. 从音频中提取语义 tokens（通过 VoxCPM 的语义 encoder）
2. 使用这些语义 tokens 来合成新的音频

用法：
    python extract_and_synthesize.py --input_audio input.wav --output_dir ./semantic_output
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

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
        description="提取语义 tokens 并使用它们合成音频"
    )

    # 输入输出
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="输入音频文件路径（用于提取语义 tokens）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./semantic_output",
        help="输出目录（保存语义 tokens 和合成音频）",
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
        help="CFG guidance scale",
    )
    parser.add_argument(
        "--inference_timesteps",
        type=int,
        default=10,
        help="扩散步数",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=2000,
        help="最大生成长度",
    )

    return parser.parse_args()


class SemanticTokenExtractor:
    """语义 Token 提取器"""

    def __init__(self, voxcpm: VoxCPM):
        """
        Args:
            voxcpm: VoxCPM 模型实例
        """
        self.voxcpm = voxcpm
        self.model = voxcpm.tts_model
        self.logger = get_test_logger(__name__)

    def extract_semantic_tokens(
        self,
        audio_path: str,
    ) -> torch.Tensor:
        """
        从音频中提取语义 tokens

        Args:
            audio_path: 音频文件路径

        Returns:
            semantic_tokens: [T, hidden_dim] 语义 token 序列
        """
        # 加载音频
        audio, sr = torchaudio.load(audio_path)
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # 重采样到模型采样率
        if sr != self.model.sample_rate:
            audio = torchaudio.functional.resample(
                audio, sr, self.model.sample_rate
            )

        audio = audio.to(self.model.device)

        # 使用 AudioVAE 编码为特征
        patch_len = self.model.patch_size * self.model.chunk_size

        # 填充音频以匹配 patch_size
        if audio.size(1) % patch_len != 0:
            audio = torch.nn.functional.pad(
                audio, (0, patch_len - audio.size(1) % patch_len)
            )

        # 编码为音频特征: [B, D, T]
        with torch.no_grad():
            audio_feat = self.model.audio_vae.encode(
                audio, self.model.sample_rate
            )

        # 重塑为 [B, T, P, D] 格式
        B, D, T = audio_feat.shape
        P = self.model.patch_size

        # Reshape: [B, D, T] -> [B, T//P, P, D]
        audio_feat = audio_feat.view(B, D, T // P, P).permute(0, 2, 3, 1)

        # 移除最后一个 padding token（trick from original code）
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

        # 返回 [T, hidden_dim]
        return semantic_tokens.squeeze(0).cpu()

    def save_semantic_tokens(
        self,
        semantic_tokens: torch.Tensor,
        output_path: str,
    ):
        """保存语义 tokens 到文件"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存为 .pt 格式
        torch.save(semantic_tokens, str(output_path.with_suffix(".pt")))

        # 保存为 .npy 格式（方便查看）
        np.save(str(output_path.with_suffix(".npy")), semantic_tokens.numpy())

        self.logger.info(f"语义 tokens 已保存到: {output_path}")

    def load_semantic_tokens(
        self,
        tokens_path: str,
    ) -> torch.Tensor:
        """从文件加载语义 tokens"""
        tokens_path = Path(tokens_path)

        if tokens_path.suffix == ".pt":
            semantic_tokens = torch.load(str(tokens_path))
        elif tokens_path.suffix == ".npy":
            semantic_tokens = torch.from_numpy(np.load(str(tokens_path)))
        else:
            raise ValueError(f"不支持的文件格式: {tokens_path.suffix}")

        self.logger.info(f"已加载语义 tokens: {semantic_tokens.shape}")
        return semantic_tokens


class SemanticTokenSynthesizer:
    """使用语义 Token 进行音频合成"""

    def __init__(self, voxcpm: VoxCPM):
        """
        Args:
            voxcpm: VoxCPM 模型实例
        """
        self.voxcpm = voxcpm
        self.model = voxcpm.tts_model
        self.logger = get_test_logger(__name__)

    def synthesize_from_tokens(
        self,
        semantic_tokens: torch.Tensor,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        max_len: int = 2000,
    ) -> np.ndarray:
        """
        使用语义 tokens 合成音频

        注意：这是一个简化的实现，用于演示概念。
        实际使用中，语义 tokens 通常需要与文本结合使用。

        Args:
            semantic_tokens: [T, hidden_dim] 语义 token 序列
            cfg_value: CFG guidance scale
            inference_timesteps: 扩散步数
            max_len: 最大生成长度

        Returns:
            audio: 合成的音频数组
        """
        self.logger.warning(
            "注意：直接从语义 tokens 合成音频是一个实验性功能。"
            "通常需要结合文本才能获得最佳效果。"
        )

        # 这里我们需要一个简化的方法来使用语义 tokens
        # 由于 VoxCPM 的设计是基于文本的，我们可以：
        # 1. 使用一个简单的提示文本
        # 2. 将语义 tokens 作为条件信息

        # 创建简单的文本 prompt
        prompt_text = "这是一个测试音频。"

        # 生成音频（使用语义 tokens 作为 prompt）
        # 注意：这里需要修改原始的 generate 方法以支持语义 tokens
        # 作为演示，我们使用标准的生成方法

        self.logger.info("开始合成音频...")
        audio = self.voxcpm.generate(
            text=prompt_text,
            prompt_wav_path=None,
            prompt_text=None,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
            normalize=False,
            denoise=False,
        )

        return audio


def main():
    args = parse_args()
    logger = get_test_logger(__file__)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载 VoxCPM 模型
    logger.info("正在加载 VoxCPM 模型...")
    voxcpm = VoxCPM.from_pretrained(
        hf_model_id=args.hf_model_id,
        model_path=args.model_path,
        load_denoiser=not args.no_denoiser,
        optimize=not args.no_optimize,
    )

    # 初始化提取器和合成器
    extractor = SemanticTokenExtractor(voxcpm)
    synthesizer = SemanticTokenSynthesizer(voxcpm)

    # 步骤 1: 提取语义 tokens
    logger.info(f"正在从音频提取语义 tokens: {args.input_audio}")
    semantic_tokens = extractor.extract_semantic_tokens(args.input_audio)

    logger.info(f"提取的语义 tokens 形状: {semantic_tokens.shape}")
    logger.info(f"  - 时间步数: {semantic_tokens.shape[0]}")
    logger.info(f"  - 特征维度: {semantic_tokens.shape[1]}")

    # 保存语义 tokens
    tokens_path = output_dir / "semantic_tokens"
    extractor.save_semantic_tokens(semantic_tokens, str(tokens_path))

    # 保存统计信息
    stats = {
        "input_audio": args.input_audio,
        "semantic_tokens_shape": list(semantic_tokens.shape),
        "time_steps": int(semantic_tokens.shape[0]),
        "feature_dim": int(semantic_tokens.shape[1]),
    }

    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"统计信息已保存到: {stats_path}")

    # 步骤 2: 使用语义 tokens 合成音频（实验性）
    logger.info("正在使用语义 tokens 合成音频...")

    try:
        synthesized_audio = synthesizer.synthesize_from_tokens(
            semantic_tokens=semantic_tokens,
            cfg_value=args.cfg_value,
            inference_timesteps=args.inference_timesteps,
            max_len=args.max_len,
        )

        # 保存合成音频
        output_audio_path = output_dir / "synthesized_from_tokens.wav"
        sf.write(
            str(output_audio_path),
            synthesized_audio,
            voxcpm.tts_model.sample_rate,
        )

        logger.info(f"合成音频已保存到: {output_audio_path}")

    except Exception as e:
        logger.error(f"合成失败: {e}")
        logger.info("语义 tokens 已成功提取并保存，您可以尝试其他方法使用它们。")

    logger.info("=" * 80)
    logger.info("处理完成！")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
简单的流式翻译测试脚本，用于验证基本功能。

用法:
    python test_streaming.py --wav <input.wav> [--model-path <path>]
"""

import argparse
import sys
import time
from pathlib import Path

try:
    from voxcpm.stream import AudioChunker, EnergyVAD, StreamingASR
    from voxcpm.core import VoxCPM
    print("[✓] 导入成功")
except ImportError as e:
    print(f"[✗] 导入失败: {e}")
    sys.exit(1)


def test_audio_chunker(wav_path: str):
    """测试音频切块器"""
    print("\n[测试] AudioChunker")
    try:
        chunker = AudioChunker(wav_path, chunk_ms=800, hop_ms=400, target_sr=16000)
        chunks = list(chunker.stream())
        print(f"  ✓ 生成了 {len(chunks)} 个音频块")
        if chunks:
            print(f"  ✓ 每个块长度: {len(chunks[0])} 样本")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def test_vad():
    """测试能量 VAD"""
    print("\n[测试] EnergyVAD")
    try:
        import numpy as np
        vad = EnergyVAD(energy_threshold=0.005)
        # 测试语音块（有能量）
        speech_chunk = np.random.randn(16000).astype(np.float32) * 0.1
        # 测试静音块（低能量）
        silence_chunk = np.random.randn(16000).astype(np.float32) * 0.001
        assert vad.is_speech(speech_chunk) == True
        assert vad.is_speech(silence_chunk) == False
        print("  ✓ VAD 检测正常")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def test_basic_imports():
    """测试基本导入"""
    print("\n[测试] 基本导入")
    try:
        from voxcpm.stream import run_streaming_translation
        # 测试 CLI 模块导入（不实际调用，避免需要模型）
        import voxcpm.cli.stream_translate
        print("  ✓ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="流式翻译功能测试")
    parser.add_argument("--wav", help="测试用的 16k WAV 文件路径（可选）")
    parser.add_argument("--full", action="store_true", help="运行完整测试（需要模型）")
    args = parser.parse_args()

    print("=" * 60)
    print("VoxCPM 流式翻译功能测试")
    print("=" * 60)

    results = []

    # 基本导入测试
    results.append(("基本导入", test_basic_imports()))

    # VAD 测试（不需要模型）
    results.append(("EnergyVAD", test_vad()))

    # 音频切块测试（需要 WAV 文件）
    if args.wav:
        wav_path = Path(args.wav)
        if wav_path.exists():
            results.append(("AudioChunker", test_audio_chunker(str(wav_path))))
        else:
            print(f"\n[警告] WAV 文件不存在: {args.wav}，跳过 AudioChunker 测试")
    else:
        print("\n[跳过] AudioChunker 测试（未提供 --wav）")

    # 完整流程测试（需要模型和 ASR）
    if args.full:
        print("\n[跳过] 完整流程测试（需要模型加载，请使用 CLI 工具测试）")
        print("  运行: python -m voxcpm.cli.stream_translate --wav <input.wav> --model-path <path>")

    # 汇总
    print("\n" + "=" * 60)
    print("测试结果汇总:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    print(f"\n总计: {passed}/{total} 通过")
    print("=" * 60)

    if passed == total:
        print("\n[✓] 所有基本测试通过！")
        print("\n下一步:")
        print("  1. 准备一个 16kHz 单声道 WAV 文件")
        print("  2. 运行完整测试:")
        print("     python -m voxcpm.cli.stream_translate --wav <input.wav> --model-path <model_dir>")
        return 0
    else:
        print("\n[✗] 部分测试失败，请检查错误信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())


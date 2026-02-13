#!/bin/bash
# VoxCPM 语义 Token 快速开始示例
# 这个脚本演示了完整的工作流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "VoxCPM 语义 Token 快速开始示例"
echo "=========================================="

# 配置
INPUT_AUDIO="example.wav"
OUTPUT_DIR="./semantic_tokens_demo"
MODEL_PATH="${VOXCPM_MODEL_PATH:-}"

# 步骤 1: 检查输入音频
echo ""
echo "步骤 1: 检查输入音频..."
if [ ! -f "$INPUT_AUDIO" ]; then
    echo "错误: 找不到输入音频 $INPUT_AUDIO"
    echo "请提供一个测试音频文件，或修改 INPUT_AUDIO 变量"
    exit 1
fi
echo "✓ 找到输入音频: $INPUT_AUDIO"

# 步骤 2: 提取语义 tokens
echo ""
echo "步骤 2: 提取语义 tokens..."
echo "命令: python batch_semantic_extraction.py \\"
echo "  --input $INPUT_AUDIO \\"
echo "  --output_dir $OUTPUT_DIR \\"
echo "  --save_stats \\"
echo "  --visualize"

if [ -n "$MODEL_PATH" ]; then
    python batch_semantic_extraction.py \
        --input "$INPUT_AUDIO" \
        --output_dir "$OUTPUT_DIR" \
        --save_stats \
        --visualize \
        --model_path "$MODEL_PATH"
else
    python batch_semantic_extraction.py \
        --input "$INPUT_AUDIO" \
        --output_dir "$OUTPUT_DIR" \
        --save_stats \
        --visualize
fi

echo "✓ 语义 tokens 提取完成"

# 步骤 3: 分析语义 tokens
echo ""
echo "步骤 3: 分析语义 tokens..."
echo "命令: python synthesize_with_semantic_tokens.py \\"
echo "  --semantic_tokens $OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy \\"
echo "  --analyze_only"

python synthesize_with_semantic_tokens.py \
    --semantic_tokens "$OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy" \
    --analyze_only

echo "✓ 分析完成"

# 步骤 4: 使用语义 tokens 合成音频
echo ""
echo "步骤 4: 使用语义 tokens 合成新音频..."
echo "命令: python synthesize_with_semantic_tokens.py \\"
echo "  --semantic_tokens $OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy \\"
echo "  --original_audio $INPUT_AUDIO \\"
echo "  --text \"你好，这是使用语义 tokens 合成的语音\" \\"
echo "  --output $OUTPUT_DIR/synthesized.wav"

if [ -n "$MODEL_PATH" ]; then
    python synthesize_with_semantic_tokens.py \
        --semantic_tokens "$OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy" \
        --original_audio "$INPUT_AUDIO" \
        --text "你好，这是使用语义 tokens 合成的语音" \
        --output "$OUTPUT_DIR/synthesized.wav" \
        --model_path "$MODEL_PATH"
else
    python synthesize_with_semantic_tokens.py \
        --semantic_tokens "$OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy" \
        --original_audio "$INPUT_AUDIO" \
        --text "你好，这是使用语义 tokens 合成的语音" \
        --output "$OUTPUT_DIR/synthesized.wav"
fi

echo "✓ 音频合成完成"

# 完成
echo ""
echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  - 语义 tokens: $OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).npy"
echo "  - 元数据: $OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).json"
echo "  - 可视化: $OUTPUT_DIR/$(basename $INPUT_AUDIO .wav).png"
echo "  - 合成音频: $OUTPUT_DIR/synthesized.wav"
echo ""
echo "下一步:"
echo "  1. 查看可视化图片了解语义 tokens 的结构"
echo "  2. 查看元数据文件了解音频的统计信息"
echo "  3. 听取合成音频，尝试使用不同的文本"
echo "  4. 参考 README_SEMANTIC_TOKENS.md 了解更多用法"
echo ""

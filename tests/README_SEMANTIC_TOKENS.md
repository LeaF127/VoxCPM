# VoxCPM 语义 Token 提取与合成

这个目录包含了用于从音频中提取语义 tokens 并使用它们进行音频合成的脚本。

## 脚本说明

### 1. `batch_semantic_extraction.py` - 批量提取语义 Tokens

从单个或多个音频文件中提取语义 tokens。

**功能特点：**
- 支持单个文件、整个目录或文件列表输入
- 保存为 PyTorch (.pt) 和 NumPy (.npy) 格式
- 可选的可视化功能（生成热力图）
- 保存详细的元数据和统计信息

**用法示例：**

```bash
# 处理单个音频
python batch_semantic_extraction.py \
    --input test.wav \
    --output_dir ./tokens

# 处理整个目录
python batch_semantic_extraction.py \
    --input_dir ./audios \
    --output_dir ./tokens \
    --save_stats \
    --visualize

# 使用文件列表
python batch_semantic_extraction.py \
    --file_list audio_list.txt \
    --output_dir ./tokens
```

**参数说明：**
- `--input`: 单个音频文件路径
- `--input_dir`: 音频文件目录
- `--file_list`: 包含音频路径列表的文件
- `--output_dir`: 输出目录（默认: ./semantic_tokens）
- `--output_format`: 输出格式（pt/npy/both）
- `--save_stats`: 保存统计信息到 JSON
- `--visualize`: 生成可视化图
- `--max_duration`: 最大音频时长（秒）

### 2. `synthesize_with_semantic_tokens.py` - 使用语义 Token 合成音频

使用之前提取的语义 tokens 来进行音频合成。

**功能特点：**
- 分析语义 tokens 的统计特性
- 比较不同 token 序列的相似度
- 使用原始音频作为 prompt 进行语音克隆
- 支持批量合成多个文本

**用法示例：**

```bash
# 分析语义 tokens
python synthesize_with_semantic_tokens.py \
    --semantic_tokens tokens.npy \
    --analyze_only

# 使用原始音频作为 prompt 合成新文本
python synthesize_with_semantic_tokens.py \
    --semantic_tokens tokens.npy \
    --original_audio original.wav \
    --text "你好，这是测试语音" \
    --output synthesized.wav

# 批量合成
python synthesize_with_semantic_tokens.py \
    --semantic_tokens tokens.npy \
    --original_audio original.wav \
    --text_list texts.txt \
    --output_dir ./outputs
```

**参数说明：**
- `--semantic_tokens`: 语义 tokens 文件路径（必需）
- `--metadata`: 元数据文件路径
- `--original_audio`: 原始音频路径（用于语音克隆）
- `--text`: 要合成的文本
- `--text_list`: 包含多行文本的文件
- `--analyze_only`: 仅分析，不合成
- `--cfg_value`: CFG guidance scale（默认: 2.0）
- `--inference_timesteps`: 扩散步数（默认: 10）

### 3. `extract_and_synthesize.py` - 完整流程演示

展示完整的提取和合成流程（基础版本）。

## 工作流程

### 标准工作流程

1. **提取语义 Tokens**
   ```bash
   python batch_semantic_extraction.py \
       --input test.wav \
       --output_dir ./tokens \
       --save_stats \
       --visualize
   ```

   这将生成：
   - `test.npy` - NumPy 格式的语义 tokens
   - `test.pt` - PyTorch 格式的语义 tokens
   - `test.json` - 元数据（音频时长、token 形状等）
   - `test.png` - 可视化热力图（如果启用）

2. **分析语义 Tokens**
   ```bash
   python synthesize_with_semantic_tokens.py \
       --semantic_tokens ./tokens/test.npy \
       --analyze_only
   ```

   这将显示：
   - Token 形状和统计信息
   - 均值、标准差、范围等
   - 时序统计特性

3. **使用原始音频进行语音克隆**
   ```bash
   python synthesize_with_semantic_tokens.py \
       --semantic_tokens ./tokens/test.npy \
       --original_audio test.wav \
       --text "这是新的文本" \
       --output cloned.wav
   ```

### 批量处理工作流程

1. **批量提取多个音频的语义 tokens**
   ```bash
   python batch_semantic_extraction.py \
       --input_dir ./input_audios \
       --output_dir ./semantic_tokens \
       --save_stats \
       --visualize
   ```

2. **批量合成多个文本**
   ```bash
   # 创建 texts.txt
   cat > texts.txt << EOF
   这是第一句话
   这是第二句话
   这是第三句话
   EOF

   python synthesize_with_semantic_tokens.py \
       --semantic_tokens ./semantic_tokens/speaker1.npy \
       --original_audio ./input_audios/speaker1.wav \
       --text_list texts.txt \
       --output_dir ./outputs
   ```

## 输出文件说明

### 语义 Token 文件

- **.npy 文件**: NumPy 数组格式，形状为 `[T, D]`
  - `T`: 时间步数（与音频时长相关）
  - `D`: 特征维度（通常为 2048，即 LM 的隐藏维度）

- **.pt 文件**: PyTorch tensor 格式，内容与 .npy 相同

- **.json 文件**: 元数据，包含：
  ```json
  {
     "audio_path": "原始音频路径",
     "original_sample_rate": 采样率,
     "original_duration": 音频时长,
     "num_frames": 帧数,
     "token_length": token 长度,
     "feature_dim": 特征维度
  }
  ```

### 可视化文件

- **.png 文件**: 语义 tokens 的可视化
  - 上图：热力图，显示每个时间步的特征值
  - 下图：特征统计，显示均值和标准差随特征维度的变化

## 语义 Token 的理解

### 什么是语义 Tokens？

语义 tokens 是 VoxCPM 模型从音频中提取的高级语义表示。它们通过以下流程获得：

1. **音频编码**: AudioVAE 将原始音频编码为潜在表示
2. **特征提取**: 将潜在表示组织成 patches
3. **语义编码**: 通过 `VoxCPMLocEnc` (语义 encoder) 提取语义特征
4. **投影**: 将语义特征投影到语言模型空间

### 语义 Tokens 的特点

- **语义-声学解耦**: 捕获音频的语义内容，而非声学细节
- **时序结构**: 保留了音频的时序信息
- **高级表示**: 在语言模型的语义空间中，与文本 token 空间对齐

### 应用场景

1. **语音克隆**: 使用源音频的语义 tokens 作为 prompt，合成相似声音的新文本
2. **语音分析**: 分析不同说话人、情感或语言的语义表示
3. **相似度计算**: 比较不同音频的语义相似度
4. **语音检索**: 基于语义内容检索音频

## 技术细节

### VoxCPM 的架构

```
音频输入
    ↓
AudioVAE (音频编码器)
    ↓
音频特征 [B, T, P, D]
    ↓
VoxCPMLocEnc (语义 encoder)
    ↓
语义 tokens [B, T, hidden_dim]
    ↓
enc_to_lm_proj (投影层)
    ↓
LM 空间语义表示 [B, T, lm_hidden_dim]
    ↓
用于 TTS 生成的条件信息
```

### 关键参数

- `patch_size`: 通常为 2
- `chunk_size`: AudioVAE 的块大小
- `feat_dim`: 音频特征维度（64）
- `hidden_dim`: 语义 encoder 的隐藏维度（1024）
- `lm_hidden_dim`: 语言模型的隐藏维度（2048）

## 常见问题

### Q: 为什么要提取语义 tokens？

A: 语义 tokens 提供了音频的高级语义表示，可以用于：
- 语音分析和理解
- 语音相似度计算
- 作为语音克隆的条件信息
- 语音检索和分类

### Q: 可以直接从语义 tokens 重建音频吗？

A: 不可以直接重建。语义 tokens 只包含了语义信息，不包含完整的声学细节。
要重建音频，需要使用 VoxCPM 的生成流程，结合文本和语义 tokens。

### Q: 语义 tokens 和 AudioVAE 的潜在表示有什么区别？

A:
- **AudioVAE 潜在表示**: 低级的声学特征，包含频谱信息
- **语义 tokens**: 通过 Transformer encoder 提取的高级语义表示，在语义空间中

### Q: 如何选择 CFG value 和推理步数？

A:
- **CFG value**: 控制生成质量与多样性的平衡
  - 较高值（2.0-3.0）: 更好的质量，但可能过度平滑
  - 较低值（1.0-1.5）: 更多样化，但可能质量不稳定
- **推理步数**: 影响生成质量和速度
  - 10 步: 快速，质量良好（推荐）
  - 20-30 步: 更高质量，但更慢

## 示例

### 示例 1: 提取单个音频的语义 tokens

```bash
python batch_semantic_extraction.py \
    --input example.wav \
    --output_dir ./tokens \
    --save_stats \
    --visualize
```

输出：
```
tokens/example.npy
tokens/example.pt
tokens/example.json
tokens/example.png
```

### 示例 2: 语音克隆

```bash
# 1. 提取语义 tokens
python batch_semantic_extraction.py \
    --input voice_sample.wav \
    --output_dir ./tokens

# 2. 使用语义 tokens 和原始音频进行语音克隆
python synthesize_with_semantic_tokens.py \
    --semantic_tokens ./tokens/voice_sample.npy \
    --original_audio voice_sample.wav \
    --text "今天天气很好" \
    --output cloned_voice.wav
```

### 示例 3: 批量处理

```bash
# 1. 批量提取
python batch_semantic_extraction.py \
    --input_dir ./speaker_audios \
    --output_dir ./semantic_tokens \
    --save_stats

# 2. 批量合成
python synthesize_with_semantic_tokens.py \
    --semantic_tokens ./semantic_tokens/speaker1.npy \
    --original_audio ./speaker_audios/speaker1.wav \
    --text_list texts_to_synthesize.txt \
    --output_dir ./synthesized_outputs
```

## 依赖

- torch
- torchaudio
- numpy
- soundfile
- tqdm
- matplotlib (用于可视化)
- voxcpm (VoxCPM 模型)

## 注意事项

1. **内存使用**: VoxCPM 模型较大，建议使用 GPU 运行
2. **音频格式**: 支持常见格式（.wav, .mp3, .flac 等），会自动转换为 16kHz
3. **音频时长**: 对于很长的音频，可以考虑先分段处理
4. **模型加载**: 首次运行会从 Hugging Face 下载模型

## 故障排除

### 问题: CUDA out of memory

解决方案：
- 使用更小的 batch size
- 使用 `--no_optimize` 禁用优化（减少内存）
- 处理更短的音频

### 问题: 模型加载慢

解决方案：
- 使用 `--model_path` 指定本地模型路径
- 使用 `--no_denoiser` 跳过降噪模型加载

### 问题: 合成音频质量不佳

解决方案：
- 调整 `--cfg_value`（尝试 1.5-3.0）
- 增加 `--inference_timesteps`（尝试 15-20）
- 确保使用高质量的 prompt 音频

## 相关文档

- VoxCPM 论文和模型: [GitHub Repository]
- AudioVAE 文档: 参见模型源码
- MiniCPM-4 文档: 参见模型源码

# VoxCPM 流式语音翻译快速上手

## 环境依赖
- Python ≥ 3.9
- funasr（ASR）: `pip install funasr`
- sounddevice（可选实时播放）: `pip install sounddevice`
- 已下载 VoxCPM 模型目录（示例: `./models/VoxCPM-0.5B`）

## 基本命令
```bash
python -m voxcpm.cli.stream_translate \
  --wav ./examples/example.wav \
  --model-path ./models/VoxCPM-0.5B \
  --prompt-wav ./examples/example.wav \
  --prompt-text "这是参考文本" \
  --chunk-ms 800 --hop-ms 400 \
  --energy-threshold 0.005 \
  --output ./outputs/stream_out.wav
```

参数要点：
- `--chunk-ms/--hop-ms`: ASR 切片窗口与步长（默认 800/400ms）
- `--prompt-wav/--prompt-text`: 可选，用于保持音色
- `--energy-threshold`: 简单能量 VAD 阈值
- `--no-play`: 仅写文件，不实时播放

## 数据流
- ASR: SenseVoiceSmall 增量识别，静音触发分段
- MT: 目前为直通占位函数，可按需替换
- TTS: `generate_with_prompt_cache_streaming` 连续输出音频块

## 测试验证

运行基本功能测试（不需要模型）:
```bash
python test_streaming.py --wav ./examples/example.wav
```

完整流程测试（需要模型）:
```bash
python -m voxcpm.cli.stream_translate \
  --wav ./examples/example.wav \
  --model-path ./models/VoxCPM-0.5B
```

## 常见问题
- 未安装 CUDA 时默认使用 CPU，延迟会升高
- 若未安装 `funasr` 会提示缺失，请先安装
- 需要 16k 单声道 PCM wav 输入；其他采样率请先转换
- 首次运行会下载 ASR 模型（SenseVoiceSmall），请确保网络连接




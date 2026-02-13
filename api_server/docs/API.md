# VoxCPM S2ST API 接口文档

## 概述

VoxCPM S2ST API 提供基于 FastAPI 的 RESTful 接口，支持流式语音识别、翻译和语音合成（Speech-to-Speech Translation, S2ST）功能。

### 特性

- **流式处理**：使用 Server-Sent Events (SSE) 实时返回识别、翻译和合成结果
- **逐句处理**：每完成一句识别+翻译+TTS，立即返回结果
- **自动拼接**：处理完成后自动拼接所有片段为完整音频文件
- **语音克隆**：支持使用原始音频片段作为 prompt，实现语音克隆效果

### 基础信息

- **Base URL**: `http://localhost:8000`
- **API 版本**: `v0.1.0`
- **协议**: HTTP/1.1, Server-Sent Events

---

## 接口列表

### 1. 根路径
1
**GET** `/`

返回 API 基本信息。

#### 响应示例

```json
{
  "name": "VoxCPM S2ST API",
  "version": "0.1.0",
  "description": "语音识别、翻译与语音合成流式 API",
  "endpoints": {
    "/streaming": "POST - 流式处理音频文件（Server-Sent Events）",
    "/health": "GET - 健康检查"
  }
}
```

---

### 2. 健康检查

**GET** `/health`

检查服务状态和模型加载情况。

#### 响应示例

```json
{
  "status": "ok",
  "tts_model_loaded": true
}
```

---

### 3. 流式 S2ST 处理

**POST** `/streaming`

上传音频文件，实时返回识别、翻译和合成结果。

#### 请求格式

- **Content-Type**: `multipart/form-data`
- **响应格式**: `text/event-stream` (Server-Sent Events)

#### 请求参数

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `audio_file` | File | 是 | - | 音频文件（建议 16kHz 单声道 WAV） |
| `ws_url` | string | 否 | `ws://175.24.179.12:9301/dotcwsasr` | ASR WebSocket 服务地址 |
| `user_id` | string | 否 | `y123456` | 用户 ID |
| `token` | string | 否 | `token12345-1730889600` | 认证 Token |
| `from_lang` | string | 否 | `zh` | 源语言代码（如：zh, en） |
| `to_lang` | string | 否 | `en` | 目标语言代码（如：zh, en） |
| `role` | string | 否 | `0` | 角色 ID |
| `lan_id` | string | 否 | `0` | 语言 ID |
| `sample_rate` | integer | 否 | `16000` | 音频采样率（Hz） |
| `bit_rate` | integer | 否 | `16` | 音频位深度（bit） |
| `interval` | float | 否 | `0.1` | 音频块间隔（秒） |
| `tts_text_source` | string | 否 | `trans` | TTS 文本来源：`trans`=翻译结果，`asr`=识别结果 |
| `cfg_value` | float | 否 | `2.0` | VoxCPM CFG 值（控制生成质量） |
| `inference_timesteps` | integer | 否 | `10` | VoxCPM 扩散步数（影响生成速度和质量） |
| `normalize` | boolean | 否 | `true` | 启用文本正则化 |
| `denoise` | boolean | 否 | `false` | 对 prompt 音频降噪 |
| `prompt_wav_path` | string | 否 | `null` | 参考音频路径（可选，用于语音克隆） |
| `prompt_text` | string | 否 | `null` | 参考音频对应文本（可选） |

#### 响应事件类型

响应使用 Server-Sent Events (SSE) 格式，每个事件为一行 JSON 数据，格式如下：

```json
{
  "event_type": "segment|progress|error|complete",
  "data": { ... }
}
```

##### 事件类型说明

1. **`segment`** - 单句处理完成
   ```json
   {
     "event_type": "segment",
     "data": {
       "segment_index": 0,
       "asr_text": "识别文本",
       "trans_text": "翻译文本",
       "tts_text": "用于合成的文本",
       "begin_ms": 0,
       "end_ms": 1500,
       "segment_file": "segments/{request_id}/segment_000.wav",
       "duration": 1.5
     }
   }
   ```

2. **`progress`** - 进度信息（可选）
   ```json
   {
     "event_type": "progress",
     "data": {
       "message": "处理进度信息"
     }
   }
   ```

3. **`error`** - 错误信息
   ```json
   {
     "event_type": "error",
     "data": {
       "message": "错误描述"
     }
   }
   ```

4. **`complete`** - 全部处理完成
   ```json
   {
     "event_type": "complete",
     "data": {
       "output_file": "s2st_demo/output/{request_id}_final.wav",
       "duration": 10.5,
       "segment_count": 5
     }
   }
   ```

#### 请求示例

##### cURL

```bash
curl -X POST "http://localhost:8000/streaming" \
  -F "audio_file=@example.wav" \
  -F "from_lang=zh" \
  -F "to_lang=en" \
  -F "tts_text_source=trans" \
  -F "normalize=true"
```

##### Python (requests)

```python
import requests

url = "http://localhost:8000/streaming"
files = {"audio_file": open("example.wav", "rb")}
data = {
    "from_lang": "zh",
    "to_lang": "en",
    "tts_text_source": "trans",
    "normalize": "true"
}

response = requests.post(url, files=files, data=data, stream=True)

for line in response.iter_lines():
    if line:
        # 解析 SSE 事件
        if line.startswith(b"data: "):
            import json
            event_data = json.loads(line[6:])  # 去掉 "data: " 前缀
            event_type = event_data["event_type"]
            data = event_data["data"]
            
            if event_type == "segment":
                print(f"句 {data['segment_index']}: {data['asr_text']} -> {data['trans_text']}")
            elif event_type == "complete":
                print(f"完成！输出文件: {data['output_file']}")
            elif event_type == "error":
                print(f"错误: {data['message']}")
```

##### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('audio_file', fileInput.files[0]);
formData.append('from_lang', 'zh');
formData.append('to_lang', 'en');
formData.append('tts_text_source', 'trans');
formData.append('normalize', 'true');

const response = await fetch('http://localhost:8000/streaming', {
  method: 'POST',
  body: formData
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  
  const chunk = decoder.decode(value);
  const lines = chunk.split('\n');
  
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const eventData = JSON.parse(line.slice(6));
      const { event_type, data } = eventData;
      
      if (event_type === 'segment') {
        console.log(`句 ${data.segment_index}: ${data.asr_text} -> ${data.trans_text}`);
      } else if (event_type === 'complete') {
        console.log(`完成！输出文件: ${data.output_file}`);
      } else if (event_type === 'error') {
        console.error(`错误: ${data.message}`);
      }
    }
  }
}
```

---

## 使用流程

### 1. 启动服务

```bash
# 方式 1：直接运行
python s2st_demo/api_server.py

# 方式 2：使用 uvicorn
uvicorn s2st_demo.api_server:app --host 0.0.0.0 --port 8000

# 方式 3：指定模型路径（环境变量）
export VOXCPM_MODEL_PATH="./models/VoxCPM-0.5B"
uvicorn s2st_demo.api_server:app --host 0.0.0.0 --port 8000
```

### 2. 环境变量配置

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `VOXCPM_MODEL_PATH` | 本地 VoxCPM 模型目录 | - |
| `VOXCPM_HF_ID` | Hugging Face 模型 ID | `openbmb/VoxCPM-0.5B` |
| `VOXCPM_CACHE_DIR` | 模型缓存目录 | - |
| `VOXCPM_LOCAL_FILES_ONLY` | 仅使用本地文件 | `false` |
| `VOXCPM_NO_DENOISER` | 禁用降噪模型 | `false` |
| `VOXCPM_NO_OPTIMIZE` | 禁用优化 | `false` |

### 3. 调用接口

参考上述请求示例，上传音频文件并接收流式响应。

---

## 输出文件说明

### 文件结构

```
s2st_demo/output/
├── {request_id}_final.wav          # 最终拼接的完整音频
└── segments/
    └── {request_id}/
        ├── segment_000.wav          # 第 1 句合成音频
        ├── segment_001.wav          # 第 2 句合成音频
        └── ...
```

### 文件访问

- **最终音频**: `s2st_demo/output/{request_id}_final.wav`
- **片段音频**: `s2st_demo/output/segments/{request_id}/segment_XXX.wav`

`request_id` 在响应事件的 `complete` 事件中返回，或从输出文件路径中提取。

---

## 错误处理

### 常见错误

1. **400 Bad Request**
   - 音频文件格式不支持
   - 音频文件损坏
   - 参数格式错误

2. **500 Internal Server Error**
   - 模型加载失败
   - ASR 服务连接失败
   - TTS 合成失败

### 错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

在 SSE 流中，错误会以 `error` 事件类型返回：

```json
{
  "event_type": "error",
  "data": {
    "message": "具体错误信息"
  }
}
```

---

## 性能与限制

### 性能指标

- **单句处理时间**: 约 2-5 秒（取决于文本长度和硬件）
- **并发请求**: 建议不超过 5 个（取决于 GPU 内存）
- **音频文件大小**: 建议不超过 100MB

### 限制

- 音频格式：建议使用 16kHz 单声道 WAV
- 最大音频时长：建议不超过 10 分钟
- 并发处理：受 GPU 内存限制

---

## 最佳实践

1. **音频预处理**
   - 确保音频为 16kHz 单声道
   - 去除背景噪音（可选）
   - 控制音频时长（建议 < 5 分钟）

2. **参数调优**
   - `cfg_value`: 2.0-3.0 之间效果较好
   - `inference_timesteps`: 10-20 之间，数值越大质量越好但速度越慢
   - `normalize`: 建议开启，特别是处理中文文本时

3. **错误处理**
   - 监听 `error` 事件并记录日志
   - 实现重试机制（对于网络错误）
   - 设置合理的超时时间

4. **资源管理**
   - 及时清理临时文件
   - 控制并发请求数量
   - 监控 GPU 内存使用

---

## 更新日志

### v0.1.0 (2025-01-XX)

- 初始版本
- 支持流式 S2ST 处理
- 支持 Server-Sent Events 响应
- 支持语音克隆（使用原始音频片段作为 prompt）

---

## 技术支持

如有问题或建议，请提交 Issue 或联系维护团队。


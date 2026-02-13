# VoxCPM S2ST API 接口文档

## 概述

本文档描述了 VoxCPM 端到端语音翻译 API 的接口规范，该接口已与 `s2st_demo/` 前端完全对齐。

---

## 接口列表

| 端点 | 方法 | 描述 |
|------|------|------|
| `/s2st` | POST | 端到端语音翻译（前端主接口） |
| `/streaming` | POST | 流式语音翻译（SSE） |
| `/config` | GET | 获取服务器配置 |
| `/health` | GET | 健康检查 |
| `/static/{filename}` | GET | 获取静态文件（音频） |

---

## 主接口：POST /s2st

### 描述

接收音频文件，返回端到端翻译后的语音。此接口与 `s2st_demo/` 前端完全兼容。

### 请求格式

- **Content-Type**: `multipart/form-data`
- **参数位置**: form-data

### 请求参数

| 字段名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| `audio` | File | 是 | - | 音频文件（WAV、MP3、WebM 等） |
| `from_lang` | string | 否 | `zh` | 源语言代码（zh/en） |
| `to_lang` | string | 否 | `en` | 目标语言代码（zh/en） |
| `tts_text_source` | string | 否 | `trans` | TTS 文本来源：`trans`=翻译结果，`asr`=识别结果 |
| `cfg_value` | float | 否 | `2.0` | VoxCPM CFG 值（控制语音相似度） |
| `inference_timesteps` | int | 否 | `10` | VoxCPM 推扩散步数 |

### 环境变量配置

以下环境变量需在服务器端配置（通过 `.env` 文件或系统环境变量）：

| 环境变量 | 必填 | 描述 | 示例 |
|----------|------|------|------|
| `WS_URL` | 是 | ASR WebSocket 服务地址 | `ws://asr-server:port/ws` |
| `USER_ID` | 是 | ASR 服务用户 ID | `your_user_id` |
| `TOKEN` | 是 | ASR 认证 Token | `your_token` |
| `FROM_LANG` | 否 | 默认源语言 | `zh` |
| `TO_LANG` | 否 | 默认目标语言 | `en` |
| `TTS_TEXT_SOURCE` | 否 | 默认 TTS 文本来源 | `trans` |
| `CFG_VALUE` | 否 | 默认 CFG 值 | `2.0` |
| `INFERENCE_TIMESTEPS` | 否 | 默认推理步数 | `10` |

### 响应格式

- **Content-Type**: `application/json`

### 响应字段

| 字段名 | 类型 | 必填 | 描述 |
|--------|------|------|------|
| `audio` | string | 是 | 翻译后的音频 URL（可通过 `/static/{filename}` 访问） |
| `src_text` | string | 否 | 源语言识别文本（ASR 结果） |
| `tgt_text` | string | 否 | 目标语言翻译文本 |

### 响应示例

#### 成功响应（包含文本）

```json
{
  "audio": "/static/abc123_output.wav",
  "src_text": "你好世界",
  "tgt_text": "Hello World"
}
```

#### 成功响应（无文本）

```json
{
  "audio": "/static/abc123_output.wav"
}
```

#### 错误响应

```json
{
  "detail": "模型初始化失败: ..."
}
```

### HTTP 状态码

| 状态码 | 描述 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

### 请求示例

#### cURL

```bash
curl -X POST http://localhost:19366/s2st \
  -F "audio=@input.wav" \
  -F "from_lang=zh" \
  -F "to_lang=en"
```

#### Python (requests)

```python
import requests

with open('input.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:19366/s2st',
        files={'audio': f},
        data={
            'from_lang': 'zh',
            'to_lang': 'en'
        }
    )
    result = response.json()
    print(result['audio'])  # 音频 URL
    print(result['src_text'])  # 原文（可选）
    print(result['tgt_text'])  # 译文（可选）
```

#### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append('audio', audioBlob);

const response = await fetch('http://localhost:19366/s2st', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result.audio);  // 音频 URL
console.log(result.src_text);  // 原文（可选）
console.log(result.tgt_text);  // 译文（可选）
```

---

## 流式接口：POST /streaming

### 描述

流式语音翻译接口，使用 Server-Sent Events (SSE) 实时返回处理进度。

### 请求格式

- **Content-Type**: `multipart/form-data`

### 请求参数

除了 `/s2st` 接口的参数外，还支持：

| 字段名 | 类型 | 必填 | 默认值 | 描述 |
|--------|------|------|--------|------|
| `audio_file` | File | 是 | - | 音频文件 |
| `ws_url` | string | 条件* | - | ASR WebSocket 地址（可通过环境变量） |
| `user_id` | string | 条件* | - | 用户 ID（可通过环境变量） |
| `token` | string | 条件* | - | 认证 Token（可通过环境变量） |
| `normalize` | bool | 否 | `true` | 启用文本正则化 |
| `denoise` | bool | 否 | `false` | 对 prompt 音频降噪 |
| `prompt_wav_path` | string | 否 | `null` | 参考音频路径 |
| `prompt_text` | string | 否 | `null` | 参考音频对应文本 |

\* 如果环境变量中已配置，请求参数可省略。

### 响应格式

- **Content-Type**: `text/event-stream`

### 事件类型

| 事件类型 | 描述 |
|----------|------|
| `segment` | 单句处理完成 |
| `progress` | 处理进度 |
| `error` | 处理错误 |
| `complete` | 全部完成 |

### 事件示例

```javascript
// segment 事件
data: {"event_type":"segment","data":{"segment_index":0,"asr_text":"你好","trans_text":"Hello",...}}

// progress 事件
data: {"event_type":"progress","data":{"message":"处理中..."}}

// complete 事件
data: {"event_type":"complete","data":{"output_file":"s2st_demo/output/abc_final.wav","duration":5.2,"segment_count":2}}

// error 事件
data: {"event_type":"error","data":{"message":"处理失败: ..."}}
```

---

## 配置接口：GET /config

### 描述

获取服务器配置（供前端使用）。

### 响应示例

```json
{
  "ws_url": "ws://asr-server:port/ws",
  "user_id": "your_user_id",
  "token": "your_token",
  "from_lang": "zh",
  "to_lang": "en",
  "role": "0",
  "lan_id": "0",
  "tts_text_source": "trans",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "silence_threshold": -40,
  "silence_duration": 1.0,
  "min_duration": 0.5
}
```

---

## 健康检查：GET /health

### 描述

检查服务健康状态。

### 响应示例

```json
{
  "status": "ok",
  "tts_model_loaded": true
}
```

---

## 音频格式要求

### 输入音频

| 参数 | 建议值 |
|------|--------|
| 文件格式 | WAV, WebM, MP3 |
| 采样率 | 16000 Hz |
| 声道数 | 1（单声道） |
| 位深度 | 16-bit |
| 时长建议 | < 60 秒 |

### 输出音频

| 参数 | 值 |
|------|------|
| 文件格式 | WAV |
| 采样率 | 24000 Hz 或 16000 Hz |
| 声道数 | 1（单声道） |
| 位深度 | 16-bit PCM |

---

## 静态文件访问

音频文件通过 `/static/{filename}` 路径访问。

示例：
- 输出文件：`s2st_demo/output/abc123_output.wav`
- 访问 URL：`http://localhost:19366/static/abc123_output.wav`

---

## 错误处理

所有错误响应格式：

```json
{
  "detail": "错误描述信息"
}
```

### 常见错误

| 错误信息 | 原因 | 解决方法 |
|----------|------|----------|
| `服务器配置错误：WS_URL 环境变量未设置` | 未配置 ASR 服务地址 | 设置 `WS_URL` 环境变量 |
| `模型初始化失败` | VoxCPM 模型加载失败 | 检查模型路径和依赖 |
| `保存音频文件失败` | 文件系统错误 | 检查磁盘空间和权限 |
| `语音合成失败` | TTS 处理错误 | 检查输入音频格式 |

---

## CORS 支持

所有接口已启用 CORS（跨域资源共享），允许来自任何源的请求。

配置：
```python
allow_origins=["*"]
allow_credentials=True
allow_methods=["*"]
allow_headers=["*"]
```

---

## 与 s2st_demo/ 前端对齐说明

### s2st_demo/config.json 配置

```json
{
  "api": {
    "url": "http://localhost:19366",
    "endpoint": "/s2st",
    "timeout": 30000
  },
  "response": {
    "audioField": "audio",
    "audioType": "url",
    "sourceTextField": "src_text",
    "targetTextField": "tgt_text"
  }
}
```

### 字段映射

| s2st_demo 字段 | API 返回字段 |
|----------------|-------------|
| `audio` | `audio` |
| `src_text` | `src_text` |
| `tgt_text` | `tgt_text` |

### 音频 URL 格式

- **类型**: `url`
- **格式**: `/static/{filename}`
- **访问**: 通过 API 服务器的 `/static` 路径

---

## 完整工作流程

```
1. 前端上传音频 → POST /s2st
                    ↓
2. 服务器保存临时文件
                    ↓
3. 调用 ASR WebSocket 服务（识别+翻译）
                    ↓
4. 使用 VoxCPM 模型合成语音
                    ↓
5. 保存输出文件到 s2st_demo/output/
                    ↓
6. 返回 JSON：{audio: "/static/xxx.wav", src_text: "...", tgt_text: "..."}
                    ↓
7. 前端通过 /static/xxx.wav 播放音频
```

---

## 性能建议

1. **模型初始化**: TTS 模型采用单例模式，首次请求后会缓存
2. **文件清理**: 临时文件在处理完成后自动删除，输出文件定期清理
3. **并发处理**: 使用异步处理，支持并发请求
4. **超时设置**: 默认请求超时 30 秒，可根据需要调整

---

## 测试方法

### 使用 curl 测试

```bash
# 测试 /s2st 接口
curl -X POST http://localhost:19366/s2st \
  -F "audio=@test.wav" \
  -F "from_lang=zh" \
  -F "to_lang=en"

# 测试健康检查
curl http://localhost:19366/health

# 测试配置获取
curl http://localhost:19366/config
```

### 使用 Python 测试

```python
import requests

# 测试 /s2st
with open('test.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:19366/s2st',
        files={'audio': f}
    )
    print(response.json())

# 测试健康检查
print(requests.get('http://localhost:19366/health').json())
```

---

## 技术栈

- **Web 框架**: FastAPI
- **ASR 服务**: WebSocket 客户端
- **TTS 模型**: VoxCPM (openbmb/VoxCPM-0.5B)
- **音频处理**: soundfile, numpy
- **异步处理**: asyncio

---

## 相关文档

- [s2st_demo/API_FORMAT.md](../s2st_demo/API_FORMAT.md) - 前端接口要求
- [s2st_demo/README.md](../s2st_demo/README.md) - 前端使用说明
- [api_server/README.md](README.md) - API 服务说明（如有）

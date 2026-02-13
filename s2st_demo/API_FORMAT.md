# 端到端语音翻译 API 接口说明

## 接口地址

```
POST /s2st
```

完整 URL: `http://your-api-host:port/s2st`

---

## 请求格式

### Content-Type
```
multipart/form-data
```

### 请求参数

| 字段名 | 类型 | 必填 | 说明 | 示例 |
|--------|------|------|------|------|
| `audio` | File | 是 | 输入音频文件（WAV、WebM 等） | - |

### 请求示例

#### cURL
```bash
curl -X POST http://localhost:8000/s2st \
  -F "audio=@input.wav"
```

#### Python (requests)
```python
import requests

with open('input.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/s2st',
        files={'audio': f}
    )
    result = response.json()
```

#### JavaScript (fetch)
```javascript
const formData = new FormData();
formData.append('audio', audioBlob, 'input.wav');

const response = await fetch('http://localhost:8000/s2st', {
    method: 'POST',
    body: formData
});

const result = await response.json();
```

---

## 响应格式

### Content-Type
```
application/json
```

### 响应字段

| 字段名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `audio` | String/Object | 是 | 翻译后的音频（URL 或 base64） |
| `src_text` | String | 否 | 源语言识别文本（可选） |
| `tgt_text` | String | 否 | 目标语言翻译文本（可选） |

---

## 响应示例

### 方式一：音频以 URL 形式返回（推荐）

```json
{
  "audio": "http://localhost:8000/static/output/translation_12345.wav",
  "src_text": "你好世界",
  "tgt_text": "Hello World"
}
```

### 方式二：音频以 base64 形式返回

```json
{
  "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=",
  "src_text": "你好世界",
  "tgt_text": "Hello World"
}
```

前端需要拼接前缀：`data:audio/wav;base64,{audio}`

### 方式三：仅返回音频（无文本）

```json
{
  "audio": "http://localhost:8000/static/output/translation_12345.wav"
}
```

**注意**：这是正常情况，前端应正常处理，不应视为错误。

---

## 音频格式要求

### 输入音频

| 参数 | 建议值 |
|------|--------|
| 文件格式 | WAV, WebM, MP3 |
| 采样率 | 16000 Hz |
| 声道数 | 1（单声道） |
| 位深度 | 16-bit |
| 时长建议 | < 60 秒（演示环境） |

### 输出音频

| 参数 | 建议值 |
|------|--------|
| 文件格式 | WAV |
| 采样率 | 24000 Hz 或 16000 Hz |
| 声道数 | 1（单声道） |
| 位深度 | 16-bit PCM |

---

## 前端配置说明

前端 `CONFIG` 对象需要根据实际后端返回格式进行调整：

```javascript
const CONFIG = {
    RESPONSE_FIELDS: {
        // 音频字段名称
        AUDIO: 'audio',

        // 音频格式类型
        // - 'url': 后端返回音频 URL
        // - 'base64': 后端返回 base64 编码
        // - 'binary': 后端直接返回二进制文件
        AUDIO_TYPE: 'url',

        // 可选文本字段（如果后端返回的字段名不同，修改这里）
        SOURCE_TEXT: 'src_text',
        TARGET_TEXT: 'tgt_text'
    }
};
```

### 配置示例

#### 如果后端返回字段名不同

假设后端返回：
```json
{
  "output_audio": "http://...",
  "original": "你好",
  "translated": "Hello"
}
```

则配置为：
```javascript
RESPONSE_FIELDS: {
    AUDIO: 'output_audio',
    SOURCE_TEXT: 'original',
    TARGET_TEXT: 'translated'
}
```

#### 如果后端返回 base64

```javascript
RESPONSE_FIELDS: {
    AUDIO_TYPE: 'base64'  // 前端会自动拼接 data URI
}
```

---

## 错误响应

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | 请求参数错误（如音频格式不支持） |
| 500 | 服务器内部错误（如模型推理失败） |

### 错误响应示例

```json
{
  "error": "音频格式不支持",
  "code": "UNSUPPORTED_FORMAT"
}
```

或

```json
{
  "error": "模型推理超时",
  "code": "INFERENCE_TIMEOUT"
}
```

---

## 完整示例

### 后端示例 (Python Flask)

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import base64

app = Flask(__name__)
CORS(app)  # 允许跨域请求

@app.route('/s2st', methods=['POST'])
def speech_to_speech_translation():
    # 获取上传的音频文件
    audio_file = request.files.get('audio')

    if not audio_file:
        return jsonify({'error': '缺少音频文件'}), 400

    # TODO: 调用端到端模型进行翻译
    # output_audio, src_text, tgt_text = your_s2st_model.process(audio_file)

    # 模拟输出（实际使用时替换为模型输出）
    output_audio_url = "http://localhost:8000/static/output/demo.wav"
    src_text = "你好世界"  # 可选
    tgt_text = "Hello World"  # 可选

    return jsonify({
        'audio': output_audio_url,
        'src_text': src_text,
        'tgt_text': tgt_text
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
```

### 后端示例 (FastAPI)

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/s2st")
async def speech_to_speech_translation(audio: UploadFile = File(...)):
    # 读取音频文件
    audio_content = await audio.read()

    # TODO: 调用端到端模型
    # output_audio, src_text, tgt_text = await your_s2st_model.process(audio_content)

    # 返回结果
    return {
        "audio": "http://localhost:8000/static/output.wav",
        "src_text": "你好世界",  # 可选
        "tgt_text": "Hello World"  # 可选
    }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## 性能建议

1. **异步处理**：端到端模型推理较慢（通常 2-10 秒），建议使用异步任务队列
2. **流式返回**：考虑使用 Server-Sent Events 或 WebSocket 流式返回音频片段
3. **缓存结果**：对相同输入的翻译结果进行缓存
4. **音频压缩**：返回音频时可考虑使用 MP3 格式减小体积

---

## 测试

### 测试用音频

建议准备以下测试音频：
- 短中文语音（3-5 秒）：如 "你好世界"
- 短英文语音（3-5 秒）：如 "Hello World"
- 较长语音（10-20 秒）：测试模型处理能力

### 浏览器控制台调试

打开浏览器开发者工具（F12），在 Network 面板查看：
- 请求是否成功发送
- 响应数据格式是否正确
- 音频 URL 是否可访问

在 Console 面板查看：
- 是否有 JavaScript 错误
- 响应解析是否成功

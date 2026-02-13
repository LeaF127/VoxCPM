# 端到端语音翻译 Web Demo

极简的 Speech-to-Speech Translation 演示页面，用于展示端到端语音翻译模型能力。

**配置分离设计**：后端配置存储在服务器端，前端运行时加载，不在源码中暴露真实接口地址。

---

## 演示效果

**核心功能**：
- 说中文 → 听到英文语音
- 说英文 → 听到中文语音

**特点**：
- 无需文本中转，直接生成目标语音
- 即使后端不返回文本，Demo 仍完整可用
- 麦克风录音 + 文件上传双模式
- **配置与前端分离**：不在源码中暴露后端地址

---

## 快速启动

```bash
cd voxcpm/s2st_demo
python server.py
```

访问：http://localhost:8080

---

## 修改配置

编辑 `config.json` 文件：

```json
{
  "api": {
    "url": "http://localhost:8000",    // 修改为你的后端地址
    "endpoint": "/s2st"                 // 修改为你的接口路径
  }
}
```

保存后刷新浏览器即可生效，**无需重启服务器**。

---

## 使用步骤

### 1. 配置后端地址（首次使用）

编辑 `config.json`，填写后端接口信息。

### 2. 选择输入方式

#### 麦克风录音
1. 点击「开始录音」
2. 对着麦克风说话（中文或英文）
3. 点击「停止录音」
4. 等待处理，自动播放翻译后的语音

#### 文件上传
1. 切换到「文件上传」标签
2. 选择音频文件（WAV、MP3 等）
3. 点击「提交处理」
4. 等待处理，自动播放翻译后的语音

### 3. 查看结果

- **主演示结果**：音频播放器自动播放翻译语音
- **文本结果**（可选）：如果后端返回了文本，会显示原文和译文

---

## 目录结构

```
voxcpm/s2st_demo/
├── config.json       # 配置文件（服务器端，不暴露给前端）
├── server.py         # 配置服务器（提供 /config 接口）
├── index.html        # 前端页面（运行时加载配置）
├── CONFIG_GUIDE.md   # 配置分离详细说明
├── API_FORMAT.md     # 后端接口格式说明
├── README.md         # 本文件
└── NOTES.md          # 注意事项与常见问题
```

---

## 配置分离说明

### 为什么这样设计？

**传统方式的问题**：
```html
<!-- 前端源码中直接暴露配置 -->
<script>
const API_URL = "http://internal-server:8000";  // ❌ 所有人可见
</script>
```

**配置分离方式**：
```javascript
// 前端只从固定路径加载配置
const config = await fetch('/config');          // ✅ 真实地址在服务器端
```

### 安全性对比

| 用户类型 | 传统方式 | 配置分离方式 |
|---------|---------|-------------|
| 普通用户（查看源码） | ✅ 看到后端地址 | ❌ 看不到 |
| 开发者（F12 Network） | ✅ 看到后端地址 | ✅ 看到后端地址 |

**结论**：配置分离可以防止普通用户（客户、演示观众）看到内部服务器地址，但不能防止技术人员通过开发者工具查看。

### 详细说明

详见 [CONFIG_GUIDE.md](./CONFIG_GUIDE.md)

---

## 配置项说明

### config.json 完整示例

```json
{
  "api": {
    "url": "http://localhost:8000",
    "endpoint": "/s2st",
    "timeout": 60000
  },
  "audio": {
    "maxFileSize": 104857600,
    "sampleRate": 16000,
    "channels": 1
  },
  "response": {
    "audioField": "audio",
    "audioType": "url",
    "sourceTextField": "src_text",
    "targetTextField": "tgt_text"
  }
}
```

### 配置项详解

| 配置组 | 字段 | 说明 | 默认值 |
|--------|------|------|--------|
| `api` | `url` | 后端 API 地址 | `http://localhost:8000` |
| `api` | `endpoint` | 接口路径 | `/s2st` |
| `api` | `timeout` | 请求超时（毫秒） | `60000` |
| `audio` | `maxFileSize` | 最大文件大小（字节） | `104857600` (100MB) |
| `audio` | `sampleRate` | 音频采样率 | `16000` |
| `audio` | `channels` | 音频声道数 | `1` |
| `response` | `audioField` | 音频字段名 | `audio` |
| `response` | `audioType` | 音频类型（url/base64） | `url` |
| `response` | `sourceTextField` | 原文字段名 | `src_text` |
| `response` | `targetTextField` | 译文字段名 | `tgt_text` |

---

## 后端接口要求

详见 [API_FORMAT.md](./API_FORMAT.md)

简要说明：
- **请求**：POST multipart/form-data，包含 `audio` 字段
- **响应**：JSON，必须包含 `audio` 字段，可选包含 `src_text` 和 `tgt_text`

---

## 演示准备

### 测试音频准备

建议准备以下测试文件：
- 短中文音频（3-5 秒）
- 短英文音频（3-5 秒）
- 较长音频（10-20 秒）

### 后端检查清单

- [ ] 后端服务正常运行
- [ ] CORS 配置正确（允许前端域名访问）
- [ ] `config.json` 中的接口地址正确
- [ ] 返回格式符合 `config.json` 中的字段配置

### 前端检查清单

- [ ] 使用 `python server.py` 启动（而非直接打开 HTML）
- [ ] 使用 localhost 访问（麦克风功能需要）
- [ ] 已授予麦克风权限
- [ ] 浏览器控制台无错误

---

## 常见问题

详见 [NOTES.md](./NOTES.md)

快速排查：
| 问题 | 可能原因 | 解决方法 |
|------|---------|---------|
| 页面显示「配置加载失败」 | 未使用 server.py 启动 | 使用 `python server.py` 启动 |
| 麦克风无法启动 | 非 localhost 访问 | 使用 localhost 访问 |
| 自动播放被阻止 | 浏览器策略 | 手动点击播放 |
| 返回格式错误 | 配置不匹配 | 检查 `config.json` 中的字段配置 |
| 请求超时 | 模型处理慢 | 增加 `timeout` 值 |

---

## 技术栈

- **前端**：纯 HTML + CSS + 原生 JavaScript
- **配置服务器**：Python 标准库 http.server
- **后端交互**：Fetch API + FormData
- **音频处理**：Web Audio API + MediaRecorder
- **无依赖**：不使用任何前端框架或库

---

## 配置服务器工作原理

```
浏览器请求 /config
        ↓
server.py 读取 config.json
        ↓
返回 JSON 配置给前端
        ↓
前端使用配置发起请求
```

**优点**：
1. 前端源码中无配置硬编码
2. 配置由服务器管理员维护
3. 修改配置无需改代码
4. 支持多环境（dev/test/prod）

---

## 架构图

```
┌─────────────┐
│   浏览器    │
│  (前端)     │
└──────┬──────┘
       │ GET /config
       ▼
┌─────────────┐
│  server.py  │ ←──→ config.json
│ (配置服务器) │
└─────────────┘
       │ POST /s2st (audio)
       ▼
┌─────────────┐
│  后端 API   │
│  (模型服务) │
└─────────────┘
```

# 配置分离说明

## 架构设计

```
┌─────────────────┐     /config      ┌──────────────────┐
│  前端 (index.html) │ ───────────────→ │  配置服务器      │
│                 │                   │  (server.py)     │
│  - 无配置硬编码   │                   │                  │
│  - 运行时加载配置 │                   │  读取 config.json│
└─────────────────┘                   └──────────────────┘
                                               │
                                               ▼
                                        ┌──────────────────┐
                                        │   config.json    │
                                        │  (服务器配置文件) │
                                        └──────────────────┘
```

---

## 文件结构

```
voxcpm/s2st_demo/
├── config.json    # 配置文件（服务器端，不暴露给前端源码）
├── server.py      # 配置服务器（提供 /config 接口和静态文件服务）
├── index.html     # 前端页面（运行时从服务器加载配置）
└── ...
```

---

## 配置文件说明 (config.json)

```json
{
  "api": {
    "url": "http://localhost:8000",     // 后端 API 地址
    "endpoint": "/s2st",                // 接口路径
    "timeout": 60000                    // 请求超时时间（毫秒）
  },
  "audio": {
    "maxFileSize": 104857600,           // 最大文件大小（100MB）
    "sampleRate": 16000,                // 采样率
    "channels": 1                       // 声道数
  },
  "response": {
    "audioField": "audio",              // 音频字段名
    "audioType": "url",                 // 音频类型：url/base64/binary
    "sourceTextField": "src_text",      // 原文字段名
    "targetTextField": "tgt_text"       // 译文字段名
  },
  "ui": {
    "showTextResults": true,            // 是否显示文本结果
    "autoPlayAudio": true               // 是否自动播放音频
  }
}
```

---

## 启动方式

### 使用配置服务器（推荐）

```bash
cd voxcpm/s2st_demo
python server.py
```

访问：http://localhost:8080

**输出示例**：
```
==================================================
端到端语音翻译 Demo - 配置服务器
==================================================

服务地址: http://localhost:8080
配置接口: http://localhost:8080/config
演示页面: http://localhost:8080/index.html

配置文件: /path/to/config.json

提示：修改 config.json 后刷新页面即可生效
==================================================

服务器已启动，按 Ctrl+C 停止
```

---

## 修改配置

1. **编辑 `config.json`**：
   ```bash
   # 使用任意文本编辑器
   notepad config.json      # Windows
   vim config.json          # Linux/Mac
   code config.json         # VS Code
   ```

2. **刷新浏览器页面**：
   - 配置会在页面加载时自动获取
   - 无需重启服务器

---

## 配置项详解

### api 配置组

| 字段 | 说明 | 示例 |
|------|------|------|
| `url` | 后端 API 服务器地址 | `http://localhost:8000` |
| `endpoint` | 翻译接口路径 | `/s2st` |
| `timeout` | 请求超时时间（毫秒） | `60000` (60秒) |

### audio 配置组

| 字段 | 说明 | 示例 |
|------|------|------|
| `maxFileSize` | 最大上传文件大小（字节） | `104857600` (100MB) |
| `sampleRate` | 音频采样率（Hz） | `16000` |
| `channels` | 音频声道数 | `1` (单声道) |

### response 配置组

| 字段 | 说明 | 可选值 |
|------|------|--------|
| `audioField` | 后端返回的音频字段名 | `audio`, `output_audio` 等 |
| `audioType` | 音频数据格式 | `url`, `base64`, `binary` |
| `sourceTextField` | 原文字段名 | `src_text`, `source` 等 |
| `targetTextField` | 译文字段名 | `tgt_text`, `target` 等 |

---

## 安全性说明

### 为什么不会暴露后端地址？

**从源码角度**：
- ✅ 前端 HTML/JS 中**没有硬编码**任何后端 URL
- ✅ 用户"查看源代码"只能看到 `/config` 这个固定路径
- ✅ 真实的后端地址存储在**服务器端的 config.json** 中

**从运行角度**：
- ⚠️ 浏览器仍需知道真实地址才能发送请求
- ⚠️ 开发者打开 F12 → Network 面板可以看到请求目标
- ⚠️ 这不是"加密"，而是"源码分离"

### 安全级别

| 用户类型 | 能否看到后端地址 | 原因 |
|---------|----------------|------|
| 普通用户（只浏览） | ❌ 看不到 | 源码中没有，不打开开发者工具看不到 |
| 技术用户（查看源码） | ❌ 看不到 | 源码中只有 `/config` |
| 开发者（F12 Network） | ✅ 可以看到 | 浏览器必须知道地址才能请求 |
| 网络管理员（抓包） | ✅ 可以看到 | 可以捕获所有网络流量 |

### 结论

这种方式适用于：
- ✅ 演示环境（避免普通用户看到内部地址）
- ✅ 客户部署（配置由服务器管理员维护）
- ✅ 多环境管理（开发/测试/生产环境使用不同配置）

**不适用于**：
- ❌ 真正的安全隔离（需要后端代理/网关）
- ❌ 保护敏感 API（需要 Token/签名验证）

---

## 高级用法

### 环境变量支持（可选扩展）

如需支持环境变量，可以修改 `server.py`：

```python
import os

config_path = os.getenv('DEMO_CONFIG_PATH', 'config.json')
```

然后启动时指定：
```bash
DEMO_CONFIG_PATH=/path/to/prod-config.json python server.py
```

### 多个演示环境

```
voxcpm/s2st_demo/
├── config-dev.json      # 开发环境
├── config-test.json     # 测试环境
├── config-prod.json     # 生产环境
└── server.py            # 根据环境变量读取配置
```

---

## 故障排查

### 配置加载失败

**问题**：页面显示「配置加载失败，使用默认配置」

**排查**：
1. 确认使用 `python server.py` 启动（而非直接打开 HTML）
2. 访问 http://localhost:8080/config 检查配置是否返回
3. 检查 `config.json` 格式是否正确（JSON 语法）

### 后端连接失败

**问题**：录音/上传后显示「处理失败」

**排查**：
1. 检查 `config.json` 中的 `api.url` 是否正确
2. 确认后端服务是否运行
3. 检查 CORS 配置（后端需要允许跨域）

---

## 开发建议

### 本地开发

```bash
# 终端 1：启动配置服务器
cd voxcpm/s2st_demo
python server.py

# 终端 2：启动后端 API
cd your-backend
python api_server.py
```

### 修改后端接口

只需修改 `config.json`：
```json
{
  "api": {
    "url": "http://192.168.1.100:9000",  // 改为新的后端地址
    "endpoint": "/api/translate"          // 改为新的接口路径
  }
}
```

刷新页面即可生效，无需修改前端代码。

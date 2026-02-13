# VoxCPM S2ST 服务启动指南

本文档描述如何启动从后端到前端的完整服务链路。

---

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                      用户浏览器                          │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP (port 8080)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  前端配置服务器 (s2st_demo/server.py)                    │
│  - 提供 /config 接口                                     │
│  - 提供静态页面 (index.html)                            │
│  端口: 8080                                              │
└───────────────────┬─────────────────────────────────────┘
                    │ HTTP POST /s2st
                    │ (port 19366)
                    ▼
┌─────────────────────────────────────────────────────────┐
│  API 服务器 (api_server/api/routes.py)                  │
│  - FastAPI 应用                                          │
│  - 提供 /s2st, /streaming 等接口                        │
│  - 提供 /static 静态文件服务                             │
│  端口: 19366                                             │
└───────────────────┬─────────────────────────────────────┘
                    │ WebSocket
                    ▼
┌─────────────────────────────────────────────────────────┐
│  ASR WebSocket 服务 (外部服务)                          │
│  - 语音识别和翻译                                        │
│  地址: 通过 WS_URL 环境变量配置                          │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  VoxCPM TTS 模型 (本地加载)                             │
│  - 语音合成                                              │
│  路径: ./models/openbmb__VoxCPM-0.5B/                   │
└─────────────────────────────────────────────────────────┘
```

---

## 前置条件

### 1. 环境要求

- Python 3.8+
- pip 包管理器
- 足够的磁盘空间（模型约 1-2GB）

### 2. 模型准备

确保 VoxCPM 模型已下载到本地：

```bash
# 默认路径
./models/openbmb__VoxCPM-0.5B/
```

如果模型未下载，可以从 HuggingFace 获取：

```bash
# 使用 huggingface-cli
huggingface-cli download openbmb/VoxCPM-0.5B --local-dir ./models/openbmb__VoxCPM-0.5B
```

### 3. ASR 服务配置

你需要有一个运行的 ASR WebSocket 服务。获取以下信息：

- WebSocket URL (`WS_URL`)
- 用户 ID (`USER_ID`)
- 认证 Token (`TOKEN`)

---

## 方式一：使用命令行参数启动（推荐用于开发）

### 步骤 1: 配置环境变量

创建 `.env` 文件或直接在命令行设置环境变量：

```bash
# Linux/macOS
export WS_URL="ws://your-asr-server:port/ws"
export USER_ID="your_user_id"
export TOKEN="your_token"

# Windows (PowerShell)
$env:WS_URL="ws://your-asr-server:port/ws"
$env:USER_ID="your_user_id"
$env:TOKEN="your_token"

# Windows (CMD)
set WS_URL=ws://your-asr-server:port/ws
set USER_ID=your_user_id
set TOKEN=your_token
```

### 步骤 2: 启动 API 服务器

```bash
# 进入项目根目录
cd d:\seki\work\VoxCPM

# 启动 API 服务器（方式一：使用 uvicorn 直接运行）
python -m uvicorn api_server.api.routes:app --host 0.0.0.0 --port 19366

# 或（方式二：使用 Python 模块运行）
python -c "from api_server.api.routes import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=19366)"
```

API 服务器将在 `http://localhost:19366` 启动。

### 步骤 3: 启动前端配置服务器

**新开一个终端窗口**，执行：

```bash
# 进入 s2st_demo 目录
cd d:\seki\work\VoxCPM\s2st_demo

# 启动前端服务器
python server.py
```

前端服务器将在 `http://localhost:8080` 启动。

### 步骤 4: 更新前端配置

编辑 `s2st_demo/config.json`：

```json
{
  "api": {
    "url": "http://localhost:19366",
    "endpoint": "/s2st",
    "timeout": 30000
  }
}
```

### 步骤 5: 访问应用

在浏览器中打开：

```
http://localhost:8080
```

---

## 方式二：使用 .env 文件启动（推荐用于生产）

### 步骤 1: 创建 .env 文件

在项目根目录创建 `.env` 文件：

```bash
# d:\seki\work\VoxCPM\.env
```

内容如下：

```env
# ASR WebSocket 服务配置
WS_URL=ws://your-asr-server:port/ws
USER_ID=your_user_id
TOKEN=your_token

# 语言配置
FROM_LANG=zh
TO_LANG=en

# TTS 配置
TTS_TEXT_SOURCE=trans
CFG_VALUE=2.0
INFERENCE_TIMESTEPS=10

# 模型配置
VOXCPM_MODEL_PATH=./models/openbmb__VoxCPM-0.5B/
VOXCPM_HF_ID=openbmb/VoxCPM-0.5B
```

### 步骤 2: 加载 .env 并启动 API 服务器

创建启动脚本 `start_api.py`：

```python
#!/usr/bin/env python3
"""API 服务器启动脚本"""
import uvicorn
from dotenv import load_dotenv
from api_server.api.routes import app

# 加载 .env 文件
load_dotenv()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=19366,
        log_level="info"
    )
```

启动：

```bash
python start_api.py
```

### 步骤 3: 启动前端服务器

同方式一的步骤 3。

---

## 方式三：使用 systemd 服务（Linux 生产环境）

### 创建 API 服务

创建 `/etc/systemd/system/voxcpm-api.service`：

```ini
[Unit]
Description=VoxCPM S2ST API Service
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/VoxCPM
Environment="WS_URL=ws://your-asr-server:port/ws"
Environment="USER_ID=your_user_id"
Environment="TOKEN=your_token"
Environment="FROM_LANG=zh"
Environment="TO_LANG=en"
ExecStart=/usr/bin/python3 -m uvicorn api_server.api.routes:app --host 0.0.0.0 --port 19366
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable voxcpm-api
sudo systemctl start voxcpm-api
sudo systemctl status voxcpm-api
```

### 创建前端服务

创建 `/etc/systemd/system/voxcpm-frontend.service`：

```ini
[Unit]
Description=VoxCPM S2ST Frontend Service
After=network.target voxcpm-api.service

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/VoxCPM/s2st_demo
ExecStart=/usr/bin/python3 server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable voxcpm-frontend
sudo systemctl start voxcpm-frontend
sudo systemctl status voxcpm-frontend
```

---

## 方式四：使用 Docker（容器化部署）

### 创建 Dockerfile

`api_server/Dockerfile`：

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++\
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 19366

# 启动命令
CMD ["python", "-m", "uvicorn", "api_server.api.routes:app", "--host", "0.0.0.0", "--port", "19366"]
```

### 创建 docker-compose.yml

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "19366:19366"
    environment:
      - WS_URL=${WS_URL}
      - USER_ID=${USER_ID}
      - TOKEN=${TOKEN}
      - FROM_LANG=zh
      - TO_LANG=en
    volumes:
      - ./models:/app/models
      - ./s2st_demo/output:/app/s2st_demo/output
    restart: always

  frontend:
    image: python:3.10-slim
    working_dir: /app/s2st_demo
    ports:
      - "8080:8080"
    volumes:
      - .:/app
    command: ["python", "server.py"]
    restart: always
    depends_on:
      - api
```

### 启动服务

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

---

## 方式五：Windows 服务（使用 NSSM）

### 安装 API 服务

```powershell
# 下载 NSSM
# https://nssm.cc/download

# 安装 API 服务
nssm install VoxCPM_API C:\Python310\python.exe -m uvicorn api_server.api.routes:app --host 0.0.0.0 --port 19366

# 设置环境变量
nssm set VoxCPM_API AppEnvironmentExtra "WS_URL=ws://your-asr-server:port/ws" "USER_ID=your_user_id" "TOKEN=your_token"

# 设置工作目录
nssm set VoxCPM_API AppDirectory D:\seki\work\VoxCPM

# 启动服务
nssm start VoxCPM_API
```

### 安装前端服务

```powershell
# 安装前端服务
nssm install VoxCPM_Frontend C:\Python310\python.exe server.py

# 设置工作目录
nssm set VoxCPM_Frontend AppDirectory D:\seki\work\VoxCPM\s2st_demo

# 启动服务
nssm start VoxCPM_Frontend
```

---

## 验证服务状态

### 1. 检查 API 服务器

```bash
# 健康检查
curl http://localhost:19366/health

# 预期响应
# {"status":"ok","tts_model_loaded":true}

# 检查配置
curl http://localhost:19366/config

# 预期响应
# {"ws_url":"ws://...","user_id":"...","token":"...",...}
```

### 2. 检查前端服务器

```bash
# 访问配置
curl http://localhost:8080/config

# 预期响应（config.json 内容）
# {"api":{"url":"http://localhost:19366","endpoint":"/s2st",...},...}
```

### 3. 端到端测试

```bash
# 使用测试音频
curl -X POST http://localhost:19366/s2st \
  -F "audio=@test.wav" \
  -F "from_lang=zh" \
  -F "to_lang=en"

# 预期响应
# {"audio":"/static/xxx.wav","src_text":"...","tgt_text":"..."}
```

---

## 常见问题排查

### 问题 1: API 服务器启动失败

**症状**：启动时提示模块未找到

**解决方法**：
```bash
# 检查 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 或使用绝对路径
cd /path/to/VoxCPM
python -m uvicorn api_server.api.routes:app --port 19366
```

### 问题 2: 模型加载失败

**症状**：`模型初始化失败`

**解决方法**：
1. 检查模型路径是否正确
2. 确保模型文件完整
3. 检查磁盘空间

```bash
# 检查模型目录
ls -la ./models/openbmb__VoxCPM-0.5B/

# 设置正确的模型路径
export VOXCPM_MODEL_PATH=/path/to/models/openbmb__VoxCPM-0.5B
```

### 问题 3: ASR 连接失败

**症状**：`WebSocket 连接失败`

**解决方法**：
1. 检查 WS_URL 是否正确
2. 确认 ASR 服务是否运行
3. 检查网络连接和防火墙

```bash
# 测试 WebSocket 连接
wscat -c ws://your-asr-server:port/ws
```

### 问题 4: CORS 错误

**症状**：前端请求被阻止

**解决方法**：
确保 API 服务器的 CORS 配置正确：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 问题 5: 音频文件无法访问

**症状**：`/static/xxx.wav` 返回 404

**解决方法**：
1. 检查输出目录是否存在
2. 确认文件权限
3. 检查 StaticFiles 挂载路径

```bash
# 检查输出目录
ls -la s2st_demo/output/

# 手动创建目录
mkdir -p s2st_demo/output
```

---

## 端口占用检查

### Linux/macOS

```bash
# 检查端口占用
lsof -i :19366
lsof -i :8080

# 杀死占用进程
kill -9 <PID>
```

### Windows

```powershell
# 检查端口占用
netstat -ano | findstr :19366
netstat -ano | findstr :8080

# 杀死占用进程
taskkill /PID <PID> /F
```

---

## 日志查看

### API 服务器日志

API 服务器使用标准输出，日志会显示在终端。

查看特定模块的日志：

```bash
# 设置日志级别
export LOG_LEVEL=DEBUG

# 启动时显示详细日志
python -m uvicorn api_server.api.routes:app --log-level debug
```

### 前端服务器日志

前端服务器日志会显示在运行 `python server.py` 的终端。

---

## 性能优化建议

### 1. 启用模型缓存

TTS 模型使用单例模式，首次加载后会缓存。

### 2. 调整工作进程数

```bash
# 使用多个 worker 进程
python -m uvicorn api_server.api.routes:app --workers 4 --port 19366
```

### 3. 使用 Gunicorn（生产环境）

```bash
pip install gunicorn

gunicorn api_server.api.routes:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:19366
```

### 4. 定期清理输出文件

创建清理脚本：

```bash
#!/bin/bash
# cleanup.sh
find s2st_demo/output/ -name "*.wav" -mtime +7 -delete
```

---

## 安全建议

1. **使用 HTTPS**：生产环境建议使用反向代理（Nginx）配置 HTTPS
2. **限制 CORS**：将 `allow_origins` 设置为具体域名而非 `"*"`
3. **环境变量保护**：不要将 `.env` 文件提交到版本控制
4. **Token 安全**：定期更换认证 Token
5. **访问控制**：配置防火墙规则限制访问

---

## 停止服务

### 手动启动的服务

```bash
# Ctrl+C 停止服务
# 或
kill <pid>
```

### systemd 服务

```bash
sudo systemctl stop voxcpm-api
sudo systemctl stop voxcpm-frontend
```

### Docker 服务

```bash
docker-compose down
```

---

## 快速启动脚本

创建 `start_all.sh`（Linux/macOS）：

```bash
#!/bin/bash

# 设置环境变量
export WS_URL="ws://your-asr-server:port/ws"
export USER_ID="your_user_id"
export TOKEN="your_token"

# 启动 API 服务器
echo "启动 API 服务器..."
python -m uvicorn api_server.api.routes:app --host 0.0.0.0 --port 19366 &
API_PID=$!

# 等待 API 启动
sleep 5

# 启动前端服务器
echo "启动前端服务器..."
cd s2st_demo
python server.py &
FRONTEND_PID=$!

echo "服务已启动"
echo "API 服务器: http://localhost:19366"
echo "前端页面: http://localhost:8080"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待信号
trap "kill $API_PID $FRONTEND_PID" EXIT
wait
```

创建 `start_all.bat`（Windows）：

```batch
@echo off
set WS_URL=ws://your-asr-server:port/ws
set USER_ID=your_user_id
set TOKEN=your_token

echo 启动 API 服务器...
start "VoxCPM API" python -m uvicorn api_server.api.routes:app --host 0.0.0.0 --port 19366

timeout /t 5 /nobreak

echo 启动前端服务器...
cd s2st_demo
start "VoxCPM Frontend" python server.py

echo 服务已启动
echo API 服务器: http://localhost:19366
echo 前端页面: http://localhost:8080
pause
```

---

## 总结

| 启动方式 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| 命令行参数 | 开发测试 | 简单直接 | 需要手动设置环境变量 |
| .env 文件 | 开发/小型部署 | 配置集中 | 需要额外依赖 python-dotenv |
| systemd | Linux 生产 | 自动重启、日志管理 | 仅限 Linux |
| Docker | 跨平台部署 | 环境一致、易于扩展 | 需要熟悉 Docker |
| NSSM | Windows 生产 | 自动重启 | 仅限 Windows |

选择适合你环境的启动方式即可。

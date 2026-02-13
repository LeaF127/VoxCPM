# VoxCPM S2ST 快速启动指南

## 配置方式

### 1. 创建配置文件

在项目根目录创建 `.env` 文件：

```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# Windows (CMD)
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

### 2. 编辑 `.env` 文件

修改以下必填配置项：

```env
# ASR WebSocket 服务配置（必填）
WS_URL=ws://your-asr-server:port/ws
USER_ID=your_user_id
TOKEN=your_token
```

其他配置项可以使用默认值，或根据需要调整。

---

## 启动方式

### Windows

#### 方式一：使用命令行（推荐开发调试）

**终端 1 - 启动 API 服务器：**
```powershell
# 进入项目目录
cd d:\seki\work\VoxCPM

# 启动 API 服务器
python -m uvicorn api_server.api.routes:app
# API 服务器将在 http://localhost:19366 启动
```

**终端 2 - 启动前端服务器：**
```powershell
# 进入前端目录
cd d:\seki\work\VoxCPM\s2st_demo

# 启动前端服务器
python server.py
# 前端将在 http://localhost:8080 启动
```

#### 方式二：使用启动脚本

创建 `start.bat` 文件：

```batch
@echo off
echo ========================================
echo VoxCPM S2ST 服务启动脚本
echo ========================================

echo.
echo [1/2] 启动 API 服务器...
start "VoxCPM API Server" python -m uvicorn api_server.api.routes:app

timeout /t 3 /nobreak

echo.
echo [2/2] 启动前端服务器...
cd s2st_demo
start "VoxCPM Frontend" python server.py
cd ..

echo.
echo ========================================
echo 服务启动完成！
echo ========================================
echo.
echo API 服务器:  http://localhost:19366
echo 前端页面:    http://localhost:8080
echo.
echo 按任意键关闭此窗口（服务将继续运行）
pause
```

双击 `start.bat` 即可启动所有服务。

---

### Linux/macOS

#### 方式一：使用命令行

**终端 1 - 启动 API 服务器：**
```bash
cd /path/to/VoxCPM
python -m uvicorn api_server.api.routes:app
```

**终端 2 - 启动前端服务器：**
```bash
cd /path/to/VoxCPM/s2st_demo
python server.py
```

#### 方式二：使用启动脚本

创建 `start.sh` 文件：

```bash
#!/bin/bash

echo "========================================"
echo "VoxCPM S2ST 服务启动脚本"
echo "========================================"

echo ""
echo "[1/2] 启动 API 服务器..."
python -m uvicorn api_server.api.routes:app &
API_PID=$!

sleep 3

echo ""
echo "[2/2] 启动前端服务器..."
cd s2st_demo
python server.py &
FRONTEND_PID=$!
cd ..

echo ""
echo "========================================"
echo "服务启动完成！"
echo "========================================"
echo ""
echo "API 服务器:  http://localhost:19366"
echo "前端页面:    http://localhost:8080"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 捕获退出信号
trap "kill $API_PID $FRONTEND_PID" EXIT INT TERM

wait
```

赋予执行权限并运行：
```bash
chmod +x start.sh
./start.sh
```

---

## 访问应用

启动完成后，在浏览器中打开：

```
http://localhost:8080
```

---

## 验证服务

### 检查 API 服务器

```bash
# 访问健康检查接口
curl http://localhost:19366/health

# 预期响应
# {"status":"ok","config_loaded":true,"output_dir":"..."}
```

### 检查前端服务器

```bash
# 访问配置接口
curl http://localhost:8080/config

# 预期返回 config.json 的内容
```

---

## 配置文件说明

`.env` 文件包含所有配置项，主要分为以下几类：

### 必填配置

| 配置项 | 说明 | 示例 |
|--------|------|------|
| `WS_URL` | ASR WebSocket 服务地址 | `ws://localhost:8000/ws` |
| `USER_ID` | 用户 ID | `test_user` |
| `TOKEN` | 认证 Token | `your_token_here` |

### 可选配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `FROM_LANG` | 源语言 | `zh` |
| `TO_LANG` | 目标语言 | `en` |
| `CFG_VALUE` | CFG 值 | `2.0` |
| `INFERENCE_TIMESTEPS` | 推理步数 | `10` |
| `API_PORT` | API 端口 | `19366` |
| `FRONTEND_PORT` | 前端端口 | `8080` |

完整配置说明请参考 `.env.example` 文件。

---

## 常见问题

### 1. 配置文件未找到

**错误信息**：`配置文件不存在`

**解决方法**：
```bash
# 确认 .env 文件存在
ls .env  # Linux/macOS
dir .env # Windows

# 如果不存在，从示例创建
cp .env.example .env  # Linux/macOS
copy .env.example .env # Windows
```

### 2. ASR 服务连接失败

**错误信息**：`服务器配置错误：WS_URL 未配置`

**解决方法**：
1. 打开 `.env` 文件
2. 设置 `WS_URL`、`USER_ID`、`TOKEN`
3. 保存并重启服务

### 3. 端口被占用

**错误信息**：`Address already in use`

**解决方法**：

**Windows：**
```powershell
# 查找占用进程
netstat -ano | findstr :19366

# 终止进程
taskkill /PID <进程ID> /F
```

**Linux/macOS：**
```bash
# 查找占用进程
lsof -i :19366

# 终止进程
kill -9 <进程ID>
```

或修改 `.env` 中的端口配置：
```env
API_PORT=19367
FRONTEND_PORT=8081
```

### 4. 模型加载失败

**错误信息**：`模型初始化失败`

**解决方法**：
1. 检查 `.env` 中的模型路径配置
2. 确保模型文件存在
```env
VOXCPM_MODEL_PATH=./models/openbmb__VoxCPM-0.5B/
```

---

## 停止服务

### 手动启动的服务

在各个终端按 `Ctrl+C` 停止服务。

### 脚本启动的服务

**Windows：** 直接关闭命令行窗口

**Linux/macOS：** 按 `Ctrl+C`，脚本会自动清理所有子进程

---

## 下一步

- 详细配置参考：[`.env.example`](.env.example)
- API 文档：[api_server/API_ALIGNMENT.md](api_server/API_ALIGNMENT.md)
- 完整部署指南：[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

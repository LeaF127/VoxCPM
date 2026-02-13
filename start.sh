#!/bin/bash

# VoxCPM S2ST 服务启动脚本 (Linux/macOS)

set -e

echo "========================================"
echo "VoxCPM S2ST 服务启动脚本"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

echo "[检查] Python 版本: $(python3 --version)"
echo ""

# 检查配置文件
if [ ! -f ".env" ]; then
    echo "[警告] 未找到 .env 配置文件"
    echo ""
    if [ -f ".env.example" ]; then
        echo "正在从 .env.example 创建配置文件..."
        cp .env.example .env
        echo "配置文件已创建: .env"
        echo ""
        echo "[重要] 请先编辑 .env 文件，配置以下必填项："
        echo "  - WS_URL: ASR WebSocket 服务地址"
        echo "  - USER_ID: 用户 ID"
        echo "  - TOKEN: 认证 Token"
        echo ""
        echo "编辑完成后，请重新运行此脚本。"
        exit 0
    else
        echo "[错误] 未找到 .env.example 文件"
        exit 1
    fi
fi

echo "[检查] 配置文件: .env"
echo ""

# 检查是否已配置
if grep -q "ws://your-asr-server" .env 2>/dev/null; then
    echo "[警告] .env 文件中的配置似乎是默认值"
    echo "请确保已正确配置 WS_URL、USER_ID、TOKEN"
    echo ""
fi

# 获取配置
source .env

# 启动函数
start_services() {
    echo "[1/2] 启动 API 服务器..."
    python3 -m uvicorn api_server.api.routes:app &
    API_PID=$!
    echo "API 服务器 PID: $API_PID"

    sleep 3

    echo "[2/2] 启动前端服务器..."
    cd s2st_demo
    python3 server.py &
    FRONTEND_PID=$!
    echo "前端服务器 PID: $FRONTEND_PID"
    cd ..

    sleep 2

    echo ""
    echo "========================================"
    echo "服务启动完成！"
    echo "========================================"
    echo ""
    echo "API 服务器:  http://localhost:19366"
    echo "前端页面:    http://localhost:8080"
    echo "API 文档:    http://localhost:19366/docs"
    echo ""
    echo "提示："
    echo "  - 配置文件: .env"
    echo "  - 停止服务: 按 Ctrl+C"
    echo ""
}

# 退出清理
cleanup() {
    echo ""
    echo "正在停止服务..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
        echo "API 服务器已停止"
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo "前端服务器已停止"
    fi
    echo "所有服务已停止"
    exit 0
}

# 捕获信号
trap cleanup SIGINT SIGTERM

# 启动服务
start_services

# 等待
wait

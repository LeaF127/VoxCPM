@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: VoxCPM S2ST 服务启动脚本 (Windows)

echo ========================================
echo VoxCPM S2ST 服务启动脚本
echo ========================================
echo.

:: 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

:: 检查配置文件
if not exist ".env" (
    echo [警告] 未找到 .env 配置文件
    echo.
    if exist ".env.example" (
        echo 正在从 .env.example 创建配置文件...
        copy .env.example .env >nul
        echo 配置文件已创建: .env
        echo.
        echo [重要] 请先编辑 .env 文件，配置以下必填项：
        echo   - WS_URL: ASR WebSocket 服务地址
        echo   - USER_ID: 用户 ID
        echo   - TOKEN: 认证 Token
        echo.
        echo 编辑完成后，请重新运行此脚本。
        pause
        exit /b 0
    ) else (
        echo [错误] 未找到 .env.example 文件
        pause
        exit /b 1
    )
)

echo [检查] 配置文件: .env
echo.

:: 检查是否已配置
findstr /C:"ws://your-asr-server" .env >nul
if not errorlevel 1 (
    echo [警告] .env 文件中的配置似乎是默认值
    echo 请确保已正确配置 WS_URL、USER_ID、TOKEN
    echo.
)

echo [1/2] 启动 API 服务器...
start "VoxCPM API Server" cmd /c "python -m uvicorn api_server.api.routes:app && pause"
timeout /t 3 /nobreak >nul

echo [2/2] 启动前端服务器...
cd s2st_demo
start "VoxCPM Frontend Server" cmd /c "python server.py && pause"
cd ..

timeout /t 2 /nobreak >nul

echo.
echo ========================================
echo 服务启动完成！
echo ========================================
echo.
echo API 服务器:  http://localhost:19366
echo 前端页面:    http://localhost:8080
echo API 文档:    http://localhost:19366/docs
echo.
echo 提示：
echo   - 配置文件: .env
echo   - 日志信息: 请查看各自的服务窗口
echo   - 停止服务: 关闭对应的命令行窗口
echo.
echo 按任意键关闭此窗口（服务将继续运行）...
pause >nul

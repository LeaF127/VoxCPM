# ASR 服务连接测试

## 说明

此测试脚本用于诊断 ASR WebSocket 服务的连接问题。

## 运行方式

### 方式 1：使用批处理脚本（推荐）

**Windows:**
```bash
test_asr_connection.bat
```

**Linux/Mac:**
```bash
chmod +x test_asr_connection.sh
./test_asr_connection.sh
```

### 方式 2：直接运行 Python 模块

```bash
python -m api_server.tests.test_asr_connection
```

## 测试内容

### 1. 基本连接测试
- 测试 WebSocket 握手是否成功
- 验证 URL 格式是否正确
- 检查认证信息

### 2. 完整功能测试（可选）
- 如果 `tests/test_audio.wav` 存在，会进行完整的识别测试
- 测试流式识别和翻译功能

## 配置要求

确保 `.env` 文件中配置了正确的参数：

```env
# ASR WebSocket 服务地址（必须以 ws:// 或 wss:// 开头）
WS_URL=ws://your-asr-server:port/ws

# 用户 ID
USER_ID=your_user_id

# 认证 Token
TOKEN=your_token

# 语言配置
FROM_LANG=zh
TO_LANG=en
```

## 常见错误诊断

### 错误：`InvalidMessage: did not receive a valid HTTP response`

**可能原因：**
1. URL 格式错误（使用了 `http://` 而不是 `ws://`）
2. ASR 服务未运行
3. 端口号错误
4. 防火墙阻止连接

**解决方法：**
1. 检查 `.env` 文件中的 `WS_URL` 是否以 `ws://` 或 `wss://` 开头
2. 确认 ASR 服务正在运行
3. 尝试用浏览器访问 ASR 服务的 HTTP 地址，确认服务可访问

### 错误：`ConnectionRefusedError`

**可能原因：**
1. ASR 服务未启动
2. 端口号错误
3. IP 地址错误

**解决方法：**
1. 检查 ASR 服务是否运行
2. 确认端口号和 IP 地址正确

### 错误：`InvalidHandshake`

**可能原因：**
1. URL 指向的不是 WebSocket 服务
2. 认证信息错误

**解决方法：**
1. 确认 URL 指向的是 WebSocket 端点
2. 检查 `USER_ID` 和 `TOKEN` 是否正确

## 测试音频文件

如需测试完整的识别功能，可以将测试音频文件放在：
```
tests/test_audio.wav
```

**推荐音频格式：**
- 格式：WAV
- 采样率：16000 Hz
- 声道：单声道
- 位深度：16 bit

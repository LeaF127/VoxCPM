# S2ST Demo 文档

本目录包含 S2ST (Speech-to-Speech Translation) Demo 的相关文档。

## 文档列表

- [API.md](./API.md) - FastAPI 接口文档，包含完整的 API 使用说明

## 快速开始

### 1. 启动 API 服务

```bash
# 方式 1：直接运行
python s2st_demo/api_server.py

# 方式 2：使用 uvicorn（推荐）
uvicorn s2st_demo.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 2. 访问 API 文档

启动服务后，访问以下地址查看自动生成的 API 文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. 测试接口

参考 [API.md](./API.md) 中的请求示例进行测试。

## 依赖安装

确保已安装以下依赖：

```bash
pip install fastapi uvicorn python-multipart
```

如果使用项目环境，这些依赖可能已包含在 `pyproject.toml` 中。

## 注意事项

1. **模型路径**：确保已设置 `VOXCPM_MODEL_PATH` 环境变量或使用 Hugging Face 模型
2. **ASR 服务**：确保 ASR WebSocket 服务可访问
3. **端口占用**：默认端口为 8000，如被占用请修改


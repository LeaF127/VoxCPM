#!/usr/bin/env python3
"""
极简配置服务器 - 为前端提供配置和静态文件服务

用法：
    python server.py

默认端口：8080
配置文件：config.json
"""

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ConfigHTTPRequestHandler(SimpleHTTPRequestHandler):
    """自定义请求处理器：支持 /config 接口和静态文件服务"""

    def __init__(self, *args, **kwargs):
        # 配置文件路径（与 server.py 同目录）
        self.config_file = Path(__file__).parent / "config.json"
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """处理 GET 请求"""
        if self.path == "/config" or self.path == "/config/":
            self.serve_config()
        else:
            # 静态文件服务
            super().do_GET()

    def serve_config(self):
        """返回配置文件内容（JSON 格式）"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 发送响应
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # 返回配置（美化输出）
            self.wfile.write(json.dumps(config, ensure_ascii=False, indent=2).encode('utf-8'))

        except FileNotFoundError:
            self.send_error(500, f"配置文件未找到: {self.config_file}")
        except json.JSONDecodeError as e:
            self.send_error(500, f"配置文件格式错误: {e}")

    def end_headers(self):
        """添加 CORS 头"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def log_message(self, format, *args):
        """自定义日志输出"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    """启动服务器"""
    PORT = 8080

    print("=" * 50)
    print("端到端语音翻译 Demo - 配置服务器")
    print("=" * 50)
    print(f"\n服务地址: http://localhost:{PORT}")
    print(f"配置接口: http://localhost:{PORT}/config")
    print(f"演示页面: http://localhost:{PORT}/index.html")
    print(f"\n配置文件: {os.path.abspath('config.json')}")
    print("\n提示：修改 config.json 后刷新页面即可生效\n")
    print("=" * 50)

    server = HTTPServer(('localhost', PORT), ConfigHTTPRequestHandler)
    print(f"\n服务器已启动，按 Ctrl+C 停止\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n服务器已停止")
        server.shutdown()


if __name__ == '__main__':
    main()

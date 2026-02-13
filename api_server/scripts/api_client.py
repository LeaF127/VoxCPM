"""
API 测试客户端。

用于测试流式 S2ST API 的命令行工具。
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests


def main():
    parser = argparse.ArgumentParser(description="测试流式 S2ST API")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000/streaming"), help="API 服务地址（可通过环境变量 API_URL 设置）")
    parser.add_argument("--audio", default="s2st_demo/input/1133_tenlin.wav", help="音频文件路径")
    parser.add_argument("--ws-url", default=os.getenv("WS_URL"), help="ASR WebSocket 服务地址（可通过环境变量 WS_URL 设置）")
    parser.add_argument("--user-id", default=os.getenv("USER_ID"), help="用户 ID（可通过环境变量 USER_ID 设置）")
    parser.add_argument("--token", default=os.getenv("TOKEN"), help="认证 Token（可通过环境变量 TOKEN 设置）")
    parser.add_argument("--from-lang", default="zh", help="源语言代码")
    parser.add_argument("--to-lang", default="en", help="目标语言代码")
    parser.add_argument("--tts-text-source", default="trans", help="TTS 文本来源：trans=翻译结果，asr=识别结果")
    parser.add_argument("--normalize", default="true", help="是否启用文本正则化")

    args = parser.parse_args()

    # 验证必填参数
    if not args.ws_url:
        print("错误: --ws-url 参数或环境变量 WS_URL 必须提供")
        return
    if not args.user_id:
        print("错误: --user-id 参数或环境变量 USER_ID 必须提供")
        return
    if not args.token:
        print("错误: --token 参数或环境变量 TOKEN 必须提供")
        return

    url = args.url
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"错误: 音频文件不存在: {audio_path}")
        return

    files = {"audio_file": open(audio_path, "rb")}
    data = {
        "ws_url": args.ws_url,
        "user_id": args.user_id,
        "token": args.token,
        "from_lang": args.from_lang,
        "to_lang": args.to_lang,
        "tts_text_source": args.tts_text_source,
        "normalize": args.normalize
    }

    response = requests.post(url, files=files, data=data, stream=True)
    start_time = time.time()
    for line in response.iter_lines():
        if line:
            # 解析 SSE 事件
            if line.startswith(b"data: "):
                import json
                event_data = json.loads(line[6:])  # 去掉 "data: " 前缀
                event_type = event_data["event_type"]
                data = event_data["data"]

                if event_type == "segment":
                    print(f"句 {data['segment_index']}: {data['asr_text']} -> {data['trans_text']}")
                    print(f"耗时: {time.time() - start_time:.2f} 秒")
                    start_time = time.time()
                elif event_type == "complete":
                    print(f"完成！输出文件: {data['output_file']}")
                elif event_type == "error":
                    print(f"错误: {data['message']}")


if __name__ == "__main__":
    main()

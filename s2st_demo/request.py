import requests
from time import time

url = "http://localhost:8000/streaming"
files = {"audio_file": open("s2st_demo/input/1133_tenlin.wav", "rb")}
data = {
    "from_lang": "zh",
    "to_lang": "en",
    "tts_text_source": "trans",
    "normalize": "true"
}

response = requests.post(url, files=files, data=data, stream=True)
start_time = time()
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
                print(f"耗时: {time() - start_time:.2f} 秒")
                start_time = time()
            elif event_type == "complete":
                print(f"完成！输出文件: {data['output_file']}")
            elif event_type == "error":
                print(f"错误: {data['message']}")
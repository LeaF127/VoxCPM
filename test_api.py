"""
VoxCPM API 测试脚本

测试通过远程 IP 访问 TTS 服务
"""

import argparse
import os
import sys
import time
import uuid

import requests
import soundfile as sf


class TTSAPITester:
    """TTS API 测试类"""

    def __init__(self, host: str, port: int, timeout: int = 300):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.session = requests.Session()

    def health_check(self) -> dict:
        """健康检查"""
        print("\n[1] 健康检查...")
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"   状态: {data['status']}")
        print(f"   TTS 模型已加载: {data['tts_model_loaded']}")
        print(f"   ASR 模型已加载: {data['asr_model_loaded']}")
        return data

    def test_get_sync(
        self,
        text: str,
        prompt_audio: str = "tests/sample_prompt.wav",
        output_path: str = None,
    ) -> bool:
        """测试 GET 同步接口

        Args:
            text: 要合成的文本
            prompt_audio: 参考音频路径
            output_path: 输出音频路径（可选）

        Returns:
            是否成功
        """
        print(f"\n[2] 测试 GET 同步接口...")
        print(f"   文本: {text}")
        print(f"   参考音频: {prompt_audio}")

        url = f"{self.base_url}/api/v1/tts/tasks"
        params = {
            "text": text,
            "prompt_audio": prompt_audio,
        }

        start_time = time.time()
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            elapsed = time.time() - start_time
            print(f"   成功! 耗时: {elapsed:.2f}s")

            # 保存音频文件
            if output_path is None:
                output_path = f"test_output_{uuid.uuid4().hex[:8]}.wav"

            with open(output_path, "wb") as f:
                f.write(response.content)

            # 读取并显示音频信息
            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate
            print(f"   保存到: {output_path}")
            print(f"   采样率: {sample_rate} Hz")
            print(f"   时长: {duration:.2f}s")

            return True

        except requests.exceptions.Timeout:
            print(f"   超时 ({self.timeout}s)")
            return False
        except requests.exceptions.RequestException as e:
            print(f"   失败: {e}")
            return False

    def test_post_sync(
        self,
        text: str,
        prompt_audio: str = "tests/sample_prompt.wav",
        output_path: str = None,
    ) -> bool:
        """测试 POST 同步接口（return_audio=True）

        Args:
            text: 要合成的文本
            prompt_audio: 参考音频路径
            output_path: 输出音频路径（可选）

        Returns:
            是否成功
        """
        print(f"\n[3] 测试 POST 同步接口 (return_audio=True)...")
        print(f"   文本: {text}")
        print(f"   参考音频: {prompt_audio}")

        url = f"{self.base_url}/api/v1/tts/tasks"
        payload = {
            "text": text,
            "spk_audio_prompt": prompt_audio,
            "return_audio": True,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }

        start_time = time.time()
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            elapsed = time.time() - start_time
            print(f"   成功! 耗时: {elapsed:.2f}s")

            # 保存音频文件
            if output_path is None:
                output_path = f"test_output_{uuid.uuid4().hex[:8]}.wav"

            with open(output_path, "wb") as f:
                f.write(response.content)

            # 读取并显示音频信息
            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate
            print(f"   保存到: {output_path}")
            print(f"   采样率: {sample_rate} Hz")
            print(f"   时长: {duration:.2f}s")

            return True

        except requests.exceptions.Timeout:
            print(f"   超时 ({self.timeout}s)")
            return False
        except requests.exceptions.RequestException as e:
            print(f"   失败: {e}")
            return False

    def test_post_async(
        self,
        text: str,
        prompt_audio: str = "tests/sample_prompt.wav",
        output_path: str = None,
    ) -> bool:
        """测试 POST 异步接口（return_audio=False）

        Args:
            text: 要合成的文本
            prompt_audio: 参考音频路径
            output_path: 输出音频路径（可选）

        Returns:
            是否成功
        """
        print(f"\n[4] 测试 POST 异步接口 (return_audio=False)...")
        print(f"   文本: {text}")
        print(f"   参考音频: {prompt_audio}")

        url = f"{self.base_url}/api/v1/tts/tasks"
        payload = {
            "text": text,
            "spk_audio_prompt": prompt_audio,
            "return_audio": False,
            "cfg_value": 2.0,
            "inference_timesteps": 10,
        }

        try:
            # 创建任务
            print("   创建任务...")
            response = self.session.post(url, json=payload, timeout=10)
            response.raise_for_status()
            task_data = response.json()
            task_id = task_data["task_id"]
            print(f"   任务 ID: {task_id}")
            print(f"   初始状态: {task_data['status']}")

            # 轮询任务状态
            status_url = f"{self.base_url}/api/v1/tts/tasks/{task_id}"
            max_wait = self.timeout
            start_time = time.time()

            while time.time() - start_time < max_wait:
                response = self.session.get(status_url, timeout=10)
                response.raise_for_status()
                task_data = response.json()

                status = task_data["status"]
                print(f"   状态: {status} - {task_data.get('message', '')}")

                if status == "completed":
                    break
                elif status == "failed":
                    print(f"   任务失败: {task_data.get('message', '')}")
                    return False

                time.sleep(1)

            elapsed = time.time() - start_time

            if task_data["status"] != "completed":
                print(f"   超时 ({max_wait}s)")
                return False

            print(f"   完成! 总耗时: {elapsed:.2f}s")

            # 获取结果
            result_url = f"{self.base_url}/api/v1/tts/tasks/{task_id}/result"
            response = self.session.get(result_url, timeout=30)
            response.raise_for_status()

            # 保存音频文件
            if output_path is None:
                output_path = f"test_output_{uuid.uuid4().hex[:8]}.wav"

            with open(output_path, "wb") as f:
                f.write(response.content)

            # 读取并显示音频信息
            audio_data, sample_rate = sf.read(output_path)
            duration = len(audio_data) / sample_rate
            print(f"   保存到: {output_path}")
            print(f"   采样率: {sample_rate} Hz")
            print(f"   时长: {duration:.2f}s")

            return True

        except requests.exceptions.RequestException as e:
            print(f"   失败: {e}")
            return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="VoxCPM API 测试脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API 服务地址（远程 IP）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API 服务端口",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="这是一个测试，用于验证 VoxCPM 文本转语音 API 的功能。",
        help="要合成的文本",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="tests/sample_prompt.wav",
        help="参考音频路径（服务端路径）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["get", "post-sync", "post-async", "all"],
        default="all",
        help="测试模式",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="请求超时时间（秒）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出音频文件路径（默认自动生成）",
    )
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("VoxCPM API 测试脚本")
    print("=" * 60)
    print(f"目标地址: {args.host}:{args.port}")
    print(f"测试模式: {args.mode}")
    print(f"超时时间: {args.timeout}s")

    tester = TTSAPITester(host=args.host, port=args.port, timeout=args.timeout)

    # 健康检查
    try:
        tester.health_check()
    except Exception as e:
        print(f"\n连接失败: {e}")
        print(f"\n请确保:")
        print(f"  1. API 服务已启动 (python api.py --model_dir <model_path>)")
        print(f"  2. 服务地址正确 ({args.host}:{args.port})")
        print(f"  3. 防火墙允许访问该端口")
        sys.exit(1)

    # 运行测试
    results = []

    if args.mode in ["get", "all"]:
        results.append(("GET 同步", tester.test_get_sync(
            text=args.text,
            prompt_audio=args.prompt,
            output_path=args.output,
        )))

    if args.mode in ["post-sync", "all"]:
        results.append(("POST 同步", tester.test_post_sync(
            text=args.text,
            prompt_audio=args.prompt,
            output_path=args.output,
        )))

    if args.mode in ["post-async", "all"]:
        results.append(("POST 异步", tester.test_post_async(
            text=args.text,
            prompt_audio=args.prompt,
            output_path=args.output,
        )))

    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print(f"\n整体结果: {'✓ 全部通过' if all_passed else '✗ 部分失败'}")


if __name__ == "__main__":
    main()

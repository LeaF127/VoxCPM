import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import torch
import websockets
from funasr import AutoModel

from voxcpm.utils import get_test_logger


class ASRTranslator:
    """
    简单封装 ASR + 翻译的流式与非流式调用。
    - 流式：边发边收，可通过回调获得中间/最终结果。
    - 非流式：发送完音频后等待最终结果，返回解析后的内容。
    """

    def __init__(
        self,
        ws_url: str,
        user_id: str,
        token: str,
        from_lang: str,
        to_lang: str,
        role: str = "0",
        lan_id: str = "0",
        sample_rate: int = 16000,
        bit_rate: int = 16,
        interval: float = 0.1,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.ws_url = ws_url
        self.user_id = user_id
        self.token = token
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.role = role
        self.lan_id = lan_id
        self.sample_rate = sample_rate
        self.bit_rate = bit_rate
        self.interval = interval
        self.bytes_per_second = self.sample_rate * self.bit_rate // 8
        self.chunk_size = int(self.bytes_per_second * self.interval)
        self.session_id = str(uuid.uuid4())
        self.logger = get_test_logger(__file__, console_output=False)

    def _build_target_url(self) -> str:
        query = (
            f"?lanid={self.lan_id}&userid={self.user_id}&token={self.token}"
            f"&fromlan={self.from_lang}&tolan={self.to_lang}&sid={self.session_id}&role={self.role}"
        )
        return f"{self.ws_url}{query}"

    @staticmethod
    def _parse_message(message: str) -> Optional[Dict[str, Any]]:
        """解析服务端返回的单条消息。"""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return None

        err_code = str(data.get("errCode", ""))
        result_text = data.get("result", "") or ""
        trans_text = data.get("trans", "") or ""

        is_sentence_final = err_code == "0"
        is_stream_finished = err_code == "10" # 这里存疑
        is_intermediate = err_code == "3"

        return {
            "err_code": err_code,
            "result_text": result_text,
            "trans_text": trans_text,
            "is_sentence_final": is_sentence_final,
            "is_stream_finished": is_stream_finished,
            "is_intermediate": is_intermediate,
            "raw": data,
        }

    async def _send_audio(self, ws: websockets.WebSocketClientProtocol, audio_path: Path, stop_event: asyncio.Event) -> None:
        start_time = time.time()
        frame_index = 0

        with audio_path.open("rb") as f:
            while not stop_event.is_set():
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                frame_index += 1
                await ws.send(chunk)
                await asyncio.sleep(self.interval)

        await ws.send("end")
        end_time = time.time()
        self.logger.info("已发送结束命令, 共耗时%.2f秒", end_time - start_time)

    async def _send_audio_stream(
        self,
        ws: websockets.WebSocketClientProtocol,
        audio_stream,
        stop_event: asyncio.Event,
    ) -> None:
        """
        从音频流实时发送音频数据。

        Args:
            ws: WebSocket 连接
            audio_stream: 音频流生成器，每次 yield 返回音频字节数据
            stop_event: 停止事件
        """
        start_time = time.time()
        frame_index = 0

        try:
            async for chunk in audio_stream:
                if stop_event.is_set():
                    break
                if chunk:
                    frame_index += 1
                    await ws.send(chunk)
                    await asyncio.sleep(self.interval)
        except Exception as e:
            self.logger.error("发送音频流时出错: %s", e)
        finally:
            await ws.send("end")
            end_time = time.time()
            self.logger.info("已发送结束命令, 共发送 %d 帧, 耗时 %.2f 秒", frame_index, end_time - start_time)

    async def _handle_message(
        self,
        message: str,
        on_intermediate: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """统一处理一条消息，并按类型触发回调。"""
        self.logger.info("Server Response: %s", message)
        parsed = self._parse_message(message)
        if not parsed:
            self.logger.error("返回值不是 JSON")
            return None

        if parsed["is_intermediate"] and on_intermediate:
            await on_intermediate(parsed)

        # 对于每一句完整的话（errCode=0），触发 on_final，方便逐句 TTS
        if parsed["is_sentence_final"] and on_final:
            await on_final(parsed)

        # 流结束时（errCode=10），如果包含文本，也应该触发 on_final，确保最后一句话被提交合成
        if parsed["is_stream_finished"] and on_final:
            result_text = parsed.get("result_text", "").strip()
            trans_text = parsed.get("trans_text", "").strip()
            if result_text or trans_text:
                await on_final(parsed)

        return parsed

    async def _stream_core(
        self,
        audio_path: Optional[Path] = None,
        audio_stream=None,
        stop_on_final: bool = False,
        on_intermediate: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        collect_all: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        内部流式核心：
        - stop_on_final=True 时，在检测到整体结束（errCode=10）后停止读取。
        - collect_all=True 时，会聚合整段音频中所有句子的结果并返回。
        """
        target_url = self._build_target_url()
        stop_event = asyncio.Event()
        final_result: Optional[Dict[str, Any]] = None
        all_messages: List[Dict[str, Any]] = []

        self.logger.info("Connecting to %s", target_url)

        async with websockets.connect(target_url) as ws:
            if audio_stream is not None:
                send_task = asyncio.create_task(self._send_audio_stream(ws, audio_stream, stop_event))
            elif audio_path is not None:
                send_task = asyncio.create_task(self._send_audio(ws, audio_path, stop_event))
            else:
                raise ValueError("必须提供 audio_path 或 audio_stream 之一")
            try:
                async for message in ws:
                    parsed = await self._handle_message(message, on_intermediate, on_final)
                    if not parsed:
                        continue

                    all_messages.append(parsed)

                    if parsed.get("is_sentence_final") or parsed.get("is_stream_finished"):
                        final_result = parsed

                    # 注意：只在整体结束标记时才根据 stop_on_final 决定是否停止
                    if stop_on_final and parsed.get("is_stream_finished"):
                        stop_event.set()
                        break
            except Exception as exc:  # noqa: BLE001
                self.logger.error("WS Error: %s", exc)
                stop_event.set()
            finally:
                stop_event.set()
                await send_task
                await ws.close()

        if collect_all:
            if not all_messages:
                return None

            # 只统计每一句完整的话（errCode=0）以及可能带文本的整体结束消息
            sentence_segments: List[Dict[str, Any]] = [
                m
                for m in all_messages
                if m.get("is_sentence_final")
                or (m.get("is_stream_finished") and (m.get("result_text") or m.get("trans_text")))
            ]

            result_text = "".join(m.get("result_text", "") for m in sentence_segments)
            trans_text = "".join(m.get("trans_text", "") for m in sentence_segments)

            is_stream_finished = any(m.get("is_stream_finished") for m in all_messages)

            # 找到最后一条“终止相关”消息，用于携带最终 errCode 等信息
            last_final: Optional[Dict[str, Any]] = None
            for m in reversed(all_messages):
                if m.get("is_sentence_final") or m.get("is_stream_finished"):
                    last_final = m
                    break

            return {
                "err_code": last_final.get("err_code") if last_final else "",
                "result_text": result_text,
                "trans_text": trans_text,
                "is_stream_finished": is_stream_finished,
                "segments": sentence_segments,
                "raw_messages": all_messages,
            }

        return final_result

    async def transcribe_streaming(
        self,
        audio_path: Optional[str] = None,
        audio_stream=None,
        on_intermediate: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> None:
        """
        流式模式：发送音频并通过回调获取结果（逐句回调）。

        Args:
            audio_path: 音频文件路径（与 audio_stream 二选一）
            audio_stream: 音频流生成器（与 audio_path 二选一）
            on_intermediate: 中间结果回调
            on_final: 最终结果回调
        """
        await self._stream_core(
            audio_path=Path(audio_path) if audio_path else None,
            audio_stream=audio_stream,
            stop_on_final=False,
            on_intermediate=on_intermediate,
            on_final=on_final,
            collect_all=False,
        )

    async def transcribe_non_streaming(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        非流式模式：
        - 等待整段音频识别/翻译结束（errCode=10 或连接结束）。
        - 聚合所有句子的 result/trans 后返回。

        返回结构示例：
        {
            "err_code": "10",
            "result_text": "整段识别文本...",
            "trans_text": "整段翻译文本...",
            "is_stream_finished": True,
            "segments": [... 每句的字典列表 ...],
            "raw_messages": [... 所有原始解析后的消息 ...],
        }
        """
        return await self._stream_core(audio_path=Path(audio_path), stop_on_final=True, collect_all=True)

    async def run(
        self,
        audio_path: Optional[str] = None,
        audio_stream=None,
        streaming: bool = True,
        on_intermediate: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        根据 streaming 参数选择流式/非流式。
        - streaming=True：边发边收，通过回调处理结果（不返回最终聚合结果）。
        - streaming=False：等待最终返回，直接返回聚合后的结果字典。

        Args:
            audio_path: 音频文件路径（与 audio_stream 二选一）
            audio_stream: 音频流生成器（与 audio_path 二选一）
            streaming: 是否使用流式模式
            on_intermediate: 中间结果回调
            on_final: 最终结果回调
        """
        if streaming:
            await self.transcribe_streaming(
                audio_path=audio_path, audio_stream=audio_stream, on_intermediate=on_intermediate, on_final=on_final
            )
            return None
        if audio_stream is not None:
            raise ValueError("非流式模式不支持 audio_stream，请使用 audio_path")
        return await self.transcribe_non_streaming(audio_path)


class ASRRecognizer:
    """
    基于 FunASR 的简单 ASR 识别类。
    参考 app.py 中的实现，仅做语音识别（不包含翻译）。
    """

    def __init__(
        self,
        model_id: str = "iic/SenseVoiceSmall",
        device: Optional[str] = None,
        disable_update: bool = True,
        log_level: str = "DEBUG",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """
        初始化 ASR 识别器。

        Args:
            model_id: FunASR 模型 ID，默认为 "iic/SenseVoiceSmall"
            device: 设备（"cuda:0", "cpu" 等），如果为 None 则自动检测
            disable_update: 是否禁用模型更新
            log_level: 日志级别
            logger: 自定义 logger，如果为 None 则使用默认 logger
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model_id = model_id
        self.device = device
        self.logger = get_test_logger(__file__, console_output=False)

        self.logger.info("初始化 ASR 模型: %s, 设备: %s", model_id, device)
        self.asr_model = AutoModel(
            model=model_id,
            disable_update=disable_update,
            log_level=log_level,
            device=device,
        )
        self.logger.info("ASR 模型加载完成")

    def recognize(self, audio_path: Optional[str], language: str = "auto", use_itn: bool = True) -> str:
        """
        识别音频文件，返回识别文本。

        Args:
            audio_path: 音频文件路径，如果为 None 或空字符串则返回空字符串
            language: 语言设置，"auto" 表示自动检测
            use_itn: 是否使用逆文本正则化（Inverse Text Normalization）

        Returns:
            识别得到的文本字符串
        """
        if not audio_path:
            return ""

        try:
            res = self.asr_model.generate(input=audio_path, language=language, use_itn=use_itn)
            text = res[0]["text"].split("|>")[-1]
            self.logger.debug("识别结果: %s", text)
            return text
        except Exception as e:  # noqa: BLE001
            self.logger.error("ASR 识别失败: %s", e)
            return ""

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 FunASR 的简单 ASR 识别 Demo")
    parser.add_argument("--audio-path", type=str, default="s2st_demo/input/example.wav", help="音频文件路径")
    parser.add_argument("--model-id", type=str, default="iic/SenseVoiceSmall", help="模型 ID")
    parser.add_argument("--device", type=str, default="cuda:0", help="设备")
    parser.add_argument("--use_itn", action="store_true", help="是否使用逆文本正则化")
    return parser

def main():
    """
    命令行入口：单次音频识别测试。
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    asr = ASRRecognizer()
    text = asr.recognize(audio_path=args.audio_path, use_itn=args.use_itn)

if __name__ == "__main__":
    main()
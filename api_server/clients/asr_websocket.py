"""
WebSocket ASR 客户端。

通过 WebSocket 连接远程 ASR/翻译服务，支持流式和非流式模式。
"""
import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import websockets

from voxcpm.utils import get_test_logger

# 导入 timing_stats（兼容处理）
try:
    from s2st_demo.utils.timing_stats import TimingStats
except ImportError:
    TimingStats = None  # type: ignore


class ASRWebSocketClient:
    """
    WebSocket ASR/翻译客户端。

    支持流式和非流式两种模式：
    - 流式：边发边收，可通过回调获得中间/最终结果
    - 非流式：发送完音频后等待最终结果，返回解析后的内容
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
        timing_stats: Optional[TimingStats] = None,
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
        self.logger = logger or get_test_logger(__file__, console_output=False)
        self.timing_stats = timing_stats
        self.global_t0: Optional[float] = None

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
        is_stream_finished = err_code == "10"
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
        start_time = time.perf_counter()
        frame_index = 0

        with audio_path.open("rb") as f:
            while not stop_event.is_set():
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                frame_index += 1
                if self.timing_stats is not None and self.timing_stats.t1 is None and self.global_t0 is not None:
                    self.timing_stats.t1 = time.perf_counter() - self.global_t0
                if self.timing_stats is not None:
                    self.timing_stats.chunks_sent += 1
                await ws.send(chunk)
                await asyncio.sleep(self.interval)

        await ws.send("end")
        end_time = time.perf_counter()
        self.logger.info("已发送结束命令, 共耗时%.2f秒", end_time - start_time)

    async def _send_audio_stream(
        self,
        ws: websockets.WebSocketClientProtocol,
        audio_stream,
        stop_event: asyncio.Event,
    ) -> None:
        """从音频流实时发送音频数据。"""
        start_time = time.perf_counter()
        frame_index = 0

        try:
            async for chunk in audio_stream:
                if stop_event.is_set():
                    break

                if self.timing_stats is not None and self.timing_stats.t1 is None and self.global_t0 is not None:
                    self.timing_stats.t1 = time.perf_counter() - self.global_t0

                if chunk:
                    frame_index += 1
                    await ws.send(chunk)
                    await asyncio.sleep(self.interval)
        except Exception as e:
            self.logger.error("发送音频流时出错: %s", e)
        finally:
            await ws.send("end")
            end_time = time.perf_counter()
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

        if parsed["is_sentence_final"] and on_final:
            if self.timing_stats is not None and self.timing_stats.t2 is None and self.global_t0 is not None:
                self.timing_stats.t2 = time.perf_counter() - self.global_t0
            await on_final(parsed)

        if parsed["is_stream_finished"] and on_final:
            result_text = parsed.get("result_text", "").strip()
            trans_text = parsed.get("trans_text", "").strip()
            if result_text or trans_text:
                if self.timing_stats is not None and self.timing_stats.t2 is None and self.global_t0 is not None:
                    self.timing_stats.t2 = time.perf_counter() - self.global_t0
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
        """内部流式核心。"""
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

                    if stop_on_final and parsed.get("is_stream_finished"):
                        stop_event.set()
                        break
            except Exception as exc:
                self.logger.error("WS Error: %s", exc)
                stop_event.set()
            finally:
                stop_event.set()
                await send_task
                await ws.close()

        if collect_all:
            if not all_messages:
                return None

            sentence_segments: List[Dict[str, Any]] = [
                m
                for m in all_messages
                if m.get("is_sentence_final")
                or (m.get("is_stream_finished") and (m.get("result_text") or m.get("trans_text")))
            ]

            result_text = "".join(m.get("result_text", "") for m in sentence_segments)
            trans_text = "".join(m.get("trans_text", "") for m in sentence_segments)

            is_stream_finished = any(m.get("is_stream_finished") for m in all_messages)

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
        非流式模式：等待整段音频识别/翻译结束，聚合所有句子的 result/trans 后返回。
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
        """
        if streaming:
            await self.transcribe_streaming(
                audio_path=audio_path, audio_stream=audio_stream, on_intermediate=on_intermediate, on_final=on_final
            )
            return None
        if audio_stream is not None:
            raise ValueError("非流式模式不支持 audio_stream，请使用 audio_path")
        return await self.transcribe_non_streaming(audio_path)

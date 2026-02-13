"""
ASR WebSocket 服务连接测试脚本

用于测试 ASR 服务的连接和基本功能。
"""
import asyncio
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from api_server.utils.settings import get_settings
from api_server.clients.asr_websocket import ASRWebSocketClient
from voxcpm.utils import get_test_logger

_logger = get_test_logger(__file__)


async def test_asr_connection(ws_url: str, user_id: str, token: str, from_lang: str = "zh", to_lang: str = "en"):
    """
    测试 ASR WebSocket 连接

    Args:
        ws_url: WebSocket 服务地址
        user_id: 用户 ID
        token: 认证 Token
        from_lang: 源语言
        to_lang: 目标语言
    """
    _logger.info("=" * 60)
    _logger.info("ASR WebSocket 连接测试")
    _logger.info("=" * 60)
    _logger.info(f"服务地址: {ws_url}")
    _logger.info(f"用户 ID: {user_id}")
    _logger.info(f"Token: {token[:4]}****{token[-4:] if len(token) > 8 else ''}")
    _logger.info(f"源语言: {from_lang} -> 目标语言: {to_lang}")
    _logger.info("-" * 60)

    # 创建客户端
    client = ASRWebSocketClient(
        ws_url=ws_url,
        user_id=user_id,
        token=token,
        from_lang=from_lang,
        to_lang=to_lang,
        logger=_logger,
    )

    # 打印完整的 WebSocket URL
    target_url = client._build_target_url()
    _logger.info(f"完整连接 URL: {target_url}")
    _logger.info("-" * 60)

    # 测试连接
    _logger.info("正在尝试连接...")
    try:
        import websockets
        async with websockets.connect(target_url, close_timeout=10) as ws:
            _logger.info("✓ 连接成功！WebSocket 握手完成")

            # 等待一下，看是否有欢迎消息
            try:
                welcome_msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                _logger.info(f"收到服务器消息: {welcome_msg}")
            except asyncio.TimeoutError:
                _logger.info("未收到欢迎消息（正常，需要发送音频数据）")

            # 发送结束命令，优雅关闭
            await ws.send("end")
            _logger.info("✓ 测试完成，连接已正常关闭")

            return True

    except websockets.exceptions.InvalidHandshake as e:
        _logger.error(f"✗ WebSocket 握手失败: {e}")
        _logger.error("可能的原因：")
        _logger.error("  1. URL 格式错误（应使用 ws:// 或 wss://）")
        _logger.error("  2. 服务器不支持 WebSocket 协议")
        _logger.error("  3. 需要认证但认证信息错误")
        return False

    except websockets.exceptions.InvalidMessage as e:
        _logger.error(f"✗ 无效的 HTTP 响应: {e}")
        _logger.error("可能的原因：")
        _logger.error("  1. 服务器未运行")
        _logger.error("  2. URL 指向的不是 WebSocket 服务")
        _logger.error("  3. 防火墙或网络问题")
        return False

    except ConnectionRefusedError:
        _logger.error("✗ 连接被拒绝")
        _logger.error("可能的原因：")
        _logger.error("  1. ASR 服务未运行")
        _logger.error("  2. 端口号错误")
        _logger.error("  3. 防火墙阻止连接")
        return False

    except OSError as e:
        _logger.error(f"✗ 系统错误: {e}")
        _logger.error("可能的原因：")
        _logger.error("  1. 主机名无法解析（DNS 问题）")
        _logger.error("  2. 网络不可达")
        return False

    except Exception as e:
        _logger.error(f"✗ 未知错误: {type(e).__name__}: {e}")
        return False


async def test_with_audio_file(
    ws_url: str,
    user_id: str,
    token: str,
    audio_path: str,
    from_lang: str = "zh",
    to_lang: str = "en",
):
    """
    使用音频文件测试完整的识别流程

    Args:
        ws_url: WebSocket 服务地址
        user_id: 用户 ID
        token: 认证 Token
        audio_path: 测试音频文件路径
        from_lang: 源语言
        to_lang: 目标语言
    """
    _logger.info("=" * 60)
    _logger.info("ASR 识别功能测试（使用音频文件）")
    _logger.info("=" * 60)
    _logger.info(f"音频文件: {audio_path}")
    _logger.info("-" * 60)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        _logger.error(f"音频文件不存在: {audio_path}")
        return False

    client = ASRWebSocketClient(
        ws_url=ws_url,
        user_id=user_id,
        token=token,
        from_lang=from_lang,
        to_lang=to_lang,
        logger=_logger,
    )

    results = []

    async def on_intermediate(parsed):
        """中间结果回调"""
        result_text = parsed.get("result_text", "")
        trans_text = parsed.get("trans_text", "")
        _logger.info(f"[中间] ASR: {result_text} | 翻译: {trans_text}")

    async def on_final(parsed):
        """最终结果回调"""
        result_text = parsed.get("result_text", "")
        trans_text = parsed.get("trans_text", "")
        is_finished = parsed.get("is_stream_finished", False)
        status = "[完成]" if is_finished else "[句子]"
        _logger.info(f"{status} ASR: {result_text} | 翻译: {trans_text}")
        results.append(parsed)

    try:
        _logger.info("开始流式识别...")
        await client.transcribe_streaming(
            audio_path=str(audio_path),
            on_intermediate=on_intermediate,
            on_final=on_final,
        )

        if results:
            _logger.info("=" * 60)
            _logger.info(f"✓ 识别完成，共收到 {len(results)} 条结果")
            return True
        else:
            _logger.warning("未收到任何识别结果")
            return False

    except Exception as e:
        _logger.error(f"识别失败: {e}", exc_info=True)
        return False


async def main():
    """主函数"""
    # 加载配置
    settings = get_settings()

    # 验证 ASR 配置
    is_valid, error_msg = settings.validate_asr_config()
    if not is_valid:
        _logger.error(f"配置错误: {error_msg}")
        _logger.info("请检查 .env 文件中的配置：")
        _logger.info("  WS_URL=ws://your-asr-server:port/ws")
        _logger.info("  USER_ID=your_user_id")
        _logger.info("  TOKEN=your_token")
        return

    # 测试基本连接
    success = await test_asr_connection(
        ws_url=settings.ws_url,
        user_id=settings.user_id,
        token=settings.token,
        from_lang=settings.from_lang,
        to_lang=settings.to_lang,
    )

    if not success:
        _logger.error("\n连接测试失败，请检查配置和服务状态")
        return

    _logger.info("\n" + "=" * 60)
    _logger.info("连接测试成功！")

    # 如果有测试音频文件，进行完整功能测试
    test_audio_path = project_root / "tests" / "test_audio.wav"
    if test_audio_path.exists():
        _logger.info("\n检测到测试音频文件，进行完整功能测试...\n")
        await test_with_audio_file(
            ws_url=settings.ws_url,
            user_id=settings.user_id,
            token=settings.token,
            audio_path=str(test_audio_path),
            from_lang=settings.from_lang,
            to_lang=settings.to_lang,
        )
    else:
        _logger.info(f"\n未找到测试音频文件: {test_audio_path}")
        _logger.info("如需测试完整识别功能，请放置测试音频文件")


if __name__ == "__main__":
    asyncio.run(main())

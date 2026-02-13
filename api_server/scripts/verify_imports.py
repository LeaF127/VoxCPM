"""
验证重构后的导入路径。

测试所有新模块是否可以正确导入。
"""
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_imports():
    """测试所有新导入路径"""
    print("=" * 60)
    print("验证重构后的导入路径")
    print("=" * 60)

    errors = []
    success_count = 0

    # 测试 1: clients 模块
    print("\n[1/6] 测试 clients 模块...")
    try:
        from s2st_demo.clients.asr_websocket import ASRWebSocketClient
        print("  ✓ ASRWebSocketClient 导入成功")
        from s2st_demo.clients.asr_local import ASRLocalRecognizer
        print("  ✓ ASRLocalRecognizer 导入成功")
        success_count += 1
    except Exception as e:
        errors.append(f"clients 模块导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 测试 2: models 模块
    print("\n[2/6] 测试 models 模块...")
    try:
        from s2st_demo.models.tts_wrapper import TTSModelWrapper
        print("  ✓ TTSModelWrapper 导入成功")
        success_count += 1
    except Exception as e:
        errors.append(f"models 模块导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 测试 3: api 模块
    print("\n[3/6] 测试 api 模块...")
    try:
        from s2st_demo.api.models import StreamingRequest, SegmentResult
        print("  ✓ StreamingRequest, SegmentResult 导入成功")
        from s2st_demo.api.dependencies import get_tts_model, get_translator
        print("  ✓ get_tts_model, get_translator 导入成功")
        from s2st_demo.api.routes import app
        print("  ✓ FastAPI app 导入成功")
        success_count += 1
    except Exception as e:
        errors.append(f"api 模块导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 测试 4: services 模块
    print("\n[4/6] 测试 services 模块...")
    try:
        from s2st_demo.services.s2st_service import S2STService
        print("  ✓ S2STService 导入成功")
        success_count += 1
    except Exception as e:
        errors.append(f"services 模块导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 测试 5: utils 模块
    print("\n[5/6] 测试 utils 模块...")
    try:
        from s2st_demo.utils.config import S2STConfig, ASRConfig, TTSConfig
        print("  ✓ S2STConfig, ASRConfig, TTSConfig 导入成功")
        from s2st_demo.utils.timing_stats import TimingStats
        print("  ✓ TimingStats 导入成功")
        success_count += 1
    except Exception as e:
        errors.append(f"utils 模块导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 测试 6: 兼容层
    print("\n[6/6] 测试兼容层（旧导入路径）...")
    try:
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from s2st_demo.asr_translate import ASRTranslator
            if w:
                print(f"  ✓ ASRTranslator 导入成功（收到 DeprecationWarning）")
            else:
                print(f"  ⚠ ASRTranslator 导入成功（未收到 DeprecationWarning）")
        success_count += 1
    except Exception as e:
        errors.append(f"兼容层导入失败: {e}")
        print(f"  ✗ 错误: {e}")

    # 打印总结
    print("\n" + "=" * 60)
    print(f"测试结果: {success_count}/6 组模块导入成功")
    if errors:
        print(f"\n发现的错误:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ 所有导入路径验证通过！")
    print("=" * 60)

    return len(errors) == 0


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

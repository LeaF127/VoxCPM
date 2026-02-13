"""
端到端 S2ST Pipeline 入口（兼容层）。

此文件保留用于向后兼容。新代码应使用：
- python -m s2st_demo.scripts.run_pipeline
"""
import warnings
import sys
from pathlib import Path

# 添加父目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# 发出迁移警告
warnings.warn(
    "直接运行 run.py 已过时。请使用: python -m s2st_demo.scripts.run_pipeline",
    DeprecationWarning,
    stacklevel=2
)

# 导入并运行新的脚本
from s2st_demo.scripts.run_pipeline import main

if __name__ == "__main__":
    main()

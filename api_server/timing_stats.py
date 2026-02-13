"""
时间统计工具类（兼容层）。

此文件保留用于向后兼容。新代码应使用：
- from s2st_demo.utils.timing_stats import TimingStats, TimingStatsCollector
"""
import warnings

# 导入新的类
from s2st_demo.utils.timing_stats import TimingStats, TimingStatsCollector

# 发出迁移警告
warnings.warn(
    "直接从 timing_stats 导入已过时。请使用: "
    "from s2st_demo.utils.timing_stats import TimingStats, TimingStatsCollector",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["TimingStats", "TimingStatsCollector"]

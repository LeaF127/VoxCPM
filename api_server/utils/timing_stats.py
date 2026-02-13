"""
时间统计工具类，用于记录和计算端到端语音翻译流程的各项性能指标。
"""
import time
from dataclasses import dataclass
from typing import Optional, List, Dict

__all__ = ["TimingStats", "TimingStatsCollector"]


@dataclass
class TimingStats:
    """单次处理的时间统计"""
    # 时间点（秒）
    t0: Optional[float] = None  # 用户音频采集完成的时间，用于计算相对时间
    t1: Optional[float] = None  # 单句话首个chunk送入ASR的时间
    t2: Optional[float] = None  # 单句话ASR完成的时间
    t3: Optional[float] = None  # 单句话TTS开始合成
    t4: Optional[float] = None  # 单句话最后一帧语音播放完成

    # 用于计算asr rtf的额外参数
    asr_send_interval: Optional[float] = 0.1 # ASR 发送音频的间隔时间，单位为秒
    chunks_sent: Optional[int] = 0 # 已发送的音频块数量

    # 音频时长（秒）
    input_audio_duration: Optional[float] = None  # 单句话输入音频时长
    output_audio_duration: Optional[float] = None  # 单句话输出音频时长

    # 获取发送间隔时间
    def __init__(self, asr_send_interval: float = 0.1):
        self.asr_send_interval = asr_send_interval

    def calculate_metrics(self) -> Dict[str, Optional[float]]:
        """计算各项指标"""
        metrics = {}

        # 计算 ASR 对于单句话的延迟和RTF
        if self.t1 is not None and self.t2 is not None:
            metrics["asr_latency"] = self.t2 - self.t1  - self.asr_send_interval * (self.chunks_sent - 1)
            metrics["asr_rtf"] = metrics["asr_latency"] / self.input_audio_duration

        # 计算 TTS 对于单句话的延迟和RTF
        if self.t3 is not None and self.t4 is not None:
            metrics["tts_latency"] = self.t4 - self.t3
            metrics["tts_rtf"] = metrics["tts_latency"] / self.output_audio_duration

        # 计算系统对于单句话的延迟和RTF
        if self.t1 is not None and self.t4 is not None:
            metrics["system_e2e"] = self.t4 - self.t1
            metrics["system_rtf"] = metrics["system_e2e"] / self.input_audio_duration

        return metrics

    def format_report(self) -> str:
        """格式化输出报告"""
        metrics = self.calculate_metrics()
        lines = []

        lines.append("=" * 80)
        lines.append("性能指标统计报告")
        lines.append("=" * 80)

        # 时间点
        lines.append("\n时间点 (秒):")
        if self.t0 is not None:
            lines.append(f"  T0 (用户音频采集完成):     {self.t0:.4f}")
        if self.t1 is not None:
            lines.append(f"  T1 (单句话首个chunk送入ASR):     {self.t1:.4f}")
        if self.t2 is not None:
            lines.append(f"  T2 (单句话ASR完成):     {self.t2:.4f}")
        if self.t3 is not None:
            lines.append(f"  T3 (单句话TTS开始合成):     {self.t3:.4f}")
        if self.t4 is not None:
            lines.append(f"  T4 (单句话最后一帧语音播放完成):     {self.t4:.4f}")

        # 音频时长
        lines.append("\n音频时长 (秒):")
        if self.input_audio_duration is not None:
            lines.append(f"  单句话输入音频时长: {self.input_audio_duration:.4f}")
        if self.output_audio_duration is not None:
            lines.append(f"  单句话输出音频时长: {self.output_audio_duration:.4f}")

        # 指标
        lines.append("\n性能指标:")
        if metrics.get("asr_latency") is not None:
            lines.append(f"  ASR 总延迟:       {metrics['asr_latency']*1000:.2f} ms")
        if metrics.get("asr_rtf") is not None:
            lines.append(f"  ASR RTF:           {metrics['asr_rtf']:.4f}")
        if metrics.get("tts_latency") is not None:
            lines.append(f"  TTS 总延迟:       {metrics['tts_latency']*1000:.2f} ms")
        if metrics.get("tts_rtf") is not None:
            lines.append(f"  TTS RTF:           {metrics['tts_rtf']:.4f}")
        if metrics.get("system_e2e") is not None:
            lines.append(f"  系统总延迟:       {metrics['system_e2e']:.4f} s")
        if metrics.get("system_rtf") is not None:
            lines.append(f"  系统RTF:           {metrics['system_rtf']:.4f}")

        lines.append("=" * 80)
        return "\n".join(lines)


class TimingStatsCollector:
    """时间统计收集器，支持多句流式处理"""
    def __init__(self):
        self.stats_list: List[TimingStats] = []
        self.current_stats: Optional[TimingStats] = None
        self.global_t0: Optional[float] = None  # 全局开始时间

    def start_global(self):
        """开始全局计时（T0）"""
        self.global_t0 = time.perf_counter()

    def start_segment(self) -> TimingStats:
        """开始新的一段处理"""
        self.current_stats = TimingStats()
        # T0 作为相对时间的基准点，设为 0.0
        # 其他时间点（t1, t2, t3, t4）都是相对于 global_t0 计算的相对时间
        self.current_stats.t0 = 0.0
        self.current_stats.chunks_sent = 0
        return self.current_stats

    def finish_segment(self):
        """完成当前段处理，保存到列表"""
        if self.current_stats is not None:
            self.stats_list.append(self.current_stats)
            self.current_stats = None

    def get_current_stats(self) -> Optional[TimingStats]:
        """获取当前段的统计"""
        return self.current_stats

    def format_summary_report(self, debug: bool = False) -> str:
        """格式化汇总报告（多句时使用）

        Args:
            debug: 如果为 True，输出所有时间点（包括None）和所有计算的指标
        """
        if not self.stats_list:
            return "无统计数据"

        lines = []
        lines.append("=" * 80)
        lines.append("性能指标汇总报告（多句处理）")
        if debug:
            lines.append("【DEBUG 模式：显示所有时间点和指标】")
        lines.append("=" * 80)
        lines.append(f"总句数: {len(self.stats_list)}")

        # 计算平均值
        all_metrics = [stats.calculate_metrics() for stats in self.stats_list]
        # 定义指标名称和中文名称的映射
        metric_map = {
            "asr_latency": "ASR 总延迟",
            "tts_latency": "TTS 总延迟",
            "system_e2e": "系统总延迟",
            "asr_rtf": "ASR RTF",
            "tts_rtf": "TTS RTF",
            "system_rtf": "系统RTF",
        }
        metric_names = ["asr_latency", "tts_latency", "system_e2e", "asr_rtf", "tts_rtf", "system_rtf"]

        lines.append("\n平均指标:")
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if m.get(metric_name) is not None]
            if values:
                avg = sum(values) / len(values)
                chinese_name = metric_map.get(metric_name, metric_name)
                # 仅排除系统RTF
                if "system" in metric_name and "e2e" not in metric_name:
                    continue
                if "rtf" in metric_name:
                    lines.append(f"  {chinese_name}: {avg:.4f}")
                elif "e2e" in metric_name:
                    lines.append(f"  {chinese_name}: {avg:.4f} s")
                else:
                    lines.append(f"  {chinese_name}: {avg*1000:.2f} ms")
            elif debug:
                # Debug 模式下，即使没有值也显示
                chinese_name = metric_map.get(metric_name, metric_name)
                lines.append(f"  {chinese_name}: None")

        # 显示每句的详细统计
        lines.append("\n各句详细统计:")
        for i, stats in enumerate(self.stats_list, 1):
            lines.append(f"\n--- 第 {i} 句 ---")

            if debug:
                # Debug 模式：显示所有时间点
                lines.append("  时间点 (秒):")
                lines.append(f"    T0: {stats.t0:.4f}" if stats.t0 is not None else "    T0: None")
                lines.append(f"    T1: {stats.t1:.4f}" if stats.t1 is not None else "    T1: None")
                lines.append(f"    T2: {stats.t2:.4f}" if stats.t2 is not None else "    T2: None")
                lines.append(f"    T3: {stats.t3:.4f}" if stats.t3 is not None else "    T3: None")
                lines.append(f"    T4: {stats.t4:.4f}" if stats.t4 is not None else "    T4: None")
                lines.append(f"    单句话输入音频时长: {stats.input_audio_duration:.4f}" if stats.input_audio_duration is not None else "    单句话输入音频时长: None")
                lines.append(f"    单句话输出音频时长: {stats.output_audio_duration:.4f}" if stats.output_audio_duration is not None else "    单句话输出音频时长: None")

            metrics = stats.calculate_metrics()

            if debug:
                # Debug 模式：显示所有计算的指标
                lines.append("  计算的指标:")
                for metric_name in metric_names:
                    chinese_name = metric_map.get(metric_name, metric_name)
                    value = metrics.get(metric_name)
                    if value is not None:
                        if "rtf" in metric_name:
                            lines.append(f"    {chinese_name}: {value:.4f}")
                        elif "e2e" in metric_name:
                            lines.append(f"    {chinese_name}: {value:.4f} s")
                        else:
                            lines.append(f"    {chinese_name}: {value*1000:.2f} ms")
                    else:
                        lines.append(f"    {chinese_name}: None")
            else:
                # 非 Debug 模式：只显示有值的指标
                if metrics.get("asr_latency") is not None:
                    lines.append(f"  ASR 总延迟: {metrics['asr_latency']*1000:.2f} ms")
                if metrics.get("tts_latency") is not None:
                    lines.append(f"  TTS 总延迟: {metrics['tts_latency']*1000:.2f} ms")
                if metrics.get("asr_rtf") is not None:
                    lines.append(f"  ASR RTF: {metrics['asr_rtf']:.4f}")
                if metrics.get("tts_rtf") is not None:
                    lines.append(f"  TTS RTF: {metrics['tts_rtf']:.4f}")

        lines.append("=" * 80)
        return "\n".join(lines)

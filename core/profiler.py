from __future__ import annotations

import time


class Profiler:
    """性能分析器 - 记录各阶段耗时 / Profiler - records elapsed time for each stage"""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}  # 阶段名 -> 开始时间 / Stage name -> start time
        self._records: dict[str, list[float]] = {}  # 阶段名 -> 耗时列表 / Stage name -> elapsed time list

    def start(self, stage: str) -> None:
        """开始计时 / Start timing"""
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> None:
        """停止计时并记录 / Stop timing and record"""
        if stage not in self._starts:
            return
        elapsed = time.perf_counter() - self._starts[stage]
        self._records.setdefault(stage, []).append(elapsed)
        del self._starts[stage]

    def report(self) -> dict[str, float]:
        """生成耗时报告 / Generate timing report"""
        result = {}
        for stage, times in self._records.items():
            result[stage] = sum(times)
        return result

    def reset(self) -> None:
        """重置所有计时记录 / Reset all timing records"""
        self._starts.clear()
        self._records.clear()

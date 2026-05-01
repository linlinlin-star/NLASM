from __future__ import annotations

from typing import Any

from .inline_cache import InlineCache
from .pipeline import PipelineV08
from .profiler import Profiler


class Runtime:
    """NLASM运行时 - 封装Pipeline提供高层执行接口 / NLASM runtime - wraps Pipeline to provide high-level execution interface.

    支持直接执行和带追踪的执行（含性能分析和缓存命中信息）。
    Supports direct execution and traced execution (with profiling and cache hit info).
    """

    def __init__(self, pipeline: PipelineV08, profiler: Profiler | None = None) -> None:
        self.pipeline = pipeline
        self.profiler = profiler or Profiler()

    def run(self, nl: str) -> object:
        """执行自然语言输入 / Execute natural language input"""
        return self.pipeline.compile_and_run(nl)

    def run_with_trace(self, nl: str) -> dict[str, Any]:
        """带追踪的执行 - 返回详细的执行过程信息 / Traced execution - returns detailed execution process info.

        返回字典包含: 原始输入、规范化文本、匹配的Pattern、分数、槽位、缓存命中、结果、性能数据。
        Returns dict with: raw input, normalized text, matched pattern, score, slots, cache hit, result, profiling data.
        """
        from .frontend import Frontend

        self.profiler.reset()

        self.profiler.start("frontend")
        packet = self.pipeline.frontend.process(nl)
        self.profiler.stop("frontend")

        self.profiler.start("match")
        match_text = packet.semantic_skeleton or packet.normalized
        matches = self.pipeline.matcher.match(match_text, top_k=3)
        self.profiler.stop("match")

        pattern, score = matches[0] if matches else (None, 0.0)

        trace: dict[str, Any] = {
            "raw": nl,
            "normalized": packet.normalized,
            "pattern": pattern.name if pattern else None,
            "score": score,
            "slots": None,
            "cache_hit": False,
            "result": None,
            "profiling": {},
        }

        if pattern is not None and score >= 0.70:
            self.profiler.start("fill")
            slots = self.pipeline.filler.fill(pattern, packet)
            self.profiler.stop("fill")
            trace["slots"] = {k: repr(v) for k, v in slots.items()}

            from .inline_cache import build_cache_key, normalize_type_signature
            cache_key = build_cache_key(pattern.name, normalize_type_signature(slots))
            compiled = self.pipeline.cache.lookup(cache_key)
            if compiled is not None:
                trace["cache_hit"] = True

            self.profiler.start("execute")
            result = self.pipeline.compile_and_run(nl)
            self.profiler.stop("execute")
            trace["result"] = result
        else:
            self.profiler.start("execute")
            result = self.pipeline.compile_and_run(nl)
            self.profiler.stop("execute")
            trace["result"] = result

        trace["profiling"] = self.profiler.report()
        return trace

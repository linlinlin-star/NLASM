from __future__ import annotations

import re
from typing import Any

from .decoder import SemanticDecoder
from .embedding_cache import EmbeddingCache
from .entities import ENTITY_OP
from .frontend import Frontend
from .inline_cache import InlineCache, build_cache_key, normalize_type_signature
from .ir_interpreter import IRInterpreter
from .ir_pattern import IRPattern
from .parallel_compiler import ParallelCompiler
from .pattern_instantiator import PatternInstantiator
from .pattern_matcher import PatternMatcher
from .python_api import NLASMFunction, FuncSignature, FuncParam, _compile_nlasm_code
from .slot_filler import SlotFiller
from .slot_types import ArraySlot
from .specialization import SpecializationEngine, AOTCompiler
from .tiered_compiler import TieredCompiler
from .cfg import CFG
from .cfg_builder import CFGBuilder
from .dominator import DominatorTree
from .ssa_builder import SSABuilder
from .optimizer import Optimizer
from .llvm_codegen import LLVMCodeGen
from .jit_executor import JITExecutor


FILTER_KEYWORDS = {"大于", "小于", "大于等于", "小于等于", "等于", "不等于", "超过", "高于", "低于", "满足条件", "过滤"}


class PipelineV08:
    """NLASM v0.8 编译流水线（高级优化版）/ NLASM v0.8 compilation pipeline (advanced optimization version).

    执行路径 / Execution paths:
    1. Pattern路径: NL -> Frontend -> PatternMatcher -> SlotFiller -> Instantiator -> TieredCompiler -> (JIT|解释)
    2. 降级路径: NL -> Frontend -> SemanticDecoder -> 解释

    新增优化 / New optimizations:
    - EmbeddingCache: 缓存语义编码结果，避免重复推理
    - TieredCompiler: 分层编译，热点自动JIT
    - SpecializationEngine: 类型专用化
    - ParallelCompiler: 后台并行编译
    - AOTCompiler: 预编译标准库
    """

    def __init__(
        self,
        frontend: Frontend,
        matcher: PatternMatcher,
        filler: SlotFiller,
        instantiator: PatternInstantiator,
        decoder: SemanticDecoder,
        cache: InlineCache,
        interpreter: type[IRInterpreter] = IRInterpreter,
        use_jit: bool = True,
    ) -> None:
        self.frontend = frontend
        self.matcher = matcher
        self.filler = filler
        self.instantiator = instantiator
        self.decoder = decoder
        self.cache = cache
        self.interpreter_cls = interpreter
        self.use_jit = use_jit

        # Embedding缓存 — 包装原始embedder / Embedding cache — wrap original embedder
        if hasattr(frontend, 'embedder') and frontend.embedder is not None:
            self.embedding_cache = EmbeddingCache(frontend.embedder)
            self._original_encode = frontend.embedder.encode
        else:
            self.embedding_cache = None
            self._original_encode = None

        # 专用化引擎 / Specialization engine
        self.specialization = SpecializationEngine()

        # 并行编译器 / Parallel compiler
        self.parallel_compiler = ParallelCompiler(max_workers=2) if use_jit else None

        # AOT编译器 / AOT compiler
        self.aot_compiler = AOTCompiler()

        # JIT编译链组件 / JIT compilation chain components
        if self.use_jit:
            self.cfg_builder = CFGBuilder()
            self.ssa_builder = SSABuilder()
            self.optimizer = Optimizer()
            self.codegen = LLVMCodeGen()
            self.jit_executor = JITExecutor(opt_level=3)

    def warmup(self) -> None:
        """预热流水线 — 缓存embedding + 预编译标准库 / Warmup pipeline — cache embeddings + precompile stdlib"""
        # 预热embedding缓存 / Warmup embedding cache
        if self.embedding_cache is not None:
            from .pattern_database import PATTERN_DB
            warmup_texts = []
            for pattern in PATTERN_DB:
                warmup_texts.append(pattern.description)
                warmup_texts.extend(pattern.examples)
            self.embedding_cache.warmup(warmup_texts)

        # 预编译标准库 / Precompile stdlib
        try:
            self.aot_compiler.precompile_stdlib()
        except Exception:
            pass

    def compile_and_run(self, text: str) -> object:
        """编译并执行自然语言输入 / Compile and execute natural language input"""
        packet = self._process_with_cache(text)
        pattern, score = self._match_pattern(packet)

        if pattern is not None and score >= 0.70:
            try:
                return self._run_pattern_path(packet, pattern, score)
            except (ValueError, KeyError):
                return self._run_fallback_path(packet)

        return self._run_fallback_path(packet)

    def _process_with_cache(self, text: str):
        """使用embedding缓存处理前端 / Process frontend with embedding cache"""
        if self.embedding_cache is not None:
            original_encode = self._original_encode
            self.frontend.embedder.encode = self.embedding_cache.encode
            try:
                packet = self.frontend.process(text)
            finally:
                self.frontend.embedder.encode = original_encode
            return packet
        return self.frontend.process(text)

    def compile_only(self, text: str) -> list:
        """仅编译不执行 / Compile only without execution"""
        packet = self._process_with_cache(text)
        pattern, score = self._match_pattern(packet)

        if pattern is not None and score >= 0.70:
            try:
                slots = self.filler.fill(pattern, packet)
                return self.instantiator.instantiate(pattern, slots)
            except (ValueError, KeyError):
                return self.decoder.decode(packet)

        return self.decoder.decode(packet)

    def compile_to_callable(self, code: str) -> NLASMFunction:
        """编译.nl代码为可调用对象 / Compile .nl code to callable"""
        ir_nodes, signature = _compile_nlasm_code(code)
        return NLASMFunction(ir_nodes=ir_nodes, signature=signature)

    def _match_pattern(self, packet) -> tuple[IRPattern | None, float]:
        """匹配最佳Pattern / Match best pattern"""
        match_text = packet.semantic_skeleton or packet.normalized
        matches = self.matcher.match(match_text, top_k=3)
        if not matches:
            return None, 0.0

        best_pattern, best_score = matches[0]

        has_filter_intent = self._has_filter_intent(packet)
        if has_filter_intent and best_pattern.name != "filter_sum":
            for pattern, score in matches:
                if pattern.name == "filter_sum":
                    adjusted = score + 0.15
                    if adjusted > best_score:
                        return pattern, adjusted
                    break

        if not has_filter_intent and best_pattern.name == "filter_sum":
            for pattern, score in matches:
                if pattern.name != "filter_sum":
                    if score + 0.05 > best_score:
                        return pattern, score + 0.05
                    break

        return best_pattern, best_score

    def _has_filter_intent(self, packet) -> bool:
        """检测过滤意图 / Detect filter intent"""
        for entity in packet.entities:
            if entity.label == ENTITY_OP:
                return True
        skeleton = packet.semantic_skeleton or packet.normalized
        for kw in FILTER_KEYWORDS:
            if kw in skeleton:
                return True
        return False

    def _run_jit_path(self, ir_nodes: list, pattern: IRPattern | None, slots: dict | None) -> object:
        """JIT编译执行路径 / JIT compilation and execution path"""
        try:
            cfg = self.cfg_builder.build(ir_nodes)
            dom_tree = DominatorTree()
            dom_tree.build(cfg)
            ssa_cfg = self.ssa_builder.build(cfg)

            if pattern and hasattr(pattern, 'constraints') and pattern.constraints:
                self.optimizer.set_pattern_hints(pattern.constraints)
            optimized_cfg = self.optimizer.run(ssa_cfg)

            llvm_module = self.codegen.lower(optimized_cfg)
            compiled_fn = self.jit_executor.compile_module(llvm_module)
            result = self.jit_executor.execute(compiled_fn)

            self._last_compiled_fn = compiled_fn
            return result

        except Exception as e:
            print(f"[警告] JIT 编译失败，回退到解释执行: {e}")
            return self._maybe_interpret(ir_nodes, slots)

    def _run_pattern_path(self, packet, pattern: IRPattern, score: float) -> object:
        """Pattern匹配路径执行 / Execute via Pattern-matched path"""
        slots = self.filler.fill(pattern, packet)
        cache_key = build_cache_key(pattern.name, normalize_type_signature(slots))

        # 检查内联缓存 / Check inline cache
        if self.use_jit:
            compiled_fn = self.cache.lookup(cache_key)
            if compiled_fn is not None:
                try:
                    result = self.jit_executor.execute(compiled_fn)
                    return result
                except Exception:
                    self.cache.cache.pop(cache_key, None)

        ir_nodes = self.instantiator.instantiate(pattern, slots)

        if self.use_jit:
            result = self._run_jit_path(ir_nodes, pattern, slots)
            if hasattr(self, '_last_compiled_fn'):
                self.cache.update(cache_key, self._last_compiled_fn)
                del self._last_compiled_fn
            return result
        else:
            self.cache.update(cache_key, ir_nodes)
            return self._maybe_interpret(ir_nodes, slots)

    def _run_fallback_path(self, packet) -> object:
        """降级路径 / Fallback path"""
        ir_nodes = self.decoder.decode(packet)
        arr_slot = self._extract_arr_from_packet(packet)
        return self._maybe_interpret(ir_nodes, arr_slot=arr_slot)

    def _maybe_interpret(self, ir_nodes: list, slots: dict[str, Any] | None = None, arr_slot: ArraySlot | None = None) -> object:
        """解释执行 / Interpret"""
        if arr_slot is None:
            arr_slot = self._extract_arr_slot(slots)
        interp = self.interpreter_cls(arr_slot=arr_slot)
        interp.run(ir_nodes)
        if interp.outputs:
            return interp.outputs[-1]
        return None

    def _extract_arr_slot(self, slots: dict[str, Any] | None) -> ArraySlot | None:
        if slots:
            for v in slots.values():
                if isinstance(v, ArraySlot):
                    return v
        return None

    def _extract_arr_from_packet(self, packet) -> ArraySlot | None:
        from .entities import ENTITY_ARRAY
        for entity in packet.entities:
            if entity.label == ENTITY_ARRAY:
                return ArraySlot(name="arr", values=entity.value)
        return None

    def get_stats(self) -> dict[str, Any]:
        """获取流水线统计信息 / Get pipeline statistics"""
        stats = {
            "cache": self.cache.stats(),
        }
        if self.embedding_cache:
            stats["embedding_cache"] = self.embedding_cache.stats()
        stats["specialization"] = self.specialization.get_stats()
        if self.parallel_compiler:
            stats["parallel_compiler"] = {
                "pending": self.parallel_compiler.pending_count,
                "completed": self.parallel_compiler.completed_count,
            }
        return stats

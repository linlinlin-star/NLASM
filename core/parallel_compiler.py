from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

from .ir import FuncDef, Stmt
from .cfg_builder import CFGBuilder
from .llvm_codegen import LLVMCodeGen
from .optimizer import Optimizer
from .ssa_builder import SSABuilder


class ParallelCompiler:
    """并行编译器 — 多线程JIT编译 + 后台编译 / Parallel compiler — multi-threaded JIT + background compilation.

    优化点 / Optimizations:
    1. 多线程编译 — 多个函数并行走编译链
    2. 后台编译 — 主线程继续解释执行，后台线程编译热函数
    3. 编译队列 — 优先编译热函数

    1. Multi-threaded compilation — multiple functions compile in parallel
    2. Background compilation — main thread continues interpreting, background compiles hot functions
    3. Compilation queue — prioritize hot functions
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._pending: dict[str, Future] = {}  # 函数名 -> 编译Future / Function name -> compile Future
        self._results: dict[str, Any] = {}  # 函数名 -> 编译结果 / Function name -> compile result
        self._lock = threading.Lock()

    def submit(self, func_name: str, func_def: FuncDef, opt_level: int = 2, callback: Any = None) -> None:
        """提交编译任务 — 非阻塞 / Submit compilation task — non-blocking"""
        with self._lock:
            if func_name in self._pending:
                return  # 已在编译中 / Already compiling

        def _compile():
            try:
                result = self._compile_func(func_def, opt_level)
                with self._lock:
                    self._results[func_name] = result
                    self._pending.pop(func_name, None)
                if callback:
                    callback(func_name, result)
            except Exception:
                with self._lock:
                    self._pending.pop(func_name, None)

        future = self._executor.submit(_compile)
        with self._lock:
            self._pending[func_name] = future

    def submit_batch(self, funcs: list[tuple[str, FuncDef, int]], callback: Any = None) -> None:
        """批量提交编译任务 / Batch submit compilation tasks"""
        for func_name, func_def, opt_level in funcs:
            self.submit(func_name, func_def, opt_level, callback)

    def get_result(self, func_name: str) -> Any | None:
        """获取编译结果 — 非阻塞 / Get compilation result — non-blocking"""
        with self._lock:
            return self._results.get(func_name)

    def wait_for(self, func_name: str, timeout: float | None = None) -> Any | None:
        """等待指定函数编译完成 / Wait for specified function compilation to complete"""
        with self._lock:
            future = self._pending.get(func_name)
            result = self._results.get(func_name)

        if result is not None:
            return result

        if future is not None:
            try:
                future.result(timeout=timeout)
            except Exception:
                pass

        with self._lock:
            return self._results.get(func_name)

    def wait_all(self, timeout: float | None = None) -> None:
        """等待所有编译任务完成 / Wait for all compilation tasks to complete"""
        with self._lock:
            futures = list(self._pending.values())

        for future in futures:
            try:
                future.result(timeout=timeout)
            except Exception:
                pass

    def _compile_func(self, func_def: FuncDef, opt_level: int) -> Any:
        """编译单个函数 — 完整编译链 / Compile single function — full compilation chain"""
        from .jit_executor import JITExecutor

        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build([func_def])

        ssa_builder = SSABuilder()
        ssa_cfg = ssa_builder.build(cfg)

        optimizer = Optimizer()
        optimized_cfg = optimizer.run(ssa_cfg)

        codegen = LLVMCodeGen()
        llvm_module = codegen.lower(optimized_cfg)

        if codegen.unsupported_nodes:
            return None

        jit = JITExecutor(opt_level=opt_level)
        func_ptr = jit.compile_module(llvm_module)
        return func_ptr

    def shutdown(self) -> None:
        """关闭并行编译器 / Shutdown parallel compiler"""
        self._executor.shutdown(wait=False)

    @property
    def pending_count(self) -> int:
        """获取待编译任务数 / Get pending compilation count"""
        with self._lock:
            return len(self._pending)

    @property
    def completed_count(self) -> int:
        """获取已完成编译数 / Get completed compilation count"""
        with self._lock:
            return len(self._results)

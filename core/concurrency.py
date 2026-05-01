from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .ir import FuncDef
from .ir_interpreter import IRInterpreter


class AsyncTask:
    """异步任务包装 / Async task wrapper"""

    def __init__(self, future: Future) -> None:
        self._future = future

    def result(self) -> Any:
        try:
            return self._future.result()
        except Exception as e:
            raise RuntimeError(f"异步任务失败: {e}") from e

    def done(self) -> bool:
        return self._future.done()

    def wait(self) -> Any:
        try:
            return self._future.result()
        except Exception as e:
            raise RuntimeError(f"异步任务失败: {e}") from e

    def cancel(self) -> bool:
        return self._future.cancel()


class NLASMConcurrency:
    """NLASM并发执行器 / NLASM concurrency executor.

    基于ThreadPoolExecutor实现，每个线程拥有独立的IRInterpreter副本和符号表快照。
    Based on ThreadPoolExecutor, each thread has its own IRInterpreter copy and symbol table snapshot.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

    def run_async(self, func_name: str, args: list, kwargs: dict, interp: IRInterpreter) -> AsyncTask:
        """异步执行函数，立即返回AsyncTask / Execute function asynchronously, return AsyncTask immediately"""
        with self._lock:
            func_def = interp._functions.get(func_name)
            symtab_snapshot = interp.symtab.snapshot()
            functions_copy = dict(interp._functions)

        if func_def is not None:
            future = self._executor.submit(self._run_func, func_def, args, kwargs, interp.bridge, symtab_snapshot, functions_copy)
            return AsyncTask(future)

        from .builtins import BUILTINS
        builtin = BUILTINS.functions.get(func_name)
        if builtin is not None:
            future = self._executor.submit(self._run_builtin, builtin, args, kwargs)
            return AsyncTask(future)

        raise NameError(f"未定义函数: {func_name}")

    def run_parallel(self, calls: list[tuple[str, list, dict]], interp: IRInterpreter) -> list[Any]:
        """并行执行多个函数，阻塞等待所有结果 / Execute multiple functions in parallel, block until all results"""
        with self._lock:
            symtab_snapshot = interp.symtab.snapshot()
            functions_copy = dict(interp._functions)

        futures: list[Future] = []
        for func_name, args, kwargs in calls:
            func_def = functions_copy.get(func_name)
            if func_def is not None:
                future = self._executor.submit(self._run_func, func_def, args, kwargs, interp.bridge, symtab_snapshot, functions_copy)
            else:
                from .builtins import BUILTINS
                builtin = BUILTINS.functions.get(func_name)
                if builtin is not None:
                    future = self._executor.submit(self._run_builtin, builtin, args, kwargs)
                else:
                    raise NameError(f"未定义函数: {func_name}")
            futures.append(future)
        return [f.result() for f in futures]

    def _run_func(self, func_def: FuncDef, args: list, kwargs: dict, bridge: Any, symtab_snapshot: Any, functions_copy: dict[str, FuncDef]) -> Any:
        """在线程池中执行函数 / Execute function in thread pool"""
        from .ir_interpreter import _ReturnSentinel
        child = IRInterpreter(bridge=bridge)
        child._functions = functions_copy
        child.symtab = symtab_snapshot.snapshot()  # 创建符号表副本，避免线程间冲突
        for i, (pname, _ptype) in enumerate(func_def.params):
            if i < len(args):
                child.symtab.define(pname, args[i])
        for k, v in kwargs.items():
            child.symtab.define(k, v)
        result = child.run(func_def.body)
        if isinstance(result, _ReturnSentinel):
            return result.value
        return result

    def _run_builtin(self, builtin: Any, args: list, kwargs: dict) -> Any:
        """在线程池中执行内置函数 / Execute builtin in thread pool"""
        return builtin(*args, **kwargs)

    def shutdown(self) -> None:
        """关闭并发执行器，等待所有任务完成 / Shutdown concurrency executor, wait for all tasks to complete"""
        self._executor.shutdown(wait=True)

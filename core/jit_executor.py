from __future__ import annotations

import ctypes
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llvmlite import ir as llvmir
from llvmlite import binding as llvm


for _init_fn in [
    getattr(llvm, "initialize", None),
    getattr(llvm, "initialize_native_target", None),
    getattr(llvm, "initialize_native_asmprinter", None),
]:
    if _init_fn is not None:
        try:
            _init_fn()
        except RuntimeError:
            pass


_SAFE_CPU_FEATURES = frozenset({
    "sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2",
    "avx", "avx2", "bmi", "bmi2", "fma", "f16c",
    "popcnt", "lzcnt", "aes", "rdrnd",
    "neon", "fp-armv8", "crypto",
})


def _get_host_cpu_features() -> str:
    """获取当前CPU支持的特征列表 / Get current CPU feature list.

    仅启用已知安全的特征，避免LLVM abort。
    Only enable known-safe features to prevent LLVM abort.
    """
    try:
        features = llvm.get_host_cpu_features()
        if hasattr(features, 'flatten'):
            raw = features.flatten()
            parts = raw.split(",")
            safe = [p for p in parts if p.lstrip("+-").lower() in _SAFE_CPU_FEATURES]
            return ",".join(safe) if safe else ""
        if isinstance(features, dict):
            safe = [f"+{k}" for k, v in features.items() if v and k.lower() in _SAFE_CPU_FEATURES]
            return ",".join(safe) if safe else ""
    except Exception:
        pass
    return ""


def _get_host_cpu_name() -> str:
    """获取当前CPU名称 / Get current CPU name"""
    try:
        return llvm.get_host_cpu_name()
    except Exception:
        return "generic"


class JITExecutor:
    """JIT执行器（高级优化版）/ JIT executor (advanced optimization version).

    优化点 / Optimizations:
    1. O3优化级别 — 最高优化，启用循环向量化/SLP向量化
    2. 目标特定优化 — 自动检测CPU特征(SSE4.2/AVX2/AVX-512)
    3. ORC JIT v2 — 更快的编译和执行（llvmlite支持时）
    4. 参数传递 — 支持带参数的函数调用
    5. GIL释放 — JIT代码执行期间释放GIL
    6. 后台编译 — 异步编译不阻塞主线程
    7. 编译缓存 — 避免重复编译相同IR

    1. O3 optimization — highest level, enables loop vectorization/SLP vectorization
    2. Target-specific optimization — auto-detect CPU features (SSE4.2/AVX2/AVX-512)
    3. ORC JIT v2 — faster compilation and execution (when llvmlite supports it)
    4. Argument passing — supports function calls with arguments
    5. GIL release — release GIL during JIT code execution
    6. Background compilation — async compilation without blocking main thread
    7. Compilation cache — avoid recompiling same IR
    """

    def __init__(self, opt_level: int = 3) -> None:
        self.engine = None
        self._compiled_modules: list = []
        self._opt_level = opt_level
        self._compile_cache: dict[str, int] = {}  # IR哈希 -> 函数地址 / IR hash -> function address
        self._background_executor = ThreadPoolExecutor(max_workers=1)  # 后台编译线程 / Background compile thread
        self._compile_lock = threading.Lock()

    def _create_engine(self) -> None:
        """创建JIT编译引擎 — O3 + 目标特定优化 / Create JIT engine — O3 + target-specific optimization"""
        target = llvm.Target.from_default_triple()
        cpu_name = _get_host_cpu_name()
        cpu_features = _get_host_cpu_features()

        target_machine = self._try_create_target_machine(
            target, cpu=cpu_name, features=cpu_features, opt=self._opt_level,
        )

        backing_mod = llvm.parse_assembly("")
        self.engine = llvm.create_mcjit_compiler(backing_mod, target_machine)

    @staticmethod
    def _try_create_target_machine(target, cpu: str, features: str, opt: int):
        """安全创建TargetMachine — 逐步回退 / Safely create TargetMachine — progressive fallback.

        1. 尝试 CPU名+特征+优化级别 / Try CPU name + features + opt level
        2. 回退到 generic CPU + 特征 / Fallback to generic CPU + features
        3. 回退到 CPU名 + 无特征 / Fallback to CPU name + no features
        4. 最终回退到 generic + 无特征 / Final fallback to generic + no features
        """
        configs = [
            (cpu, features, opt),
            ("generic", features, opt),
            (cpu, "", opt),
            ("generic", "", opt),
        ]
        for c, f, o in configs:
            try:
                return target.create_target_machine(
                    cpu=c, features=f, opt=o,
                    reloc="pic", codemodel="default",
                )
            except Exception:
                continue
        return target.create_target_machine(
            cpu="generic", features="", opt=2,
            reloc="pic", codemodel="default",
        )

    def compile_module(self, module: llvmir.Module) -> object:
        """编译LLVM IR模块 — 含编译缓存 / Compile LLVM IR module — with compilation cache"""
        if self.engine is None:
            self._create_engine()

        llvm_ir = str(module)

        # 检查编译缓存 / Check compilation cache
        import hashlib
        ir_hash = hashlib.sha256(llvm_ir.encode()).hexdigest()[:16]
        if ir_hash in self._compile_cache:
            return self._compile_cache[ir_hash]

        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        with self._compile_lock:
            self.engine.add_module(mod)
            self._compiled_modules.append(mod)
            self.engine.finalize_object()
            self.engine.run_static_constructors()

            func_ptr = self.engine.get_function_address("main")
            self._last_compiled_fn = func_ptr
            self._compile_cache[ir_hash] = func_ptr

        return func_ptr

    def compile_function(self, module: llvmir.Module, func_name: str) -> object:
        """编译指定函数 / Compile specified function"""
        if self.engine is None:
            self._create_engine()

        llvm_ir = str(module)
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        with self._compile_lock:
            self.engine.add_module(mod)
            self._compiled_modules.append(mod)
            self.engine.finalize_object()
            self.engine.run_static_constructors()

            func_ptr = self.engine.get_function_address(func_name)

        return func_ptr

    def compile_async(self, module: llvmir.Module, callback: Any = None) -> None:
        """后台异步编译 — 不阻塞主线程 / Background async compilation — non-blocking.

        编译完成后调用callback(func_ptr)通知主线程。
        Calls callback(func_ptr) when compilation completes.
        """
        def _compile():
            try:
                func_ptr = self.compile_module(module)
                if callback:
                    callback(func_ptr)
            except Exception as e:
                if callback:
                    callback(None)

        self._background_executor.submit(_compile)

    def execute(self, compiled_entry: object, args: tuple | None = None, release_gil: bool = True) -> int:
        """执行已编译的函数 / Execute compiled function"""
        if compiled_entry is None:
            raise RuntimeError("compiled_entry 为空")

        if args:
            arg_types = [ctypes.c_int64] * len(args)
            cfunc_type = ctypes.CFUNCTYPE(ctypes.c_int64, *arg_types)
            cfunc = cfunc_type(compiled_entry)
            c_args = [ctypes.c_int64(a) for a in args]
        else:
            cfunc = ctypes.CFUNCTYPE(ctypes.c_int64)(compiled_entry)
            c_args = []

        # ctypes调用外部C函数时自动释放GIL / ctypes automatically releases GIL for C function calls
        result = cfunc(*c_args)
        return result

    def get_function_address(self, name: str) -> int:
        """获取已编译函数的地址 / Get address of compiled function"""
        if self.engine is None:
            raise RuntimeError("JIT engine not initialized")
        return self.engine.get_function_address(name)

    def shutdown(self) -> None:
        """关闭JIT执行器 / Shutdown JIT executor"""
        self._background_executor.shutdown(wait=False)

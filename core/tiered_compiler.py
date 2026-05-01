from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from .ir import FuncDef, Stmt
from .ir_interpreter import IRInterpreter
from .python_bridge import PythonBridge


# 热点阈值 — 函数执行超过此次数触发JIT编译 / Hot threshold — function execution beyond this count triggers JIT compilation
HOT_THRESHOLD = 100

# 极热阈值 — 超过此次数启用更激进的优化 / Very hot threshold — beyond this count enables more aggressive optimization
VERY_HOT_THRESHOLD = 1000


@dataclass
class FunctionProfile:
    """函数执行计数器 — 用于热点检测 / Function execution counter — for hot path detection"""
    name: str
    call_count: int = 0
    is_compiled: bool = False
    compiled_fn: Any = None
    opt_level: int = 0  # 0=未编译, 1=O1, 2=O2, 3=O3 / 0=not compiled, 1=O1, 2=O2, 3=O3
    type_feedback: dict[str, int] = field(default_factory=dict)  # 类型反馈 / Type feedback


class TieredCompiler:
    """分层编译器 — 解释器快速启动 + 热点JIT编译 / Tiered compiler — interpreter for fast startup + JIT for hot paths.

    分层策略 / Tiering strategy:
    - Tier 0 (解释执行): 所有函数首次执行走解释器，快速启动零延迟
    - Tier 1 (O1 JIT): 函数调用超过 HOT_THRESHOLD 次后，后台JIT编译O1版本
    - Tier 2 (O3 JIT): 函数调用超过 VERY_HOT_THRESHOLD 次后，重新编译O3版本

    - Tier 0 (interpreted): All functions start in interpreter, zero startup latency
    - Tier 1 (O1 JIT): After HOT_THRESHOLD calls, background JIT compile with O1
    - Tier 2 (O3 JIT): After VERY_HOT_THRESHOLD calls, recompile with O3
    """

    def __init__(self, interp: IRInterpreter, use_jit: bool = True) -> None:
        self.interp = interp
        self.use_jit = use_jit
        self._profiles: dict[str, FunctionProfile] = {}  # 函数名 -> 执行计数 / Function name -> execution count
        self._lock = threading.Lock()
        self._jit_available = False

        # 检测JIT是否可用 / Check if JIT is available
        if use_jit:
            try:
                from .jit_executor import JITExecutor
                self._jit = JITExecutor(opt_level=1)
                self._jit_available = True
            except Exception:
                self._jit_available = False

    def execute(self, func_name: str, func_def: FuncDef, args: list, kwargs: dict) -> Any:
        """执行函数 — 根据热度选择执行层级 / Execute function — select tier based on hotness"""
        profile = self._get_or_create_profile(func_name)

        # 已编译且可用 — 直接执行JIT代码 / Already compiled and available — execute JIT code directly
        if profile.is_compiled and profile.compiled_fn is not None:
            try:
                return self._jit.execute(profile.compiled_fn, args=tuple(args))
            except Exception:
                profile.is_compiled = False
                profile.compiled_fn = None

        # 解释执行 / Interpret
        result = self.interp._call_func_with_frame(func_def, args, kwargs)

        # 更新调用计数 / Update call count
        with self._lock:
            profile.call_count += 1

            # 记录参数类型反馈 / Record argument type feedback
            for i, arg in enumerate(args):
                type_name = type(arg).__name__
                key = f"arg{i}:{type_name}"
                profile.type_feedback[key] = profile.type_feedback.get(key, 0) + 1

            # 热点检测 — 触发后台JIT编译 / Hot path detection — trigger background JIT compilation
            if self._jit_available and not profile.is_compiled:
                if profile.call_count >= HOT_THRESHOLD:
                    self._schedule_compilation(func_name, func_def, profile, opt_level=1)
                elif profile.call_count >= VERY_HOT_THRESHOLD and profile.opt_level < 3:
                    self._schedule_compilation(func_name, func_def, profile, opt_level=3)

        return result

    def _get_or_create_profile(self, func_name: str) -> FunctionProfile:
        """获取或创建函数执行配置 / Get or create function execution profile"""
        if func_name not in self._profiles:
            self._profiles[func_name] = FunctionProfile(name=func_name)
        return self._profiles[func_name]

    def _schedule_compilation(self, func_name: str, func_def: FuncDef, profile: FunctionProfile, opt_level: int) -> None:
        """调度后台JIT编译 / Schedule background JIT compilation"""
        try:
            from .cfg_builder import CFGBuilder
            from .llvm_codegen import LLVMCodeGen
            from .optimizer import Optimizer
            from .ssa_builder import SSABuilder

            # IR -> CFG -> SSA -> 优化 -> LLVM IR / IR -> CFG -> SSA -> Optimize -> LLVM IR
            cfg_builder = CFGBuilder()
            cfg = cfg_builder.build([func_def])

            ssa_builder = SSABuilder()
            ssa_cfg = ssa_builder.build(cfg)

            optimizer = Optimizer()
            optimized_cfg = optimizer.run(ssa_cfg)

            codegen = LLVMCodeGen()
            llvm_module = codegen.lower(optimized_cfg)

            if codegen.unsupported_nodes:
                return  # 有不支持的节点，不编译 / Has unsupported nodes, skip compilation

            # 后台编译 / Background compilation
            def on_compiled(func_ptr):
                if func_ptr is not None:
                    with self._lock:
                        profile.is_compiled = True
                        profile.compiled_fn = func_ptr
                        profile.opt_level = opt_level

            self._jit.compile_async(llvm_module, callback=on_compiled)

        except Exception:
            pass  # 编译失败，继续解释执行 / Compilation failed, continue interpreting

    def get_profiles(self) -> dict[str, dict[str, Any]]:
        """获取所有函数的执行配置 / Get execution profiles for all functions"""
        result = {}
        for name, profile in self._profiles.items():
            result[name] = {
                "call_count": profile.call_count,
                "is_compiled": profile.is_compiled,
                "opt_level": profile.opt_level,
                "type_feedback": profile.type_feedback,
            }
        return result

    def force_compile(self, func_name: str, func_def: FuncDef, opt_level: int = 3) -> bool:
        """强制编译指定函数 — 用于AOT预热 / Force compile specified function — for AOT warmup"""
        profile = self._get_or_create_profile(func_name)
        try:
            from .cfg_builder import CFGBuilder
            from .llvm_codegen import LLVMCodeGen
            from .optimizer import Optimizer
            from .ssa_builder import SSABuilder

            cfg_builder = CFGBuilder()
            cfg = cfg_builder.build([func_def])
            ssa_builder = SSABuilder()
            ssa_cfg = ssa_builder.build(cfg)
            optimizer = Optimizer()
            optimized_cfg = optimizer.run(ssa_cfg)
            codegen = LLVMCodeGen()
            llvm_module = codegen.lower(optimized_cfg)

            if codegen.unsupported_nodes:
                return False

            func_ptr = self._jit.compile_module(llvm_module)
            if func_ptr is not None:
                profile.is_compiled = True
                profile.compiled_fn = func_ptr
                profile.opt_level = opt_level
                return True
        except Exception:
            pass
        return False

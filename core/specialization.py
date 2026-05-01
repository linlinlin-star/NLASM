from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ir import (
    Add,
    Assign,
    Cmp,
    Div,
    FuncDef,
    If,
    Literal,
    Loop,
    Mod,
    Mul,
    Print,
    Return,
    Sub,
    Var,
)
from .cfg import CFG
from .cfg_builder import CFGBuilder
from .llvm_codegen import LLVMCodeGen
from .optimizer import Optimizer
from .ssa_builder import SSABuilder


@dataclass
class TypeFeedback:
    """类型反馈 — 记录函数参数的运行时类型 / Type feedback — records runtime types of function arguments"""
    arg_types: dict[int, str] = field(default_factory=dict)  # 参数索引 -> 最常见类型 / Arg index -> most common type
    call_count: int = 0


class SpecializationEngine:
    """专用化引擎 — 为常见类型组合生成专用代码 / Specialization engine — generate specialized code for common type combinations.

    策略 / Strategy:
    1. 收集类型反馈 — 记录每次调用的参数类型
    2. 识别热类型组合 — 超过阈值的类型组合触发专用化
    3. 生成专用IR — 将通用IR中的类型替换为具体类型
    4. 编译专用版本 — 生成更优化的机器码

    1. Collect type feedback — record argument types on each call
    2. Identify hot type combinations — combinations exceeding threshold trigger specialization
    3. Generate specialized IR — replace generic types with concrete types in IR
    4. Compile specialized version — generate more optimized machine code
    """

    SPECIALIZATION_THRESHOLD = 50  # 触发专用化的调用次数阈值 / Call count threshold for specialization

    def __init__(self) -> None:
        self._feedback: dict[str, TypeFeedback] = {}  # 函数名 -> 类型反馈 / Function name -> type feedback
        self._specialized: dict[str, dict[str, Any]] = {}  # 函数名 -> {类型签名 -> 编译结果} / Function name -> {type sig -> compiled result}

    def record_call(self, func_name: str, args: list) -> None:
        """记录函数调用的参数类型 / Record argument types for function call"""
        if func_name not in self._feedback:
            self._feedback[func_name] = TypeFeedback()

        fb = self._feedback[func_name]
        fb.call_count += 1
        for i, arg in enumerate(args):
            type_name = type(arg).__name__
            if i not in fb.arg_types:
                fb.arg_types[i] = type_name
            # 简化: 只记录最常见类型 / Simplified: only record most common type

    def should_specialize(self, func_name: str) -> bool:
        """检查函数是否应该专用化 / Check if function should be specialized"""
        fb = self._feedback.get(func_name)
        if fb is None:
            return False
        return fb.call_count >= self.SPECIALIZATION_THRESHOLD

    def get_type_signature(self, func_name: str) -> str:
        """获取函数的类型签名 / Get function's type signature"""
        fb = self._feedback.get(func_name)
        if fb is None:
            return "generic"
        parts = [fb.arg_types.get(i, "any") for i in sorted(fb.arg_types.keys())]
        return "_".join(parts)

    def specialize_ir(self, func_def: FuncDef, type_sig: str) -> list:
        """生成专用化IR — 根据类型签名替换通用操作 / Generate specialized IR — replace generic ops based on type signature"""
        specialized_name = f"{func_def.name}__{type_sig}"
        specialized_body = []

        for stmt in func_def.body:
            spec_stmt = self._specialize_stmt(stmt, type_sig)
            specialized_body.append(spec_stmt)

        return [
            FuncDef(
                name=specialized_name,
                params=func_def.params,
                body=specialized_body,
                return_type=func_def.return_type,
            )
        ]

    def _specialize_stmt(self, stmt, type_sig: str):
        """专用化语句 / Specialize statement"""
        results: list = []
        work_stack: list[tuple] = [(stmt, False)]

        while work_stack:
            current, processed = work_stack.pop()

            if isinstance(current, If):
                if processed:
                    orelse = results[-len(current.orelse):]
                    del results[-len(current.orelse):]
                    body = results[-len(current.body):]
                    del results[-len(current.body):]
                    results.append(If(
                        condition=self._specialize_expr(current.condition, type_sig),
                        body=body,
                        orelse=orelse,
                    ))
                else:
                    work_stack.append((current, True))
                    for s in reversed(current.orelse):
                        work_stack.append((s, False))
                    for s in reversed(current.body):
                        work_stack.append((s, False))
                continue

            if isinstance(current, Assign):
                results.append(Assign(
                    target=current.target,
                    value=self._specialize_expr(current.value, type_sig),
                ))
            elif isinstance(current, Return) and current.value:
                results.append(Return(value=self._specialize_expr(current.value, type_sig)))
            elif isinstance(current, Print):
                results.append(Print(value=self._specialize_expr(current.value, type_sig)))
            else:
                results.append(current)

        return results[0] if results else stmt

    def _specialize_expr(self, expr, type_sig: str):
        """专用化表达式 — 根据类型选择更优指令 / Specialize expression — choose better instructions based on type"""
        # int专用: 整数除法用sdiv替代通用除法 / int specialized: use sdiv instead of generic div
        # float专用: 使用fadd/fsub/fmul/fdiv / float specialized: use fadd/fsub/fmul/fdiv
        # 当前为占位实现，直接返回原表达式 / Currently placeholder, return original expression
        return expr

    def compile_specialized(self, func_name: str, func_def: FuncDef) -> Any:
        """编译专用化版本 / Compile specialized version"""
        type_sig = self.get_type_signature(func_name)
        specialized_ir = self.specialize_ir(func_def, type_sig)

        try:
            cfg_builder = CFGBuilder()
            cfg = cfg_builder.build(specialized_ir)

            ssa_builder = SSABuilder()
            ssa_cfg = ssa_builder.build(cfg)

            optimizer = Optimizer()
            optimized_cfg = optimizer.run(ssa_cfg)

            codegen = LLVMCodeGen()
            llvm_module = codegen.lower(optimized_cfg)

            if codegen.unsupported_nodes:
                return None

            from .jit_executor import JITExecutor
            jit = JITExecutor(opt_level=3)
            func_ptr = jit.compile_module(llvm_module)

            if func_name not in self._specialized:
                self._specialized[func_name] = {}
            self._specialized[func_name][type_sig] = func_ptr

            return func_ptr
        except Exception:
            return None

    def get_specialized(self, func_name: str, args: list) -> Any | None:
        """获取专用化版本的函数指针 / Get specialized version function pointer"""
        type_sig = self.get_type_signature(func_name)
        func_specializations = self._specialized.get(func_name, {})
        return func_specializations.get(type_sig)

    def get_stats(self) -> dict[str, Any]:
        """获取专用化统计 / Get specialization statistics"""
        return {
            "tracked_functions": len(self._feedback),
            "specialized_functions": len(self._specialized),
            "feedback": {
                name: {"call_count": fb.call_count, "arg_types": fb.arg_types}
                for name, fb in self._feedback.items()
            },
        }


class AOTCompiler:
    """AOT编译器 — 预编译标准库和生成独立可执行文件 / AOT compiler — precompile stdlib and generate standalone executables.

    功能 / Features:
    1. 预编译标准库 — 将.nl标准库编译为LLVM IR，启动时直接加载
    2. 生成独立可执行文件 — 将NLASM程序编译为原生可执行文件
    3. 缓存编译结果 — 避免重复编译

    1. Precompile stdlib — compile .nl stdlib to LLVM IR, load directly at startup
    2. Generate standalone executables — compile NLASM programs to native executables
    3. Cache compilation results — avoid recompilation
    """

    def __init__(self, stdlib_path: str = "./stdlib") -> None:
        self.stdlib_path = stdlib_path
        self._compiled_cache: dict[str, str] = {}  # 模块名 -> LLVM IR / Module name -> LLVM IR

    def precompile_stdlib(self) -> dict[str, str]:
        """预编译标准库 — 将所有.nl标准库文件编译为LLVM IR / Precompile stdlib — compile all .nl stdlib files to LLVM IR"""
        from pathlib import Path
        from .file_parser import NLFileParser

        stdlib_dir = Path(self.stdlib_path)
        if not stdlib_dir.exists():
            return {}

        results = {}
        for nl_file in stdlib_dir.glob("*.nl"):
            module_name = nl_file.stem
            try:
                parser = NLFileParser()
                stmts = parser.parse_file(str(nl_file))

                cfg_builder = CFGBuilder()
                cfg = cfg_builder.build(stmts)

                ssa_builder = SSABuilder()
                ssa_cfg = ssa_builder.build(cfg)

                optimizer = Optimizer()
                optimized_cfg = optimizer.run(ssa_cfg)

                codegen = LLVMCodeGen()
                llvm_module = codegen.lower(optimized_cfg)
                llvm_ir = str(llvm_module)

                self._compiled_cache[module_name] = llvm_ir
                results[module_name] = llvm_ir
            except Exception as e:
                results[module_name] = f"ERROR: {e}"

        return results

    def compile_to_executable(self, filepath: str, output_path: str | None = None) -> str:
        """编译.nl文件为独立可执行文件 / Compile .nl file to standalone executable"""
        from pathlib import Path
        from .file_parser import NLFileParser

        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        parser = NLFileParser()
        stmts = parser.parse_file(filepath)

        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build(stmts)
        ssa_builder = SSABuilder()
        ssa_cfg = ssa_builder.build(cfg)
        optimizer = Optimizer()
        optimized_cfg = optimizer.run(ssa_cfg)
        codegen = LLVMCodeGen()
        llvm_module = codegen.lower(optimized_cfg)

        if output_path is None:
            output_path = str(path.with_suffix('.ll'))

        Path(output_path).write_text(str(llvm_module), encoding="utf-8")
        return output_path

    def get_cached_ir(self, module_name: str) -> str | None:
        """获取缓存的LLVM IR / Get cached LLVM IR"""
        return self._compiled_cache.get(module_name)

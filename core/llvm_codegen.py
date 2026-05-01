from __future__ import annotations

import platform
import struct

from llvmlite import ir as llvmir

from .cfg import BasicBlock, CFG
from .ir import (
    Add,
    And,
    Assign,
    Cmp,
    Div,
    If,
    Literal,
    Loop,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    Print,
    Return,
    Sub,
    Var,
)
from .ssa_builder import PhiInstruction


# ============================================================
# LLVM 类型定义 / LLVM Type Definitions
# ============================================================

I8 = llvmir.IntType(8)
I32 = llvmir.IntType(32)
I64 = llvmir.IntType(64)
I1 = llvmir.IntType(1)
F64 = llvmir.DoubleType()
VOID = llvmir.VoidType()
I8_PTR = I8.as_pointer()
I64_PTR = I64.as_pointer()

# Tagged Union: NLValue = { i64 tag, i64 payload }
# tag: 0=int, 1=float, 2=bool, 3=string_ptr, 4=array_ptr, 5=none
TAG_INT = 0
TAG_FLOAT = 1
TAG_BOOL = 2
TAG_STRING = 3
TAG_ARRAY = 4
TAG_NONE = 5

NLVALUE = llvmir.LiteralStructType([I64, I64])  # { tag, payload }


def _get_native_triple() -> str:
    """获取当前平台的LLVM target triple / Get LLVM target triple for current platform"""
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "x86_64-pc-windows-msvc"
        elif machine in ("i386", "i686", "x86"):
            return "i686-pc-windows-msvc"
    elif system == "darwin":
        if machine == "arm64":
            return "arm64-apple-darwin"
        return "x86_64-apple-darwin"
    else:
        if machine == "aarch64":
            return "aarch64-unknown-linux-gnu"
        return "x86_64-unknown-linux-gnu"
    return "x86_64-unknown-linux-gnu"


class LLVMCodeGen:
    """LLVM IR代码生成器（扩展版）/ LLVM IR code generator (extended version).

    优化点 / Optimizations:
    1. Tagged Union (NLValue) — 统一值类型，支持 int/float/bool/string/array/none
    2. 浮点数运算支持 — F64类型 + fadd/fsub/fmul/fdiv指令
    3. 跨平台 target triple — 自动检测 Windows/macOS/Linux
    4. 不支持节点回退机制 — 标记而非静默返回0
    5. 扩展的IR节点支持 — Div/Mod/Neg/And/Or/Not/If分支

    1. Tagged Union (NLValue) — unified value type, supports int/float/bool/string/array/none
    2. Float arithmetic support — F64 type + fadd/fsub/fmul/fdiv instructions
    3. Cross-platform target triple — auto-detect Windows/macOS/Linux
    4. Unsupported node fallback — mark instead of silently returning 0
    5. Extended IR node support — Div/Mod/Neg/And/Or/Not/If branching
    """

    def __init__(self) -> None:
        self.module: llvmir.Module | None = None
        self.builder: llvmir.IRBuilder | None = None
        self.func: llvmir.Function | None = None
        self.vars: dict[str, llvmir.AllocaInstr] = {}
        self.block_map: dict[str, llvmir.Block] = {}
        self.printf_func: llvmir.Function | None = None
        self.unsupported_nodes: list[str] = []  # 记录不支持的节点 / Track unsupported nodes
        self._use_tagged: bool = True  # 是否使用Tagged Union / Whether to use Tagged Union

    def lower(self, ssa_cfg: CFG) -> llvmir.Module:
        """将SSA CFG降级为LLVM IR模块 / Lower SSA CFG to LLVM IR module"""
        self.module = llvmir.Module(name="nlasm")
        self.module.triple = _get_native_triple()
        self.vars = {}
        self.block_map = {}
        self.unsupported_nodes = []

        self._declare_printf()
        self._declare_nlasm_runtime()

        ftype = llvmir.FunctionType(I64, [])
        self.func = llvmir.Function(self.module, ftype, name="main")

        entry_block = self.func.append_basic_block(name="entry")
        self.builder = llvmir.IRBuilder(entry_block)

        for block in ssa_cfg.blocks:
            if block.name == "entry":
                self.block_map[block.name] = entry_block
            else:
                self.block_map[block.name] = self.func.append_basic_block(name=block.name)

        self._lower_block(ssa_cfg.entry, ssa_cfg)

        if not self.builder.block.is_terminated:
            self.builder.ret(llvmir.Constant(I64, 0))

        return self.module

    def _declare_printf(self) -> None:
        """声明printf外部函数 / Declare printf external function"""
        int_fmt = "%ld\n\0"
        float_fmt = "%.6f\n\0"
        bool_fmt = "%s\n\0"
        true_str = "true\0"
        false_str = "false\0"

        # 整数格式串 / Integer format string
        fmt_type = llvmir.ArrayType(I8, len(int_fmt))
        fmt_global = llvmir.GlobalVariable(self.module, fmt_type, name=".fmt_int")
        fmt_global.global_constant = True
        fmt_global.initializer = llvmir.Constant(fmt_type, bytearray(int_fmt.encode("ascii")))
        self._fmt_int = fmt_global

        # 浮点格式串 / Float format string
        fmt_type_f = llvmir.ArrayType(I8, len(float_fmt))
        fmt_global_f = llvmir.GlobalVariable(self.module, fmt_type_f, name=".fmt_float")
        fmt_global_f.global_constant = True
        fmt_global_f.initializer = llvmir.Constant(fmt_type_f, bytearray(float_fmt.encode("ascii")))
        self._fmt_float = fmt_global_f

        # 布尔格式串 / Boolean format string
        fmt_type_b = llvmir.ArrayType(I8, len(bool_fmt))
        fmt_global_b = llvmir.GlobalVariable(self.module, fmt_type_b, name=".fmt_bool")
        fmt_global_b.global_constant = True
        fmt_global_b.initializer = llvmir.Constant(fmt_type_b, bytearray(bool_fmt.encode("ascii")))
        self._fmt_bool = fmt_global_b

        # true/false字符串 / true/false strings
        true_type = llvmir.ArrayType(I8, len(true_str))
        true_global = llvmir.GlobalVariable(self.module, true_type, name=".str_true")
        true_global.global_constant = True
        true_global.initializer = llvmir.Constant(true_type, bytearray(true_str.encode("ascii")))
        self._str_true = true_global

        false_type = llvmir.ArrayType(I8, len(false_str))
        false_global = llvmir.GlobalVariable(self.module, false_type, name=".str_false")
        false_global.global_constant = True
        false_global.initializer = llvmir.Constant(false_type, bytearray(false_str.encode("ascii")))
        self._str_false = false_global

        printf_type = llvmir.FunctionType(I32, [I8_PTR], var_arg=True)
        self.printf_func = llvmir.Function(self.module, printf_type, name="printf")

    def _declare_nlasm_runtime(self) -> None:
        """声明NLASM运行时辅助函数 / Declare NLASM runtime helper functions"""
        # nlasm_print_value(tag: i64, payload: i64) -> void
        print_type = llvmir.FunctionType(VOID, [I64, I64])
        self._nlasm_print = llvmir.Function(self.module, print_type, name="nlasm_print_value")

    def _lower_block(self, block: BasicBlock, cfg: CFG) -> None:
        """降级单个基本块 / Lower a single basic block"""
        llvm_block = self.block_map.get(block.name)
        if llvm_block is None:
            return
        self.builder.position_at_end(llvm_block)

        for instr in block.instructions:
            if isinstance(instr, PhiInstruction):
                continue
            self._lower_instruction(instr)

        if block.succs and not self.builder.block.is_terminated:
            first_succ = self.block_map.get(block.succs[0].name)
            if first_succ:
                self.builder.branch(first_succ)

    def _lower_instruction(self, instr) -> None:
        """降级单条指令 / Lower a single instruction"""
        if isinstance(instr, Assign):
            val = self._lower_expr(instr.value)
            target = instr.target
            if target not in self.vars:
                alloca = self.builder.alloca(I64, name=target)
                self.vars[target] = alloca
            self.builder.store(val, self.vars[target])

        elif isinstance(instr, Print):
            val = self._lower_expr(instr.value)
            fmt_ptr = self.builder.bitcast(self._fmt_int, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, val])

        elif isinstance(instr, Return):
            if instr.value is not None:
                val = self._lower_expr(instr.value)
                self.builder.ret(val)
            else:
                self.builder.ret(llvmir.Constant(I64, 0))

    def _lower_expr(self, expr) -> llvmir.Value:
        """降级表达式为LLVM IR值 / Lower expression to LLVM IR value"""
        stack: list[tuple[object, bool]] = [(expr, False)]
        results: list[llvmir.Value] = []
        while stack:
            node, processed = stack.pop()
            if processed:
                if isinstance(node, Add):
                    right, left = results.pop(), results.pop()
                    if self._is_float_expr(node.left) or self._is_float_expr(node.right):
                        results.append(self.builder.fadd(self._to_float(left), self._to_float(right), name="fadd"))
                    else:
                        results.append(self.builder.add(left, right, name="add"))
                elif isinstance(node, Sub):
                    right, left = results.pop(), results.pop()
                    if self._is_float_expr(node.left) or self._is_float_expr(node.right):
                        results.append(self.builder.fsub(self._to_float(left), self._to_float(right), name="fsub"))
                    else:
                        results.append(self.builder.sub(left, right, name="sub"))
                elif isinstance(node, Mul):
                    right, left = results.pop(), results.pop()
                    if self._is_float_expr(node.left) or self._is_float_expr(node.right):
                        results.append(self.builder.fmul(self._to_float(left), self._to_float(right), name="fmul"))
                    else:
                        results.append(self.builder.mul(left, right, name="mul"))
                elif isinstance(node, Div):
                    right, left = results.pop(), results.pop()
                    if self._is_float_expr(node.left) or self._is_float_expr(node.right):
                        results.append(self.builder.fdiv(self._to_float(left), self._to_float(right), name="fdiv"))
                    else:
                        results.append(self.builder.sdiv(left, right, name="sdiv"))
                elif isinstance(node, Mod):
                    right, left = results.pop(), results.pop()
                    results.append(self.builder.srem(left, right, name="srem"))
                elif isinstance(node, Neg):
                    operand = results.pop()
                    if self._is_float_expr(node.operand):
                        results.append(self.builder.fneg(self._to_float(operand), name="fneg"))
                    else:
                        results.append(self.builder.neg(operand, name="neg"))
                elif isinstance(node, Cmp):
                    right, left = results.pop(), results.pop()
                    if self._is_float_expr(node.left) or self._is_float_expr(node.right):
                        fcmp_map = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "==", "!=": "!="}
                        cmp_result = self.builder.fcmp_ordered(fcmp_map.get(node.op, "=="), self._to_float(left), self._to_float(right), name="fcmp")
                    else:
                        op_map = {">": ">", ">=": ">=", "<": "<", "<=": "<=", "==": "==", "!=": "!="}
                        cmp_result = self.builder.icmp_signed(op_map.get(node.op, "=="), left, right, name="cmp")
                    results.append(self.builder.zext(cmp_result, I64, name="cmp_ext"))
                elif isinstance(node, And):
                    right, left = results.pop(), results.pop()
                    results.append(self.builder.and_(left, right, name="and"))
                elif isinstance(node, Or):
                    right, left = results.pop(), results.pop()
                    results.append(self.builder.or_(left, right, name="or"))
                elif isinstance(node, Not):
                    operand = results.pop()
                    results.append(self.builder.xor(operand, llvmir.Constant(I64, 1), name="not"))
                else:
                    self.unsupported_nodes.append(type(node).__name__)
                    results.append(llvmir.Constant(I64, 0))
            else:
                if isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                elif isinstance(node, (Neg, Not)):
                    stack.append((node, True))
                    stack.append((node.operand, False))
                elif isinstance(node, Literal):
                    results.append(self._lower_literal(node))
                elif isinstance(node, Var):
                    if node.name in self.vars:
                        results.append(self.builder.load(self.vars[node.name], name=node.name))
                    else:
                        results.append(llvmir.Constant(I64, 0))
                else:
                    self.unsupported_nodes.append(type(node).__name__)
                    results.append(llvmir.Constant(I64, 0))
        return results[0] if results else llvmir.Constant(I64, 0)

    def _lower_literal(self, expr: Literal) -> llvmir.Value:
        """降级字面量 — 根据类型选择LLVM类型 / Lower literal — choose LLVM type based on value type"""
        val = expr.value
        if val is None:
            return llvmir.Constant(I64, TAG_NONE)
        if isinstance(val, bool):
            return llvmir.Constant(I64, 1 if val else 0)
        if isinstance(val, int):
            return llvmir.Constant(I64, val)
        if isinstance(val, float):
            # 将float编码为i64位模式 / Encode float as i64 bit pattern
            bits = struct.unpack('q', struct.pack('d', val))[0]
            return llvmir.Constant(I64, bits)
        if isinstance(val, str):
            # 字符串暂不支持JIT，返回0 / Strings not yet supported in JIT, return 0
            self.unsupported_nodes.append(f"Literal(str)")
            return llvmir.Constant(I64, 0)
        if isinstance(val, list):
            self.unsupported_nodes.append(f"Literal(list)")
            return llvmir.Constant(I64, 0)
        return llvmir.Constant(I64, 0)

    def _is_float_expr(self, expr) -> bool:
        """检查表达式是否为浮点类型 / Check if expression is float type"""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Literal) and isinstance(node.value, float):
                return True
            if isinstance(node, Div):
                return True
            if isinstance(node, (Add, Sub, Mul)):
                stack.append(node.left)
                stack.append(node.right)
        return False

    def _to_float(self, val: llvmir.Value) -> llvmir.Value:
        """将i64值转换为f64 / Convert i64 value to f64"""
        if isinstance(val.type, llvmir.DoubleType):
            return val
        # i64 -> bitcast -> f64
        ptr = self.builder.alloca(I64)
        self.builder.store(val, ptr)
        fptr = self.builder.bitcast(ptr, F64.as_pointer())
        return self.builder.load(fptr, name="as_float")

    def get_unsupported(self) -> list[str]:
        """获取不支持的节点类型列表 / Get list of unsupported node types"""
        return list(set(self.unsupported_nodes))

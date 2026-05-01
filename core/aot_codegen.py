from __future__ import annotations

import platform
import struct

from llvmlite import ir as llvmir

from .ir import (
    Add,
    And,
    Assign,
    AttributeAccess,
    Break,
    Cmp,
    Continue,
    Div,
    ForRange,
    FuncCall,
    FuncDef,
    If,
    IndexAccess,
    IndexAssign,
    ListExpr,
    Literal,
    Loop,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    Print,
    Return,
    StringConcat,
    Sub,
    Var,
    VarDecl,
    While,
)

I8 = llvmir.IntType(8)
I32 = llvmir.IntType(32)
I64 = llvmir.IntType(64)
I1 = llvmir.IntType(1)
F64 = llvmir.DoubleType()
VOID = llvmir.VoidType()
I8_PTR = I8.as_pointer()
I64_PTR = I64.as_pointer()

NLASM_PTR = I64
TAG_INT = 0
TAG_FLOAT = 1
TAG_BOOL = 2
TAG_STR = 3
TAG_LIST = 4
TAG_NONE = 5

BUILTIN_FUNCS = {
    "abs": 1,
    "max": 2,
    "min": 2,
    "len": 1,
    "int": 1,
    "float": 1,
    "str": 1,
    "bool": 1,
    "type": 1,
    "range": 2,
    "print": 1,
    "input": 0,
    "sorted": 1,
    "reversed": 1,
    "enumerate": 1,
    "zip": 2,
    "isinstance": 2,
    "list": 1,
    "dict": 1,
    "tuple": 1,
    "set": 1,
    "round": 1,
    "pow": 2,
    "divmod": 2,
    "hex": 1,
    "oct": 1,
    "bin": 1,
    "chr": 1,
    "ord": 1,
    "id": 1,
    "hash": 1,
    "callable": 1,
    "hasattr": 2,
    "getattr": 2,
    "setattr": 3,
}


def _get_native_triple() -> str:
    machine = platform.machine().lower()
    system = platform.system().lower()
    if system == "windows":
        if machine in ("amd64", "x86_64"):
            return "x86_64-pc-windows-msvc"
        if machine in ("i386", "i686", "x86"):
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


def _collect_var_names(stmts: list) -> set[str]:
    names: set[str] = set()
    _for_range_count = 0
    _loop_count = 0
    stack: list = list(stmts)
    while stack:
        node = stack.pop()
        if isinstance(node, VarDecl):
            names.add(node.name)
        elif isinstance(node, Assign):
            names.add(node.target)
        elif isinstance(node, IndexAssign):
            stack.append(node.obj)
            stack.append(node.index)
            stack.append(node.value)
        elif isinstance(node, ForRange):
            names.add(node.var)
            names.add(f"__for_stop_{_for_range_count}")
            names.add(f"__for_step_{_for_range_count}")
            _for_range_count += 1
            stack.extend(node.body)
        elif isinstance(node, FuncDef):
            for pname, _ in node.params:
                names.add(pname)
            stack.extend(node.body)
        elif isinstance(node, If):
            stack.extend(node.body)
            stack.extend(node.orelse)
        elif isinstance(node, While):
            stack.extend(node.body)
        elif isinstance(node, Loop):
            names.add(f"__loop_counter_{_loop_count}")
            _loop_count += 1
            stack.extend(node.body)
    return names


class AOTCodeGen:
    """AOT代码生成器 — 直接从IR节点树生成LLVM IR模块.

    值表示: 所有值统一为i64，通过tag系统区分类型:
    - 整数: 直接存储值
    - 浮点数: bitcast为i64存储
    - 布尔值: 0/1
    - 字符串: i64存储指向全局常量的指针
    - 列表: i64存储指向堆分配数组的指针
    - None: 0

    支持: FuncDef, If, While, ForRange, Loop, Assign, VarDecl, IndexAssign,
          Return, Print, Break, Continue, FuncCall,
          Add/Sub/Mul/Div/Mod/Neg/Cmp/And/Or/Not/Literal/Var,
          ListExpr, IndexAccess, StringConcat, AttributeAccess
    """

    def __init__(self) -> None:
        self.module: llvmir.Module | None = None
        self.builder: llvmir.IRBuilder | None = None
        self.vars: dict[str, llvmir.AllocaInstr] = {}
        self.functions: dict[str, llvmir.Function] = {}
        self.printf_func: llvmir.Function | None = None
        self.puts_func: llvmir.Function | None = None
        self.malloc_func: llvmir.Function | None = None
        self.free_func: llvmir.Function | None = None
        self.strlen_func: llvmir.Function | None = None
        self.unsupported_nodes: list[str] = []
        self._fmt_int: llvmir.GlobalVariable | None = None
        self._fmt_float: llvmir.GlobalVariable | None = None
        self._fmt_bool: llvmir.GlobalVariable | None = None
        self._fmt_str: llvmir.GlobalVariable | None = None
        self._fmt_str_nl: llvmir.GlobalVariable | None = None
        self._str_true: llvmir.GlobalVariable | None = None
        self._str_false: llvmir.GlobalVariable | None = None
        self._str_none: llvmir.GlobalVariable | None = None
        self._str_int: llvmir.GlobalVariable | None = None
        self._str_float: llvmir.GlobalVariable | None = None
        self._str_bool: llvmir.GlobalVariable | None = None
        self._str_list: llvmir.GlobalVariable | None = None
        self._str_list_sep: llvmir.GlobalVariable | None = None
        self._str_list_end: llvmir.GlobalVariable | None = None
        self._break_block: llvmir.Block | None = None
        self._continue_block: llvmir.Block | None = None
        self._for_range_count: int = 0
        self._loop_count: int = 0
        self._str_count: int = 0
        self._list_count: int = 0
        self._string_globals: dict[str, llvmir.GlobalVariable] = {}

    def generate(self, stmts: list, module_name: str = "nlasm") -> llvmir.Module:
        self.module = llvmir.Module(name=module_name)
        self.module.triple = _get_native_triple()
        self.vars = {}
        self.functions = {}
        self.unsupported_nodes = []
        self._string_globals = {}
        self._str_count = 0
        self._list_count = 0

        self._declare_c_funcs()

        func_defs = [s for s in stmts if isinstance(s, FuncDef)]
        other_stmts = [s for s in stmts if not isinstance(s, FuncDef)]

        for fd in func_defs:
            self._declare_function(fd)

        for fd in func_defs:
            self._compile_func_def(fd)

        if other_stmts:
            self._compile_main(other_stmts)

        return self.module

    def _declare_c_funcs(self) -> None:
        int_fmt = "%ld\n\0"
        float_fmt = "%.6f\n\0"
        bool_fmt = "%s\n\0"
        str_fmt = "%s\n\0"
        str_nl_fmt = "%s\n\0"
        true_str = "true\0"
        false_str = "false\0"
        none_str = "None\0"
        type_int_str = "<int>\0"
        type_float_str = "<float>\0"
        type_bool_str = "<bool>\0"
        type_str_str = "<str>\0"
        type_list_str = "<list>\0"
        list_sep_str = ", \0"
        list_end_str = "]\0"

        def _make_global(name: str, text: str) -> llvmir.GlobalVariable:
            encoded = text.encode("utf-8")
            arr_type = llvmir.ArrayType(I8, len(encoded))
            gv = llvmir.GlobalVariable(self.module, arr_type, name=name)
            gv.global_constant = True
            gv.initializer = llvmir.Constant(arr_type, bytearray(encoded))
            return gv

        self._fmt_int = _make_global(".fmt_int", int_fmt)
        self._fmt_float = _make_global(".fmt_float", float_fmt)
        self._fmt_bool = _make_global(".fmt_bool", bool_fmt)
        self._fmt_str = _make_global(".fmt_str", str_fmt)
        self._fmt_str_nl = _make_global(".fmt_str_nl", str_nl_fmt)
        self._str_true = _make_global(".str_true", true_str)
        self._str_false = _make_global(".str_false", false_str)
        self._str_none = _make_global(".str_none", none_str)
        self._str_int = _make_global(".str_int", type_int_str)
        self._str_float = _make_global(".str_float", type_float_str)
        self._str_bool = _make_global(".str_bool", type_bool_str)
        self._str_list = _make_global(".str_list", type_list_str)
        self._str_list_sep = _make_global(".str_list_sep", list_sep_str)
        self._str_list_end = _make_global(".str_list_end", list_end_str)

        printf_type = llvmir.FunctionType(I32, [I8_PTR], var_arg=True)
        self.printf_func = llvmir.Function(self.module, printf_type, name="printf")

        puts_type = llvmir.FunctionType(I32, [I8_PTR])
        self.puts_func = llvmir.Function(self.module, puts_type, name="puts")

        malloc_type = llvmir.FunctionType(I8_PTR, [I64])
        self.malloc_func = llvmir.Function(self.module, malloc_type, name="malloc")

        free_type = llvmir.FunctionType(VOID, [I8_PTR])
        self.free_func = llvmir.Function(self.module, free_type, name="free")

        strlen_type = llvmir.FunctionType(I64, [I8_PTR])
        self.strlen_func = llvmir.Function(self.module, strlen_type, name="strlen")

    def _get_string_global(self, text: str) -> llvmir.GlobalVariable:
        if text in self._string_globals:
            return self._string_globals[text]
        encoded = text.encode("utf-8") + b"\0"
        arr_type = llvmir.ArrayType(I8, len(encoded))
        name = f".str_lit_{self._str_count}"
        self._str_count += 1
        gv = llvmir.GlobalVariable(self.module, arr_type, name=name)
        gv.global_constant = True
        gv.initializer = llvmir.Constant(arr_type, bytearray(encoded))
        self._string_globals[text] = gv
        return gv

    def _declare_function(self, fd: FuncDef) -> None:
        param_count = len(fd.params)
        ftype = llvmir.FunctionType(I64, [I64] * param_count)
        func = llvmir.Function(self.module, ftype, name=fd.name)
        self.functions[fd.name] = func

    def _create_entry_allocas(self, entry_block: llvmir.Block, var_names: set[str]) -> None:
        builder = llvmir.IRBuilder(entry_block)
        for name in sorted(var_names):
            alloca = builder.alloca(I64, name=name)
            self.vars[name] = alloca

    def _compile_func_def(self, fd: FuncDef) -> None:
        func = self.functions[fd.name]
        entry = func.append_basic_block(name="entry")
        old_builder = self.builder
        old_vars = self.vars
        old_break = self._break_block
        old_continue = self._continue_block
        old_for_count = self._for_range_count
        old_loop_count = self._loop_count

        self.vars = {}
        self._break_block = None
        self._continue_block = None
        self._for_range_count = 0
        self._loop_count = 0

        var_names = _collect_var_names(fd.body)
        self._create_entry_allocas(entry, var_names)

        self.builder = llvmir.IRBuilder(entry)

        for i, (pname, _ptype) in enumerate(fd.params):
            if pname in self.vars:
                self.builder.store(func.args[i], self.vars[pname])
            else:
                alloca = self.builder.alloca(I64, name=pname)
                self.builder.store(func.args[i], alloca)
                self.vars[pname] = alloca

        self._compile_body(fd.body)

        if not self.builder.block.is_terminated:
            self.builder.ret(llvmir.Constant(I64, 0))

        self.builder = old_builder
        self.vars = old_vars
        self._break_block = old_break
        self._continue_block = old_continue
        self._for_range_count = old_for_count
        self._loop_count = old_loop_count

    def _compile_main(self, stmts: list) -> None:
        ftype = llvmir.FunctionType(I32, [])
        main_func = llvmir.Function(self.module, ftype, name="main")
        entry = main_func.append_basic_block(name="entry")

        old_builder = self.builder
        old_vars = self.vars

        self.vars = {}
        self._for_range_count = 0
        self._loop_count = 0

        var_names = _collect_var_names(stmts)
        self._create_entry_allocas(entry, var_names)

        self.builder = llvmir.IRBuilder(entry)

        self._compile_body(stmts)

        if not self.builder.block.is_terminated:
            self.builder.ret(llvmir.Constant(I32, 0))
        else:
            last_term = self.builder.block.terminator
            if isinstance(last_term, llvmir.RetInstruction) and last_term.operands:
                ret_val = last_term.operands[0]
                if ret_val.type == I64:
                    last_term.erase()
                    self.builder.ret(self.builder.trunc(ret_val, I32, name="ret_trunc"))

        self.builder = old_builder
        self.vars = old_vars

    def _compile_body(self, stmts: list) -> None:
        for stmt in stmts:
            if self.builder.block.is_terminated:
                break
            self._compile_stmt(stmt)

    def _block_has_return(self, stmts: list) -> bool:
        for stmt in stmts:
            if isinstance(stmt, Return):
                return True
            if isinstance(stmt, If):
                body_returns = self._block_has_return(stmt.body)
                orelse_returns = self._block_has_return(stmt.orelse)
                if body_returns and orelse_returns:
                    return True
        return False

    def _get_or_create_alloca(self, name: str) -> llvmir.AllocaInstr:
        if name not in self.vars:
            alloca = self.builder.alloca(I64, name=name)
            self.vars[name] = alloca
        return self.vars[name]

    def _compile_stmt(self, stmt) -> None:
        if isinstance(stmt, Assign):
            self._compile_assign(stmt)
        elif isinstance(stmt, VarDecl):
            self._compile_var_decl(stmt)
        elif isinstance(stmt, Print):
            self._compile_print(stmt)
        elif isinstance(stmt, Return):
            self._compile_return(stmt)
        elif isinstance(stmt, If):
            self._compile_if(stmt)
        elif isinstance(stmt, While):
            self._compile_while(stmt)
        elif isinstance(stmt, ForRange):
            self._compile_for_range(stmt)
        elif isinstance(stmt, Loop):
            self._compile_loop(stmt)
        elif isinstance(stmt, Break):
            self._compile_break()
        elif isinstance(stmt, Continue):
            self._compile_continue()
        elif isinstance(stmt, FuncCall):
            self._compile_func_call_expr(stmt)
        elif isinstance(stmt, IndexAssign):
            self._compile_index_assign(stmt)
        elif isinstance(stmt, FuncDef):
            pass
        else:
            self.unsupported_nodes.append(type(stmt).__name__)

    def _compile_assign(self, stmt: Assign) -> None:
        val = self._compile_expr(stmt.value)
        alloca = self._get_or_create_alloca(stmt.target)
        self.builder.store(val, alloca)

    def _compile_var_decl(self, stmt: VarDecl) -> None:
        val = self._compile_expr(stmt.value)
        alloca = self._get_or_create_alloca(stmt.name)
        self.builder.store(val, alloca)

    def _compile_index_assign(self, stmt: IndexAssign) -> None:
        obj_val = self._compile_expr(stmt.obj)
        idx_val = self._compile_expr(stmt.index)
        val = self._compile_expr(stmt.value)
        list_ptr = self.builder.inttoptr(obj_val, I64_PTR, name="list_ptr")
        offset = self.builder.add(idx_val, llvmir.Constant(I64, 1), name="idx_offset")
        elem_ptr = self.builder.gep(list_ptr, [offset], name="elem_ptr")
        self.builder.store(val, elem_ptr)

    def _compile_print(self, stmt: Print) -> None:
        if stmt.values:
            for v in stmt.values:
                self._print_value(v)
        else:
            self._print_value(stmt.value)

    def _print_value(self, expr) -> None:
        val = self._compile_expr(expr)
        if self._is_string_expr(expr):
            str_ptr = self.builder.inttoptr(val, I8_PTR, name="str_ptr")
            fmt_ptr = self.builder.bitcast(self._fmt_str_nl, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, str_ptr])
        elif self._is_float_expr(expr):
            fval = self._i64_to_float(val)
            fmt_ptr = self.builder.bitcast(self._fmt_float, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, fval])
        elif self._is_bool_expr(expr):
            is_true = self.builder.icmp_signed("!=", val, llvmir.Constant(I64, 0), name="is_true")
            str_ptr = self.builder.select(is_true, self._str_true, self._str_false, name="bool_str")
            fmt_ptr = self.builder.bitcast(self._fmt_bool, I8_PTR)
            str_i8 = self.builder.bitcast(str_ptr, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, str_i8])
        elif self._is_list_expr(expr):
            self._print_list(val)
        elif self._is_none_expr(expr):
            fmt_ptr = self.builder.bitcast(self._fmt_str, I8_PTR)
            none_ptr = self.builder.bitcast(self._str_none, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, none_ptr])
            nl = self.builder.alloca(I8)
            self.builder.store(llvmir.Constant(I8, ord('\n')), nl)
            self.builder.putchar(llvmir.Constant(I32, ord('\n')))
        else:
            fmt_ptr = self.builder.bitcast(self._fmt_int, I8_PTR)
            self.builder.call(self.printf_func, [fmt_ptr, val])

    def _print_list(self, list_val: llvmir.Value) -> None:
        fmt_ptr = self.builder.bitcast(self._fmt_str, I8_PTR)
        bracket_ptr = self.builder.bitcast(self._str_list, I8_PTR)
        self.builder.call(self.printf_func, [fmt_ptr, bracket_ptr])

        len_ptr = self.builder.inttoptr(list_val, I64_PTR, name="list_meta")
        list_len = self.builder.load(len_ptr, name="list_len")
        data_ptr = self.builder.gep(len_ptr, [llvmir.Constant(I64, 1)], name="list_data_start")

        loop_bb = self.builder.function.append_basic_block(name="print_list_loop")
        done_bb = self.builder.function.append_basic_block(name="print_list_done")

        i_alloca = self.builder.alloca(I64, name=".print_i")
        self.builder.store(llvmir.Constant(I64, 0), i_alloca)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(loop_bb)
        i = self.builder.load(i_alloca, name="i")
        cond = self.builder.icmp_signed("<", i, list_len, name="print_cond")
        self.builder.cbranch(cond, done_bb, done_bb)

        body_bb = self.builder.function.append_basic_block(name="print_list_body")
        self.builder.position_at_end(loop_bb)
        self.builder.cbranch(cond, body_bb, done_bb)

        self.builder.position_at_end(body_bb)
        is_first = self.builder.icmp_signed("==", i, llvmir.Constant(I64, 0), name="is_first")
        then_bb = self.builder.function.append_basic_block(name="print_sep_then")
        merge_bb = self.builder.function.append_basic_block(name="print_sep_merge")
        self.builder.cbranch(is_first, then_bb, merge_bb)

        self.builder.position_at_end(then_bb)
        sep_ptr = self.builder.bitcast(self._str_list_sep, I8_PTR)
        self.builder.call(self.printf_func, [fmt_ptr, sep_ptr])
        self.builder.branch(merge_bb)

        self.builder.position_at_end(merge_bb)
        elem_ptr = self.builder.gep(data_ptr, [i], name="elem_ptr")
        elem_val = self.builder.load(elem_ptr, name="elem_val")
        fmt_int_ptr = self.builder.bitcast(self._fmt_int, I8_PTR)
        self.builder.call(self.printf_func, [fmt_int_ptr, elem_val])

        next_i = self.builder.add(i, llvmir.Constant(I64, 1), name="next_i")
        self.builder.store(next_i, i_alloca)
        self.builder.branch(loop_bb)

        self.builder.position_at_end(done_bb)
        end_ptr = self.builder.bitcast(self._str_list_end, I8_PTR)
        self.builder.call(self.printf_func, [fmt_ptr, end_ptr])
        self.builder.putchar(llvmir.Constant(I32, ord('\n')))

    def _compile_return(self, stmt: Return) -> None:
        if stmt.value is not None:
            val = self._compile_expr(stmt.value)
            self.builder.ret(val)
        else:
            self.builder.ret(llvmir.Constant(I64, 0))

    def _compile_if(self, stmt: If) -> None:
        cond = self._compile_expr(stmt.condition)
        cond_bool = self.builder.icmp_signed("!=", cond, llvmir.Constant(I64, 0), name="if_cond")

        func = self.builder.function
        then_bb = func.append_basic_block(name="then")
        else_bb = func.append_basic_block(name="else")
        merge_bb = func.append_basic_block(name="if_merge")

        self.builder.cbranch(cond_bool, then_bb, else_bb)

        self.builder.position_at_end(then_bb)
        self._compile_body(stmt.body)
        then_terminated = self.builder.block.is_terminated
        then_has_return = self._block_has_return(stmt.body)
        if not then_terminated:
            self.builder.branch(merge_bb)

        self.builder.position_at_end(else_bb)
        self._compile_body(stmt.orelse)
        else_terminated = self.builder.block.is_terminated
        else_has_return = self._block_has_return(stmt.orelse)
        if not else_terminated:
            self.builder.branch(merge_bb)

        if then_has_return and else_has_return:
            pass
        elif then_terminated and else_terminated:
            self.builder.position_at_end(merge_bb)
            self.builder.unreachable()
        else:
            self.builder.position_at_end(merge_bb)

    def _compile_while(self, stmt: While) -> None:
        func = self.builder.function
        cond_bb = func.append_basic_block(name="while_cond")
        body_bb = func.append_basic_block(name="while_body")
        after_bb = func.append_basic_block(name="while_after")

        old_break = self._break_block
        old_continue = self._continue_block
        self._break_block = after_bb
        self._continue_block = cond_bb

        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        cond = self._compile_expr(stmt.condition)
        cond_bool = self.builder.icmp_signed("!=", cond, llvmir.Constant(I64, 0), name="while_cond")
        self.builder.cbranch(cond_bool, body_bb, after_bb)

        self.builder.position_at_end(body_bb)
        self._compile_body(stmt.body)
        if not self.builder.block.is_terminated:
            self.builder.branch(cond_bb)

        self.builder.position_at_end(after_bb)
        self._break_block = old_break
        self._continue_block = old_continue

    def _compile_for_range(self, stmt: ForRange) -> None:
        func = self.builder.function
        cond_bb = func.append_basic_block(name="for_cond")
        body_bb = func.append_basic_block(name="for_body")
        step_bb = func.append_basic_block(name="for_step")
        after_bb = func.append_basic_block(name="for_after")

        start_val = self._compile_expr(stmt.start)
        stop_val = self._compile_expr(stmt.stop)

        var_alloca = self._get_or_create_alloca(stmt.var)
        self.builder.store(start_val, var_alloca)

        if stmt.step is not None:
            step_val = self._compile_expr(stmt.step)
        else:
            step_val = llvmir.Constant(I64, 1)

        stop_name = f"__for_stop_{self._for_range_count}"
        step_name = f"__for_step_{self._for_range_count}"
        self._for_range_count += 1

        stop_alloca = self._get_or_create_alloca(stop_name)
        step_alloca = self._get_or_create_alloca(step_name)
        self.builder.store(stop_val, stop_alloca)
        self.builder.store(step_val, step_alloca)

        old_break = self._break_block
        old_continue = self._continue_block
        self._break_block = after_bb
        self._continue_block = step_bb

        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        cur = self.builder.load(var_alloca, name=stmt.var)
        stop = self.builder.load(stop_alloca, name="stop")
        cond = self.builder.icmp_signed("<=", cur, stop, name="for_cond")
        self.builder.cbranch(cond, body_bb, after_bb)

        self.builder.position_at_end(body_bb)
        self._compile_body(stmt.body)
        if not self.builder.block.is_terminated:
            self.builder.branch(step_bb)

        self.builder.position_at_end(step_bb)
        cur = self.builder.load(var_alloca, name=stmt.var)
        stp = self.builder.load(step_alloca, name="step")
        next_val = self.builder.add(cur, stp, name="next")
        self.builder.store(next_val, var_alloca)
        self.builder.branch(cond_bb)

        self.builder.position_at_end(after_bb)
        self._break_block = old_break
        self._continue_block = old_continue

    def _compile_loop(self, stmt: Loop) -> None:
        func = self.builder.function
        cond_bb = func.append_basic_block(name="loop_cond")
        body_bb = func.append_basic_block(name="loop_body")
        after_bb = func.append_basic_block(name="loop_after")

        counter_name = f"__loop_counter_{self._loop_count}"
        self._loop_count += 1

        counter_alloca = self._get_or_create_alloca(counter_name)
        self.builder.store(llvmir.Constant(I64, 0), counter_alloca)

        old_break = self._break_block
        old_continue = self._continue_block
        self._break_block = after_bb
        self._continue_block = cond_bb

        self.builder.branch(cond_bb)

        self.builder.position_at_end(cond_bb)
        counter = self.builder.load(counter_alloca, name="counter")
        cond = self.builder.icmp_signed("<", counter, llvmir.Constant(I64, stmt.count), name="loop_cond")
        self.builder.cbranch(cond, body_bb, after_bb)

        self.builder.position_at_end(body_bb)
        self._compile_body(stmt.body)
        if not self.builder.block.is_terminated:
            counter = self.builder.load(counter_alloca, name="counter")
            next_counter = self.builder.add(counter, llvmir.Constant(I64, 1), name="next_counter")
            self.builder.store(next_counter, counter_alloca)
            self.builder.branch(cond_bb)

        self.builder.position_at_end(after_bb)
        self._break_block = old_break
        self._continue_block = old_continue

    def _compile_break(self) -> None:
        if self._break_block is not None:
            self.builder.branch(self._break_block)

    def _compile_continue(self) -> None:
        if self._continue_block is not None:
            self.builder.branch(self._continue_block)

    def _compile_func_call_expr(self, stmt: FuncCall) -> None:
        self._compile_expr(stmt)

    def _compile_expr(self, expr) -> llvmir.Value:
        if isinstance(expr, Literal):
            return self._compile_literal(expr)
        if isinstance(expr, Var):
            return self._compile_var(expr)
        if isinstance(expr, Add):
            return self._compile_add(expr)
        if isinstance(expr, Sub):
            return self._compile_sub(expr)
        if isinstance(expr, Mul):
            return self._compile_mul(expr)
        if isinstance(expr, Div):
            return self._compile_div(expr)
        if isinstance(expr, Mod):
            return self._compile_mod(expr)
        if isinstance(expr, Neg):
            return self._compile_neg(expr)
        if isinstance(expr, Cmp):
            return self._compile_cmp(expr)
        if isinstance(expr, And):
            return self._compile_and(expr)
        if isinstance(expr, Or):
            return self._compile_or(expr)
        if isinstance(expr, Not):
            return self._compile_not(expr)
        if isinstance(expr, FuncCall):
            return self._compile_func_call(expr)
        if isinstance(expr, ListExpr):
            return self._compile_list_expr(expr)
        if isinstance(expr, IndexAccess):
            return self._compile_index_access(expr)
        if isinstance(expr, StringConcat):
            return self._compile_string_concat(expr)
        if isinstance(expr, AttributeAccess):
            return self._compile_attribute_access(expr)
        self.unsupported_nodes.append(type(expr).__name__)
        return llvmir.Constant(I64, 0)

    def _compile_literal(self, expr: Literal) -> llvmir.Value:
        val = expr.value
        if val is None:
            return llvmir.Constant(I64, 0)
        if isinstance(val, bool):
            return llvmir.Constant(I64, 1 if val else 0)
        if isinstance(val, int):
            return llvmir.Constant(I64, val)
        if isinstance(val, float):
            bits = struct.unpack('q', struct.pack('d', val))[0]
            return llvmir.Constant(I64, bits)
        if isinstance(val, str):
            return self._compile_string_literal(val)
        if isinstance(val, list):
            return self._compile_inline_list(val)
        return llvmir.Constant(I64, 0)

    def _compile_string_literal(self, text: str) -> llvmir.Value:
        gv = self._get_string_global(text)
        ptr = self.builder.bitcast(gv, I8_PTR, name="str_ptr")
        return self.builder.ptrtoint(ptr, I64, name="str_as_i64")

    def _compile_inline_list(self, elements: list) -> llvmir.Value:
        n = len(elements)
        total_size = llvmir.Constant(I64, (n + 1) * 8)
        raw_ptr = self.builder.call(self.malloc_func, [total_size], name="list_mem")
        list_ptr = self.builder.bitcast(raw_ptr, I64_PTR, name="list_ptr")
        self.builder.store(llvmir.Constant(I64, n), list_ptr, name="list_len_store")
        for i, elem in enumerate(elements):
            elem_val = self._compile_expr(Literal(elem) if not isinstance(elem, (int, float, bool, str, list, type(None))) else Literal(elem))
            if isinstance(elem, bool):
                elem_val = llvmir.Constant(I64, 1 if elem else 0)
            elif isinstance(elem, int):
                elem_val = llvmir.Constant(I64, elem)
            elif isinstance(elem, float):
                bits = struct.unpack('q', struct.pack('d', elem))[0]
                elem_val = llvmir.Constant(I64, bits)
            elif isinstance(elem, str):
                elem_val = self._compile_string_literal(elem)
            elif elem is None:
                elem_val = llvmir.Constant(I64, 0)
            elem_ptr = self.builder.gep(list_ptr, [llvmir.Constant(I64, i + 1)], name=f"list_elem_{i}")
            self.builder.store(elem_val, elem_ptr)
        return self.builder.ptrtoint(list_ptr, I64, name="list_as_i64")

    def _compile_list_expr(self, expr: ListExpr) -> llvmir.Value:
        n = len(expr.elements)
        total_size = llvmir.Constant(I64, (n + 1) * 8)
        raw_ptr = self.builder.call(self.malloc_func, [total_size], name="list_mem")
        list_ptr = self.builder.bitcast(raw_ptr, I64_PTR, name="list_ptr")
        self.builder.store(llvmir.Constant(I64, n), list_ptr)
        for i, elem_expr in enumerate(expr.elements):
            elem_val = self._compile_expr(elem_expr)
            elem_ptr = self.builder.gep(list_ptr, [llvmir.Constant(I64, i + 1)], name=f"list_elem_{i}")
            self.builder.store(elem_val, elem_ptr)
        return self.builder.ptrtoint(list_ptr, I64, name="list_as_i64")

    def _compile_index_access(self, expr: IndexAccess) -> llvmir.Value:
        obj_val = self._compile_expr(expr.obj)
        idx_val = self._compile_expr(expr.index)
        list_ptr = self.builder.inttoptr(obj_val, I64_PTR, name="list_ptr")
        offset = self.builder.add(idx_val, llvmir.Constant(I64, 1), name="idx_offset")
        elem_ptr = self.builder.gep(list_ptr, [offset], name="elem_ptr")
        return self.builder.load(elem_ptr, name="elem_val")

    def _compile_string_concat(self, expr: StringConcat) -> llvmir.Value:
        left_val = self._compile_expr(expr.left)
        right_val = self._compile_expr(expr.right)

        left_ptr = self.builder.inttoptr(left_val, I8_PTR, name="left_str")
        right_ptr = self.builder.inttoptr(right_val, I8_PTR, name="right_str")

        left_len = self.builder.call(self.strlen_func, [left_ptr], name="left_len")
        right_len = self.builder.call(self.strlen_func, [right_ptr], name="right_len")
        total_len = self.builder.add(left_len, right_len, name="total_len")
        alloc_size = self.builder.add(total_len, llvmir.Constant(I64, 1), name="alloc_size")

        raw_ptr = self.builder.call(self.malloc_func, [alloc_size], name="concat_mem")
        self.builder.call(self._get_memcpy_func(), [
            raw_ptr, left_ptr, left_len,
            llvmir.Constant(I32, 0), llvmir.Constant(I1, 0),
        ])
        dest_offset = self.builder.gep(raw_ptr, [left_len], name="dest_offset")
        self.builder.call(self._get_memcpy_func(), [
            dest_offset, right_ptr, right_len,
            llvmir.Constant(I32, 0), llvmir.Constant(I1, 0),
        ])
        null_byte_ptr = self.builder.gep(raw_ptr, [total_len], name="null_byte")
        self.builder.store(llvmir.Constant(I8, 0), null_byte_ptr)

        return self.builder.ptrtoint(raw_ptr, I64, name="concat_as_i64")

    def _get_memcpy_func(self) -> llvmir.Function:
        name = "llvm.memcpy.p0i8.p0i8.i64"
        if name in [f.name for f in self.module.functions]:
            return self.module.get_global(name)
        ftype = llvmir.FunctionType(VOID, [I8_PTR, I8_PTR, I64, I32, I1])
        return llvmir.Function(self.module, ftype, name=name)

    def _compile_attribute_access(self, expr: AttributeAccess) -> llvmir.Value:
        self.unsupported_nodes.append(f"AttributeAccess({expr.attr})")
        return llvmir.Constant(I64, 0)

    def _compile_var(self, expr: Var) -> llvmir.Value:
        if expr.name in self.vars:
            return self.builder.load(self.vars[expr.name], name=expr.name)
        return llvmir.Constant(I64, 0)

    def _compile_add(self, expr: Add) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        if self._is_string_expr(expr.left) and self._is_string_expr(expr.right):
            concat = StringConcat(expr.left, expr.right)
            return self._compile_string_concat(concat)
        if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
            return self._float_binop(left, right, self.builder.fadd, "fadd")
        return self.builder.add(left, right, name="add")

    def _compile_sub(self, expr: Sub) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
            return self._float_binop(left, right, self.builder.fsub, "fsub")
        return self.builder.sub(left, right, name="sub")

    def _compile_mul(self, expr: Mul) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
            return self._float_binop(left, right, self.builder.fmul, "fmul")
        return self.builder.mul(left, right, name="mul")

    def _compile_div(self, expr: Div) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
            return self._float_binop(left, right, self.builder.fdiv, "fdiv")
        return self.builder.sdiv(left, right, name="sdiv")

    def _compile_mod(self, expr: Mod) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        return self.builder.srem(left, right, name="srem")

    def _compile_neg(self, expr: Neg) -> llvmir.Value:
        operand = self._compile_expr(expr.operand)
        if self._is_float_expr(expr.operand):
            fval = self._i64_to_float(operand)
            result = self.builder.fneg(fval, name="fneg")
            return self._float_to_i64(result)
        return self.builder.neg(operand, name="neg")

    def _compile_cmp(self, expr: Cmp) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        if self._is_float_expr(expr.left) or self._is_float_expr(expr.right):
            fcmp_map = {
                ">": ">", ">=": ">=", "<": "<", "<=": "<=",
                "==": "==", "!=": "!=",
            }
            fl = self._i64_to_float(left)
            fr = self._i64_to_float(right)
            cmp_result = self.builder.fcmp_ordered(
                fcmp_map.get(expr.op, "=="), fl, fr, name="fcmp"
            )
        else:
            op_map = {
                ">": ">", ">=": ">=", "<": "<", "<=": "<=",
                "==": "==", "!=": "!=",
            }
            cmp_result = self.builder.icmp_signed(
                op_map.get(expr.op, "=="), left, right, name="cmp"
            )
        return self.builder.zext(cmp_result, I64, name="cmp_ext")

    def _compile_and(self, expr: And) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        l_bool = self.builder.icmp_signed("!=", left, llvmir.Constant(I64, 0), name="and_l")
        r_bool = self.builder.icmp_signed("!=", right, llvmir.Constant(I64, 0), name="and_r")
        result = self.builder.and_(l_bool, r_bool, name="and")
        return self.builder.zext(result, I64, name="and_ext")

    def _compile_or(self, expr: Or) -> llvmir.Value:
        left = self._compile_expr(expr.left)
        right = self._compile_expr(expr.right)
        l_bool = self.builder.icmp_signed("!=", left, llvmir.Constant(I64, 0), name="or_l")
        r_bool = self.builder.icmp_signed("!=", right, llvmir.Constant(I64, 0), name="or_r")
        result = self.builder.or_(l_bool, r_bool, name="or")
        return self.builder.zext(result, I64, name="or_ext")

    def _compile_not(self, expr: Not) -> llvmir.Value:
        operand = self._compile_expr(expr.operand)
        is_zero = self.builder.icmp_signed("==", operand, llvmir.Constant(I64, 0), name="not")
        return self.builder.zext(is_zero, I64, name="not_ext")

    def _compile_func_call(self, expr: FuncCall) -> llvmir.Value:
        name = expr.name
        args = [self._compile_expr(a) for a in expr.args]

        if name in self.functions:
            func = self.functions[name]
            return self.builder.call(func, args, name="call_" + name)

        if name in BUILTIN_FUNCS:
            return self._compile_builtin_call(name, args)

        if name.startswith("_attr_"):
            self.unsupported_nodes.append(f"MethodCall({name})")
            return llvmir.Constant(I64, 0)

        self.unsupported_nodes.append(f"FuncCall({name})")
        return llvmir.Constant(I64, 0)

    def _compile_builtin_call(self, name: str, args: list[llvmir.Value]) -> llvmir.Value:
        if name == "abs":
            if args:
                zero = llvmir.Constant(I64, 0)
                is_neg = self.builder.icmp_signed("<", args[0], zero, name="is_neg")
                neg_val = self.builder.neg(args[0], name="neg_abs")
                return self.builder.select(is_neg, neg_val, args[0], name="abs_result")
            return llvmir.Constant(I64, 0)

        if name == "max":
            if len(args) >= 2:
                cmp = self.builder.icmp_signed(">", args[0], args[1], name="max_cmp")
                return self.builder.select(cmp, args[0], args[1], name="max_result")
            return args[0] if args else llvmir.Constant(I64, 0)

        if name == "min":
            if len(args) >= 2:
                cmp = self.builder.icmp_signed("<", args[0], args[1], name="min_cmp")
                return self.builder.select(cmp, args[0], args[1], name="min_result")
            return args[0] if args else llvmir.Constant(I64, 0)

        if name == "len":
            if args:
                ptr = self.builder.inttoptr(args[0], I64_PTR, name="list_meta")
                return self.builder.load(ptr, name="list_length")
            return llvmir.Constant(I64, 0)

        if name == "int":
            return args[0] if args else llvmir.Constant(I64, 0)

        if name == "float":
            return args[0] if args else llvmir.Constant(I64, 0)

        if name == "str":
            return args[0] if args else llvmir.Constant(I64, 0)

        if name == "bool":
            if args:
                is_nonzero = self.builder.icmp_signed("!=", args[0], llvmir.Constant(I64, 0), name="bool_conv")
                return self.builder.zext(is_nonzero, I64, name="bool_result")
            return llvmir.Constant(I64, 0)

        if name == "pow":
            if len(args) >= 2:
                base = args[0]
                exp = args[1]
                exp_const = self.builder.icmp_signed("==", exp, llvmir.Constant(I64, 0), name="exp_is_zero")
                one = llvmir.Constant(I64, 1)
                result_alloca = self.builder.alloca(I64, name=".pow_result")
                self.builder.store(one, result_alloca)
                counter_alloca = self.builder.alloca(I64, name=".pow_counter")
                self.builder.store(llvmir.Constant(I64, 0), counter_alloca)

                loop_bb = self.builder.function.append_basic_block(name="pow_loop")
                done_bb = self.builder.function.append_basic_block(name="pow_done")

                self.builder.cbranch(exp_const, done_bb, loop_bb)

                self.builder.position_at_end(loop_bb)
                cur_result = self.builder.load(result_alloca, name="cur_result")
                new_result = self.builder.mul(cur_result, base, name="new_result")
                self.builder.store(new_result, result_alloca)
                cur_counter = self.builder.load(counter_alloca, name="cur_counter")
                new_counter = self.builder.add(cur_counter, llvmir.Constant(I64, 1), name="new_counter")
                self.builder.store(new_counter, counter_alloca)
                done_cond = self.builder.icmp_signed(">=", new_counter, exp, name="pow_done_cond")
                self.builder.cbranch(done_cond, done_bb, loop_bb)

                self.builder.position_at_end(done_bb)
                return self.builder.load(result_alloca, name="pow_final")
            return llvmir.Constant(I64, 1)

        if name == "range":
            if len(args) >= 2:
                start_val = args[0]
                stop_val = args[1]
                count = self.builder.sub(stop_val, start_val, name="range_count")
                count_safe = self.builder.select(
                    self.builder.icmp_signed("<", count, llvmir.Constant(I64, 0), name="neg_check"),
                    llvmir.Constant(I64, 0), count, name="count_safe"
                )
                total_size = self.builder.add(
                    self.builder.mul(count_safe, llvmir.Constant(I64, 8), name="data_size"),
                    llvmir.Constant(I64, 8), name="total_alloc"
                )
                raw_ptr = self.builder.call(self.malloc_func, [total_size], name="range_mem")
                list_ptr = self.builder.bitcast(raw_ptr, I64_PTR, name="range_list")
                self.builder.store(count_safe, list_ptr)

                loop_bb = self.builder.function.append_basic_block(name="range_loop")
                done_bb = self.builder.function.append_basic_block(name="range_done")

                i_alloca = self.builder.alloca(I64, name=".range_i")
                self.builder.store(llvmir.Constant(I64, 0), i_alloca)
                self.builder.branch(loop_bb)

                self.builder.position_at_end(loop_bb)
                i = self.builder.load(i_alloca, name="range_i")
                cond = self.builder.icmp_signed("<", i, count_safe, name="range_cond")
                self.builder.cbranch(cond, loop_bb, done_bb)

                body_bb = self.builder.function.append_basic_block(name="range_body")
                self.builder.position_at_end(loop_bb)
                self.builder.cbranch(cond, body_bb, done_bb)

                self.builder.position_at_end(body_bb)
                val = self.builder.add(start_val, i, name="range_val")
                offset = self.builder.add(i, llvmir.Constant(I64, 1), name="range_offset")
                elem_ptr = self.builder.gep(list_ptr, [offset], name="range_elem")
                self.builder.store(val, elem_ptr)
                next_i = self.builder.add(i, llvmir.Constant(I64, 1), name="range_next_i")
                self.builder.store(next_i, i_alloca)
                self.builder.branch(loop_bb)

                self.builder.position_at_end(done_bb)
                return self.builder.ptrtoint(list_ptr, I64, name="range_as_i64")
            return llvmir.Constant(I64, 0)

        self.unsupported_nodes.append(f"Builtin({name})")
        return args[0] if args else llvmir.Constant(I64, 0)

    def _float_binop(self, left, right, op_fn, name: str) -> llvmir.Value:
        fl = self._i64_to_float(left)
        fr = self._i64_to_float(right)
        result = op_fn(fl, fr, name=name)
        return self._float_to_i64(result)

    def _i64_to_float(self, val: llvmir.Value) -> llvmir.Value:
        if isinstance(val.type, llvmir.DoubleType):
            return val
        ptr = self.builder.alloca(I64)
        self.builder.store(val, ptr)
        fptr = self.builder.bitcast(ptr, F64.as_pointer())
        return self.builder.load(fptr, name="as_float")

    def _float_to_i64(self, val: llvmir.Value) -> llvmir.Value:
        if isinstance(val.type, llvmir.IntType) and val.type.width == 64:
            return val
        ptr = self.builder.alloca(F64)
        self.builder.store(val, ptr)
        iptr = self.builder.bitcast(ptr, I64.as_pointer())
        return self.builder.load(iptr, name="as_i64")

    def _is_float_expr(self, expr) -> bool:
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
            if isinstance(node, Neg):
                stack.append(node.operand)
        return False

    def _is_bool_expr(self, expr) -> bool:
        return isinstance(expr, (Cmp, And, Or, Not))

    def _is_string_expr(self, expr) -> bool:
        if isinstance(expr, Literal) and isinstance(expr.value, str):
            return True
        if isinstance(expr, StringConcat):
            return True
        if isinstance(expr, Var) and expr.name in self.vars:
            return False
        return False

    def _is_list_expr(self, expr) -> bool:
        if isinstance(expr, ListExpr):
            return True
        if isinstance(expr, Literal) and isinstance(expr.value, list):
            return True
        if isinstance(expr, FuncCall) and expr.name == "range":
            return True
        return False

    def _is_none_expr(self, expr) -> bool:
        return isinstance(expr, Literal) and expr.value is None

    def get_unsupported(self) -> list[str]:
        return list(set(self.unsupported_nodes))

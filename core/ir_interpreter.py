from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from .ir import (
    Add,
    And,
    AsyncCall,
    Assign,
    AttributeAccess,
    AwaitExpr,
    Break,
    CallPython,
    ClassDef,
    Cmp,
    Continue,
    DictExpr,
    Div,
    Expr,
    For,
    ForRange,
    FuncCall,
    FuncDef,
    If,
    ImportStmt,
    IndexAccess,
    IndexAssign,
    ListExpr,
    Literal,
    Loop,
    Match,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    ParallelCall,
    Print,
    Raise,
    Return,
    Stmt,
    StringConcat,
    Sub,
    TryExcept,
    Var,
    VarDecl,
    While,
)
from .python_bridge import PythonBridge
from .slot_types import ArraySlot
from .symbol_table import SymbolTable


class _Sentinel:
    __slots__ = ('value',)

class _ReturnSentinel(_Sentinel):
    __slots__ = ('value',)
    def __init__(self, value: Any = None) -> None:
        self.value = value

class _BreakSentinel(_Sentinel):
    __slots__ = ()

class _ContinueSentinel(_Sentinel):
    __slots__ = ()

_RETURN = _ReturnSentinel()
_BREAK = _BreakSentinel()
_CONTINUE = _ContinueSentinel()

class _RaiseSentinel(_Sentinel):
    __slots__ = ('exc',)
    def __init__(self, exc: Any) -> None:
        self.exc = exc

_RAISE = _RaiseSentinel(None)

class _TailCallSentinel(_Sentinel):
    __slots__ = ('func_name', 'args', 'kwargs', 'is_method', 'instance', 'method_name')
    def __init__(self, func_name: str, args: list, kwargs: dict, is_method: bool = False, instance: Any = None, method_name: str | None = None) -> None:
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        self.is_method = is_method
        self.instance = instance
        self.method_name = method_name

def _is_control_signal(result: Any) -> bool:
    return isinstance(result, _Sentinel)


class BreakSignal(Exception):
    pass

class ContinueSignal(Exception):
    pass

class ReturnSignal(Exception):
    def __init__(self, value: Any = None) -> None:
        self.value = value



class IRInterpreter:
    """IR解释执行器（无递归版）/ IR Interpreter (recursion-free version).

    所有递归已消除 / All recursion eliminated:
    1. _eval_expr → 显式求值栈 (node, processed) + results
    2. _exec_stmt/_exec_body → 工作列表 + 控制流信号
    3. _call_func_with_frame → 显式调用栈 _CallFrame
    4. NLASMMethod.call → 同 _call_func_with_frame
    """

    MAX_CALL_DEPTH = 5000

    def __init__(self, arr_slot: ArraySlot | None = None, bridge: PythonBridge | None = None, parent_symtab: SymbolTable | None = None, project_dir: str | None = None) -> None:
        if parent_symtab is not None:
            self.symtab: SymbolTable = parent_symtab.enter_scope()
        else:
            self.symtab = SymbolTable()
        self.outputs: list[object] = []
        self.bridge: PythonBridge = bridge or PythonBridge()
        self._functions: dict[str, FuncDef] = {}
        self._call_depth: int = 0
        self._concurrency: Any = None
        self._project_dir: str | None = project_dir
        if arr_slot is not None and arr_slot.values is not None:
            self.symtab.define(arr_slot.name, list(arr_slot.values))
        required_limit = self.MAX_CALL_DEPTH * 4 + 1000
        if sys.getrecursionlimit() < required_limit:
            sys.setrecursionlimit(required_limit)

    @property
    def env(self) -> dict[str, object]:
        return {name: self.symtab.get_value(name) for name in self.symtab.all_names()}

    @env.setter
    def env(self, value: dict[str, object]) -> None:
        for k, v in value.items():
            self.symtab.define_or_set(k, v)

    def run(self, nodes: list) -> object:
        if getattr(self, '_in_large_stack_thread', False):
            return self._run_stmts(nodes)

        result_box: list[object | None] = [None]
        exc_box: list[BaseException | None] = [None]

        def _worker() -> None:
            self._in_large_stack_thread = True
            try:
                result_box[0] = self._run_stmts(nodes)
            except RecursionError as e:
                exc_box[0] = RecursionError(f"递归深度超过最大限制 {self.MAX_CALL_DEPTH}，请检查是否存在无限递归，或改用尾递归/循环实现")
            except BaseException as e:
                exc_box[0] = e
            finally:
                self._in_large_stack_thread = False

        old_size = threading.stack_size()
        try:
            threading.stack_size(64 * 1024 * 1024)
        except (ValueError, threading.ThreadError):
            return self._run_stmts(nodes)

        try:
            t = threading.Thread(target=_worker)
            t.start()
            t.join()
        finally:
            try:
                threading.stack_size(old_size if old_size > 0 else 0)
            except (ValueError, threading.ThreadError):
                pass

        if exc_box[0] is not None:
            raise exc_box[0]
        return result_box[0]

    def _run_stmts(self, stmts: list) -> object:
        if self._call_depth > self.MAX_CALL_DEPTH:
            raise RecursionError(f"递归深度超过最大限制 {self.MAX_CALL_DEPTH}，请检查是否存在无限递归，或改用尾递归/循环实现")
        result: object = None
        for stmt in stmts:
            result = self._exec_stmt_iter(stmt)
            if isinstance(result, _ReturnSentinel):
                return result
            if isinstance(result, _TailCallSentinel):
                return result
            if isinstance(result, _BreakSentinel) or isinstance(result, _ContinueSentinel):
                return result
        return result

    def _exec_stmt_iter(self, node) -> object:
        if isinstance(node, Assign):
            value = self._eval_expr_iter(node.value)
            if node.target.startswith("self."):
                attr_name = node.target[5:]
                self_obj = self.symtab.get_or_none("self")
                if isinstance(self_obj, NLASMInstance):
                    self_obj.attrs[attr_name] = value
            elif "[" in node.target:
                self._assign_index(node.target, value)
            else:
                self.symtab.define_or_set(node.target, value)
            return None

        if isinstance(node, IndexAssign):
            obj = self._eval_expr_iter(node.obj)
            idx = self._eval_expr_iter(node.index)
            value = self._eval_expr_iter(node.value)
            obj[int(idx)] = value
            return None

        if isinstance(node, VarDecl):
            value = self._eval_expr_iter(node.value)
            self.symtab.define(node.name, value, type_hint=node.type_hint)
            return None

        if isinstance(node, ImportStmt):
            self._exec_import(node)
            return None

        if isinstance(node, FuncDef):
            self._functions[node.name] = node
            self.symtab.define(node.name, node, kind="function")
            return None

        if isinstance(node, ClassDef):
            self._exec_class_def(node)
            return None

        if isinstance(node, If):
            cond = self._eval_expr_iter(node.condition)
            branch = node.body if cond else node.orelse
            for stmt in branch:
                result = self._exec_stmt_iter(stmt)
                if _is_control_signal(result):
                    return result
            return None

        if isinstance(node, While):
            while self._eval_expr_iter(node.condition):
                for stmt in node.body:
                    result = self._exec_stmt_iter(stmt)
                    if isinstance(result, _BreakSentinel):
                        return result
                    if isinstance(result, _ContinueSentinel):
                        break
                    if isinstance(result, _ReturnSentinel):
                        return result
                    if isinstance(result, _TailCallSentinel):
                        return result
            return None

        if isinstance(node, Loop):
            for _ in range(node.count):
                for stmt in node.body:
                    result = self._exec_stmt_iter(stmt)
                    if isinstance(result, _BreakSentinel):
                        return result
                    if isinstance(result, _ReturnSentinel):
                        return result
                    if isinstance(result, _TailCallSentinel):
                        return result
            return None

        if isinstance(node, For):
            iterable = self._eval_expr_iter(node.iterable)
            var_names = node.var.split(",") if "," in node.var else [node.var]
            for item in iterable:
                if len(var_names) > 1:
                    if isinstance(item, (list, tuple)) and len(item) >= len(var_names):
                        for i, vn in enumerate(var_names):
                            self.symtab.define(vn, item[i])
                    else:
                        self.symtab.define(node.var, item)
                else:
                    self.symtab.define(node.var, item)
                for stmt in node.body:
                    result = self._exec_stmt_iter(stmt)
                    if isinstance(result, _BreakSentinel):
                        return result
                    if isinstance(result, _ContinueSentinel):
                        break
                    if isinstance(result, _ReturnSentinel):
                        return result
                    if isinstance(result, _TailCallSentinel):
                        return result
            return None

        if isinstance(node, ForRange):
            start = int(self._eval_expr_iter(node.start))
            stop = int(self._eval_expr_iter(node.stop))
            step = int(self._eval_expr_iter(node.step)) if node.step else 1
            for i in range(start, stop + 1, step):
                self.symtab.define(node.var, i)
                for stmt in node.body:
                    result = self._exec_stmt_iter(stmt)
                    if isinstance(result, _BreakSentinel):
                        return result
                    if isinstance(result, _ContinueSentinel):
                        break
                    if isinstance(result, _ReturnSentinel):
                        return result
                    if isinstance(result, _TailCallSentinel):
                        return result
            return None

        if isinstance(node, Break):
            return _BREAK

        if isinstance(node, Continue):
            return _CONTINUE

        if isinstance(node, Match):
            match_val = self._eval_expr_iter(node.value)
            for case_val_expr, case_body in node.cases:
                case_val = self._eval_expr_iter(case_val_expr)
                if match_val == case_val:
                    for stmt in case_body:
                        result = self._exec_stmt_iter(stmt)
                        if _is_control_signal(result):
                            return result
                    return None
            if node.default is not None:
                for stmt in node.default:
                    result = self._exec_stmt_iter(stmt)
                    if _is_control_signal(result):
                        return result
            return None

        if isinstance(node, Print):
            if node.values:
                vals = [self._eval_expr_iter(v) for v in node.values]
                self.outputs.append(tuple(vals) if len(vals) > 1 else vals[0])
            else:
                value = self._eval_expr_iter(node.value)
                self.outputs.append(value)
            return None

        if isinstance(node, Return):
            if node.value is not None:
                if isinstance(node.value, FuncCall):
                    eval_args = [self._eval_expr_iter(a) for a in node.value.args]
                    eval_kwargs: dict[str, Any] = {}
                    for k, v in node.value.kwargs.items():
                        eval_kwargs[k] = self._eval_expr_iter(v)
                    name = node.value.name
                    if name.startswith("_attr_"):
                        method_name = name[6:]
                        instance = eval_args[0] if eval_args else None
                        method_args = eval_args[1:]
                        return _TailCallSentinel(name, method_args, eval_kwargs, is_method=True, instance=instance, method_name=method_name)
                    return _TailCallSentinel(name, eval_args, eval_kwargs)
                return _ReturnSentinel(self._eval_expr_iter(node.value))
            return _RETURN

        if isinstance(node, TryExcept):
            return self._exec_try_except_iter(node)

        if isinstance(node, Raise):
            if node.value is not None:
                exc_val = self._eval_expr_iter(node.value)
                if isinstance(exc_val, str):
                    return _RaiseSentinel(RuntimeError(exc_val))
                if isinstance(exc_val, NLASMInstance):
                    return _RaiseSentinel(NLASMException(exc_val))
                return _RaiseSentinel(exc_val)
            return _RaiseSentinel(RuntimeError("主动抛出异常"))

        if isinstance(node, FuncCall):
            self._eval_expr_iter(node)
            return None

        if isinstance(node, CallPython):
            self._eval_expr_iter(node)
            return None

        if isinstance(node, Expr):
            self._eval_expr_iter(node)
            return None

        raise TypeError(f"未知语句类型: {type(node).__name__}")

    def _exec_try_except_iter(self, node: TryExcept) -> _Sentinel | None:
        raised_exc: Any = None
        try:
            for stmt in node.body:
                result = self._exec_stmt_iter(stmt)
                if isinstance(result, _RaiseSentinel):
                    raised_exc = result.exc
                    break
                elif _is_control_signal(result):
                    return result
        except Exception as exc:
            raised_exc = exc

        if raised_exc is not None:
            e = raised_exc
            handled = False
            for exc_type, exc_name, handler_body in node.handlers:
                if exc_type is None:
                    if exc_name:
                        self.symtab.define(exc_name, e)
                    for stmt in handler_body:
                        result = self._exec_stmt_iter(stmt)
                        if _is_control_signal(result):
                            return result
                    handled = True
                    break
                if self._exception_matches(e, exc_type):
                    if exc_name:
                        self.symtab.define(exc_name, e)
                    for stmt in handler_body:
                        result = self._exec_stmt_iter(stmt)
                        if _is_control_signal(result):
                            return result
                    handled = True
                    break
            if not handled:
                raise e

        if node.finally_body is not None:
            for stmt in node.finally_body:
                result = self._exec_stmt_iter(stmt)
                if _is_control_signal(result):
                    return result
        return None

    def _exec_import(self, node: ImportStmt) -> None:
        try:
            from .module_system import ModuleLoader
            if not hasattr(self, '_module_loader'):
                project_dir = self._project_dir or os.getcwd()
                self._module_loader = ModuleLoader(
                    search_paths=[".", "./stdlib", str(Path(__file__).resolve().parent.parent / "stdlib")],
                    project_dir=project_dir,
                )
            module = self._module_loader.import_module(node.module)
            for export_name, export_val in module.exports.items():
                if isinstance(export_val, FuncDef):
                    self._functions[export_name] = export_val
                    self.symtab.define(export_name, export_val, kind="function")
            module_name = node.alias or node.module
            self.symtab.define(module_name, module, kind="module")
        except ImportError:
            self.bridge.import_module(node.module, node.alias)

    def _exec_class_def(self, node: ClassDef) -> None:
        methods: dict[str, FuncDef] = {}
        class_vars: dict[str, Any] = {}
        for stmt in node.body:
            if isinstance(stmt, FuncDef):
                methods[stmt.name] = stmt
            elif isinstance(stmt, (Assign, VarDecl)):
                if isinstance(stmt, Assign):
                    class_vars[stmt.target] = self._eval_expr_iter(stmt.value)
                elif isinstance(stmt, VarDecl):
                    class_vars[stmt.name] = self._eval_expr_iter(stmt.value)
        cls_obj = NLASMClass(name=node.name, bases=node.bases, methods=methods, class_vars=class_vars)
        self.symtab.define(node.name, cls_obj, kind="class")

    # ============================================================
    # 表达式求值 — 显式求值栈，无递归
    # Expression evaluation — explicit evaluation stack, no recursion
    # ============================================================

    def _eval_expr(self, node) -> object:
        return self._eval_expr_iter(node)

    def _eval_expr_iter(self, node) -> object:
        results: list[Any] = []
        work_stack: list[tuple] = [(node, False)]

        while work_stack:
                item = work_stack.pop()
                current = item[0]
                processed = item[1]

                if processed:
                    marker = item[2] if len(item) > 2 else None

                    if marker == "add":
                        right = results.pop()
                        left = results.pop()
                        if isinstance(left, str) or isinstance(right, str):
                            results.append(str(left) + str(right))
                        else:
                            results.append(left + right)
                    elif marker == "sub":
                        right = results.pop()
                        left = results.pop()
                        results.append(left - right)
                    elif marker == "mul":
                        right = results.pop()
                        left = results.pop()
                        results.append(left * right)
                    elif marker == "div":
                        right = results.pop()
                        left = results.pop()
                        if right == 0:
                            raise ZeroDivisionError("除零错误")
                        if isinstance(left, int) and isinstance(right, int):
                            results.append(left // right)
                        else:
                            results.append(left / right)
                    elif marker == "mod":
                        right = results.pop()
                        left = results.pop()
                        results.append(left % right)
                    elif marker == "neg":
                        results.append(-results.pop())
                    elif marker == "not":
                        results.append(not results.pop())
                    elif marker == "and":
                        right = results.pop()
                        left = results.pop()
                        results.append(left and right)
                    elif marker == "or":
                        right = results.pop()
                        left = results.pop()
                        results.append(left or right)
                    elif marker == "cmp":
                        op = item[3]
                        right = results.pop()
                        left = results.pop()
                        results.append(self._compare(left, op, right))
                    elif marker == "negate":
                        results.append(-results.pop())
                    elif marker == "index_access":
                        idx = results.pop()
                        obj = results.pop()
                        if isinstance(idx, str):
                            results.append(obj[idx])
                        else:
                            results.append(obj[int(idx)])
                    elif marker == "attr_access":
                        attr = item[3]
                        obj = results.pop()
                        results.append(self._resolve_attr(obj, attr))
                    elif marker == "list_expr":
                        n = item[3]
                        elems = results[-n:] if n > 0 else []
                        if n > 0:
                            del results[-n:]
                        results.append(elems)
                    elif marker == "dict_expr":
                        n = item[3]
                        pairs = []
                        for _ in range(n):
                            k = results.pop()
                            v = results.pop()
                            pairs.insert(0, (k, v))
                        results.append(dict(pairs))
                    elif marker == "string_concat":
                        right = results.pop()
                        left = results.pop()
                        results.append(str(left) + " " + str(right))
                    elif marker == "call_python":
                        n_args = item[3]
                        n_kwargs = item[4]
                        kw_items = []
                        for _ in range(n_kwargs):
                            k = results.pop()
                            v = results.pop()
                            kw_items.insert(0, (k, v))
                        kwargs = dict(kw_items)
                        args = results[-n_args:] if n_args > 0 else []
                        if n_args > 0:
                            del results[-n_args:]
                        module = item[5]
                        function = item[6]
                        results.append(self.bridge.call_function(module, function, *args, **kwargs))
                    elif marker == "func_call":
                        n_args = item[3]
                        n_kwargs = item[4]
                        kw_items = []
                        for _ in range(n_kwargs):
                            k = results.pop()
                            v = results.pop()
                            kw_items.insert(0, (k, v))
                        kwargs_d = dict(kw_items)
                        args = results[-n_args:] if n_args > 0 else []
                        if n_args > 0:
                            del results[-n_args:]
                        name = item[5]
                        results.append(self._eval_func_call_resolved(name, args, kwargs_d))
                    elif marker == "async_call":
                        n_args = item[3]
                        n_kwargs = item[4]
                        kw_items = []
                        for _ in range(n_kwargs):
                            k = results.pop()
                            v = results.pop()
                            kw_items.insert(0, (k, v))
                        kwargs_d = dict(kw_items)
                        args = results[-n_args:] if n_args > 0 else []
                        if n_args > 0:
                            del results[-n_args:]
                        name = item[5]
                        results.append(self._exec_async_call(name, args, kwargs_d))
                    elif marker == "await_expr":
                        task = results.pop()
                        results.append(self._exec_await(task))
                    continue

                if isinstance(current, Literal):
                    results.append(current.value)
                elif isinstance(current, Var):
                    results.append(self._resolve_var(current.name))
                elif isinstance(current, Add):
                    work_stack.append((current, True, "add"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Sub):
                    work_stack.append((current, True, "sub"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Mul):
                    work_stack.append((current, True, "mul"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Div):
                    work_stack.append((current, True, "div"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Mod):
                    work_stack.append((current, True, "mod"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Neg):
                    work_stack.append((current, True, "neg"))
                    work_stack.append((current.operand, False))
                elif isinstance(current, Not):
                    work_stack.append((current, True, "not"))
                    work_stack.append((current.operand, False))
                elif isinstance(current, And):
                    work_stack.append((current, True, "and"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Or):
                    work_stack.append((current, True, "or"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Cmp):
                    work_stack.append((current, True, "cmp", current.op))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, IndexAccess):
                    work_stack.append((current, True, "index_access"))
                    work_stack.append((current.index, False))
                    work_stack.append((current.obj, False))
                elif isinstance(current, AttributeAccess):
                    work_stack.append((current, True, "attr_access", current.attr))
                    work_stack.append((current.obj, False))
                elif isinstance(current, ListExpr):
                    n = len(current.elements)
                    work_stack.append((current, True, "list_expr", n))
                    for e in reversed(current.elements):
                        work_stack.append((e, False))
                elif isinstance(current, DictExpr):
                    n = len(current.pairs)
                    work_stack.append((current, True, "dict_expr", n))
                    for k, v in reversed(current.pairs):
                        work_stack.append((k, False))
                        work_stack.append((v, False))
                elif isinstance(current, StringConcat):
                    work_stack.append((current, True, "string_concat"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, CallPython):
                    n_args = len(current.args)
                    n_kwargs = len(current.kwargs)
                    work_stack.append((current, True, "call_python", n_args, n_kwargs, current.module, current.function))
                    for k, v in reversed(list(current.kwargs.items())):
                        work_stack.append((Literal(k), False))
                        work_stack.append((v, False))
                    for a in reversed(current.args):
                        work_stack.append((a, False))
                elif isinstance(current, FuncCall):
                    n_args = len(current.args)
                    n_kwargs = len(current.kwargs)
                    work_stack.append((current, True, "func_call", n_args, n_kwargs, current.name))
                    for k, v in reversed(list(current.kwargs.items())):
                        work_stack.append((Literal(k), False))
                        work_stack.append((v, False))
                    for a in reversed(current.args):
                        work_stack.append((a, False))
                elif isinstance(current, AsyncCall):
                    n_args = len(current.args)
                    n_kwargs = len(current.kwargs)
                    work_stack.append((current, True, "async_call", n_args, n_kwargs, current.name))
                    for k, v in reversed(list(current.kwargs.items())):
                        work_stack.append((Literal(k), False))
                        work_stack.append((v, False))
                    for a in reversed(current.args):
                        work_stack.append((a, False))
                elif isinstance(current, AwaitExpr):
                    work_stack.append((current, True, "await_expr"))
                    work_stack.append((current.task, False))
                elif isinstance(current, ParallelCall):
                    results.append(self._exec_parallel_call_direct(current))
                else:
                    raise TypeError(f"未知表达式类型: {type(current).__name__}")

        return results[0] if results else None

    def _resolve_attr(self, obj: Any, attr: str) -> Any:
        if isinstance(obj, NLSuperProxy):
            return obj.get_attr(attr)
        if isinstance(obj, NLASMInstance):
            if attr in obj.attrs:
                return obj.attrs[attr]
            if attr in obj.methods:
                return NLASMMethod(instance=obj, method_name=attr)
            method_def = obj.cls.resolve_method(attr, self)
            if method_def is not None:
                return NLASMMethod(instance=obj, method_name=attr, method_def=method_def, source_cls=obj.cls)
            return None
        from .module_system import Module
        if isinstance(obj, Module):
            if attr in obj.exports:
                return obj.exports[attr]
            # 尝试中文方法名映射 / Try Chinese method name mapping
            mapped_attr = _BUILTIN_METHOD_MAP.get(attr, attr)
            if mapped_attr in obj.exports:
                return obj.exports[mapped_attr]
            # 如果有原始 Python 模块，尝试从那里获取属性
            if obj._py_module is not None:
                try:
                    py_attr = getattr(obj._py_module, mapped_attr)
                    return py_attr
                except AttributeError:
                    pass
            raise AttributeError(f"模块 {obj.name} 没有属性: {attr}")
        if isinstance(obj, NLASMClass):
            if attr in obj.methods:
                return obj.methods[attr]
            if attr in obj.class_vars:
                return obj.class_vars[attr]
            return None
        # 尝试中文方法名映射 for Python objects
        mapped_attr = _BUILTIN_METHOD_MAP.get(attr, attr)
        try:
            return getattr(obj, mapped_attr)
        except AttributeError:
            try:
                return getattr(obj, attr)
            except AttributeError:
                return None

    def _eval_func_call_resolved(self, name: str, args: list, kwargs: dict) -> Any:
        if name.startswith("_attr_"):
            attr_name = name[6:]
            if args:
                obj = args[0]
                method_args = args[1:]
                from .module_system import Module
                if isinstance(obj, Module):
                    if attr_name in obj.exports:
                        func_def = obj.exports[attr_name]
                        if isinstance(func_def, FuncDef):
                            return self._invoke_func(func_def, method_args, kwargs)
                        elif callable(func_def):
                            return func_def(*method_args, **kwargs)
                    # 尝试中文映射
                    mapped_name = _BUILTIN_METHOD_MAP.get(attr_name, attr_name)
                    if mapped_name in obj.exports:
                        func_def = obj.exports[mapped_name]
                        if isinstance(func_def, FuncDef):
                            return self._invoke_func(func_def, method_args, kwargs)
                        elif callable(func_def):
                            return func_def(*method_args, **kwargs)
                    # 尝试从 Python 模块获取
                    if obj._py_module is not None:
                        try:
                            py_method = getattr(obj._py_module, mapped_name)
                            if callable(py_method):
                                return py_method(*method_args, **kwargs)
                        except AttributeError:
                            pass
                    raise AttributeError(f"模块 {obj.name} 没有函数: {attr_name}")
                if isinstance(obj, NLASMMethod):
                    return self._invoke_method(obj, method_args, kwargs)
                if isinstance(obj, NLASMInstance):
                    method_def = obj.methods.get(attr_name)
                    if method_def is not None:
                        method = NLASMMethod(instance=obj, method_name=attr_name, method_def=method_def, source_cls=obj.cls)
                        return self._invoke_method(method, method_args, kwargs)
                    resolved = obj.cls.resolve_method(attr_name, self)
                    if resolved is not None:
                        method = NLASMMethod(instance=obj, method_name=attr_name, method_def=resolved, source_cls=obj.cls)
                        return self._invoke_method(method, method_args, kwargs)
                    if attr_name in obj.attrs:
                        attr_val = obj.attrs[attr_name]
                        if callable(attr_val):
                            return attr_val(*method_args, **kwargs)
                    raise AttributeError(f"{obj.cls.name} 没有方法: {attr_name}")
                if isinstance(obj, NLSuperProxy):
                    parent_method = obj.get_attr(attr_name)
                    if isinstance(parent_method, NLASMMethod):
                        return self._invoke_method(parent_method, method_args, kwargs)
                    if callable(parent_method):
                        return parent_method(*method_args, **kwargs)
                py_method_name = _BUILTIN_METHOD_MAP.get(attr_name, attr_name)
                if py_method_name == "len" and hasattr(obj, "__len__"):
                    return len(obj)
                if py_method_name == "__len__" and hasattr(obj, "__len__"):
                    return obj.__len__()
                if hasattr(obj, py_method_name):
                    method = getattr(obj, py_method_name)
                    if callable(method):
                        return method(*method_args, **kwargs)

        if name == "super":
            return self._handle_super_call(args)

        func_def = self._functions.get(name)
        if func_def is not None:
            return self._invoke_func(func_def, args, kwargs)

        cls_val = self.symtab.get_or_none(name)
        if isinstance(cls_val, NLASMClass):
            return cls_val.instantiate(args, kwargs, self)

        builtin = _BUILTINS.get(name)
        if builtin is not None:
            return builtin(*args, **kwargs)

        # 检查中文映射 / Check Chinese mapping
        mapped_name = _BUILTIN_METHOD_MAP.get(name, name)
        builtin = _BUILTINS.get(mapped_name)
        if builtin is not None:
            return builtin(*args, **kwargs)

        # 检查变量是否是可调用对象
        var_val = self.symtab.get_or_none(name)
        if var_val is not None and callable(var_val):
            return var_val(*args, **kwargs)

        raise NameError(f"未定义函数: {name}")

    def _get_concurrency(self):
        if self._concurrency is None:
            from .concurrency import NLASMConcurrency
            self._concurrency = NLASMConcurrency(max_workers=4)
        return self._concurrency

    def _exec_async_call(self, name: str, args: list, kwargs: dict) -> Any:
        conc = self._get_concurrency()
        return conc.run_async(name, args, kwargs, self)

    def _exec_await(self, task: Any) -> Any:
        from .concurrency import AsyncTask
        if isinstance(task, AsyncTask):
            return task.result()
        return task

    def _exec_parallel_call_direct(self, node: ParallelCall) -> list:
        conc = self._get_concurrency()
        calls: list[tuple[str, list, dict]] = []
        for call in node.calls:
            args = [self._eval_expr_iter(a) for a in call.args]
            kwargs = {k: self._eval_expr_iter(v) for k, v in call.kwargs.items()}
            calls.append((call.name, args, kwargs))
        return conc.run_parallel(calls, self)

    def _invoke_func(self, func_def: FuncDef, args: list, kwargs: dict) -> Any:
        self._call_depth += 1
        if self._call_depth > self.MAX_CALL_DEPTH:
            self._call_depth -= 1
            raise RecursionError(f"递归深度超过最大限制 {self.MAX_CALL_DEPTH}，请检查是否存在无限递归，或改用尾递归/循环实现")

        saved_symtab = self.symtab

        try:
            while True:
                child_symtab = SymbolTable(parent=saved_symtab)

                positional_consumed = 0
                for i, (pname, _ptype) in enumerate(func_def.params):
                    if i < len(args):
                        child_symtab.define(pname, args[i])
                        positional_consumed += 1
                    elif pname in kwargs:
                        child_symtab.define(pname, kwargs[pname])
                    elif func_def.defaults and pname in func_def.defaults:
                        default_val = self._eval_expr_iter(func_def.defaults[pname])
                        child_symtab.define(pname, default_val)

                if func_def.variadic is not None:
                    child_symtab.define(func_def.variadic, list(args[positional_consumed:]))

                param_names = {p[0] for p in func_def.params}
                for k, v in kwargs.items():
                    if k not in param_names:
                        child_symtab.define(k, v)

                self.symtab = child_symtab
                result = self._run_stmts(func_def.body)

                if isinstance(result, _TailCallSentinel) and not result.is_method:
                    tc_func_def = self._functions.get(result.func_name)
                    if tc_func_def is func_def:
                        args = result.args
                        kwargs = result.kwargs
                        continue
                    self.symtab = saved_symtab
                    return self._eval_func_call_resolved(result.func_name, result.args, result.kwargs)

                self.symtab = saved_symtab
                if isinstance(result, _TailCallSentinel):
                    return self._eval_func_call_resolved(result.func_name, [result.instance] + result.args if result.is_method else result.args, result.kwargs)
                if isinstance(result, _ReturnSentinel):
                    return result.value
                return result
        finally:
            self._call_depth -= 1

    def _invoke_method(self, method: NLASMMethod, args: list, kwargs: dict) -> Any:
        self._call_depth += 1
        if self._call_depth > self.MAX_CALL_DEPTH:
            self._call_depth -= 1
            raise RecursionError(f"递归深度超过最大限制 {self.MAX_CALL_DEPTH}，请检查是否存在无限递归，或改用尾递归/循环实现")

        if method.method_def is not None:
            method_def = method.method_def
        else:
            method_def = method.instance.methods.get(method.method_name)
        if method_def is None:
            self._call_depth -= 1
            raise NameError(f"未定义方法: {method.method_name}")

        saved_symtab = self.symtab

        try:
            while True:
                child_symtab = SymbolTable(parent=saved_symtab)
                child_symtab.define("self", method.instance)
                child_symtab.define("__class__", method.source_cls if method.source_cls is not None else method.instance.cls)

                arg_idx = 0
                positional_consumed = 0
                for pname, _ptype in method_def.params:
                    if pname == "self":
                        continue
                    if arg_idx < len(args):
                        child_symtab.define(pname, args[arg_idx])
                        arg_idx += 1
                        positional_consumed += 1
                    elif pname in kwargs:
                        child_symtab.define(pname, kwargs[pname])
                    elif method_def.defaults and pname in method_def.defaults:
                        default_val = self._eval_expr_iter(method_def.defaults[pname])
                        child_symtab.define(pname, default_val)

                if method_def.variadic is not None:
                    child_symtab.define(method_def.variadic, list(args[positional_consumed:]))

                param_names = {p[0] for p in method_def.params}
                for k, v in kwargs.items():
                    if k not in param_names:
                        child_symtab.define(k, v)

                self.symtab = child_symtab
                result = self._run_stmts(method_def.body)

                if isinstance(result, _TailCallSentinel):
                    if result.is_method and result.instance is method.instance:
                        tc_method_def = method.instance.cls.resolve_method(result.method_name, self) if result.method_name else None
                        if tc_method_def is method_def:
                            args = result.args
                            kwargs = result.kwargs
                            continue
                        if tc_method_def is not None:
                            self.symtab = saved_symtab
                            new_method = NLASMMethod(instance=method.instance, method_name=result.method_name, method_def=tc_method_def, source_cls=method.instance.cls)
                            return self._invoke_method(new_method, result.args, result.kwargs)
                    self.symtab = saved_symtab
                    return self._eval_func_call_resolved(result.func_name, [result.instance] + result.args if result.is_method else result.args, result.kwargs)

                self.symtab = saved_symtab
                if isinstance(result, _ReturnSentinel):
                    return result.value
                return result
        finally:
            self._call_depth -= 1

    def _handle_super_call(self, args: list) -> Any:
        self_obj = self.symtab.get_or_none("self")
        current_cls = self.symtab.get_or_none("__class__")
        if not isinstance(self_obj, NLASMInstance) or not isinstance(current_cls, NLASMClass):
            raise NameError("super 只能在类方法中使用")
        mro = current_cls.get_mro(self)
        if len(mro) < 2:
            return None
        parent_cls = mro[1]
        init_method = parent_cls.methods.get("初始化") or parent_cls.methods.get("__init__")
        if init_method is None:
            return None
        method = NLASMMethod(instance=self_obj, method_name="初始化", method_def=init_method, source_cls=parent_cls)
        return self._invoke_method(method, args, {})

    def _compare(self, left: object, op: str, right: object) -> bool:
        if op == ">": return left > right
        if op == ">=": return left >= right
        if op == "<": return left < right
        if op == "<=": return left <= right
        if op == "==": return left == right
        if op == "!=": return left != right
        raise ValueError(f"未知比较运算符: {op}")

    def _resolve_var(self, name: str) -> object:
        if name == "super":
            return self._resolve_super()
        if name.startswith("self."):
            attr_name = name[5:]
            self_obj = self.symtab.get_or_none("self")
            if isinstance(self_obj, NLASMInstance):
                if attr_name in self_obj.attrs:
                    return self_obj.attrs[attr_name]
                if attr_name in self_obj.methods:
                    return NLASMMethod(instance=self_obj, method_name=attr_name)
                method_def = self_obj.cls.resolve_method(attr_name, self)
                if method_def is not None:
                    return NLASMMethod(instance=self_obj, method_name=attr_name, method_def=method_def, source_cls=self_obj.cls)
                return None
            raise NameError(f"未定义属性: {name}")
        if "[" in name:
            return self._resolve_index(name)
        val = self.symtab.get_or_none(name)
        if val is not None:
            return val
        func_def = self._functions.get(name)
        if func_def is not None:
            return func_def
        # 检查内置函数 / Check built-in functions
        mapped_name = _BUILTIN_METHOD_MAP.get(name, name)
        if mapped_name in _BUILTINS:
            return _BUILTINS[mapped_name]
        raise NameError(f"未定义变量: {name}")

    def _resolve_super(self) -> NLSuperProxy:
        self_obj = self.symtab.get_or_none("self")
        current_cls = self.symtab.get_or_none("__class__")
        if not isinstance(self_obj, NLASMInstance) or not isinstance(current_cls, NLASMClass):
            raise NameError("super 只能在类方法中使用")
        return NLSuperProxy(instance=self_obj, current_cls=current_cls, interp=self)

    def _resolve_index(self, name: str) -> object:
        base, idx_expr = name.split("[", 1)
        idx_expr = idx_expr.rstrip("]")
        idx_val = self.symtab.get_or_none(idx_expr)
        idx = int(idx_val) if idx_val is not None else int(idx_expr)
        arr = self.symtab.get_or_none(base)
        if arr is None:
            raise NameError(f"未定义数组: {base}")
        return arr[idx]

    def _assign_index(self, name: str, value: object) -> None:
        base, idx_expr = name.split("[", 1)
        idx_expr = idx_expr.rstrip("]")
        idx_val = self.symtab.get_or_none(idx_expr)
        idx = int(idx_val) if idx_val is not None else int(idx_expr)
        arr = self.symtab.get_or_none(base)
        if arr is None:
            raise NameError(f"未定义数组: {base}")
        arr[idx] = value

    def _exception_matches(self, exc: Exception, exc_type: str) -> bool:
        exc_cls = type(exc)
        if exc_cls.__name__ == exc_type:
            return True
        cls_obj = self.symtab.get_or_none(exc_type)
        if isinstance(cls_obj, NLASMClass):
            if isinstance(exc, NLASMException):
                return exc.is_instance_of(exc_type, self)
        for parent_cls in exc_cls.__mro__:
            if parent_cls.__name__ == exc_type:
                return True
        return False


from .builtins import BUILTINS as _BUILTIN_REGISTRY

_BUILTINS: dict[str, Any] = dict(_BUILTIN_REGISTRY.functions)

_BUILTIN_METHOD_MAP: dict[str, str] = {
    "追加": "append", "扩展": "extend", "插入": "insert",
    "删除": "remove", "弹出": "pop", "清空": "clear",
    "索引": "index", "计数": "count", "排序": "sort",
    "反转": "reverse", "拷贝": "copy", "获取": "get",
    "键": "keys", "值": "values", "项": "items",
    "更新": "update", "设置默认": "setdefault", "弹出项": "popitem",
    "分割": "split", "连接": "join", "替换": "replace",
    "去空格": "strip", "去左空格": "lstrip", "去右空格": "rstrip",
    "大写": "upper", "小写": "lower", "首字母大写": "capitalize",
    "标题化": "title", "开头": "startswith", "结尾": "endswith",
    "查找": "find", "右查找": "rfind", "是否数字": "isdigit",
    "是否字母": "isalpha", "是否字母数字": "isalnum", "是否空白": "isspace",
    "居中": "center", "左对齐": "ljust", "右对齐": "rjust",
    "补零": "zfill", "添加": "append", "丢弃": "discard",
    "并集": "union", "交集": "intersection", "差集": "difference",
    "放": "append", "加入": "append", "加入开头": "insert",
    "添加": "append", "尾部添加": "append", "末尾添加": "append",
    "长度": "len", "字符": "__getitem__", "取": "get",
    "查": "get", "设": "__setitem__", "有没有": "__contains__",
    "存在": "__contains__", "第几个": "index", "首个": "__iter__",
    "末尾": "end", "首": "start", "最后一个": "pop",
    "为空": "__bool__", "非空": "__bool__", "为空吗": "__bool__",
    "首字符": "__getitem__", "末字符": "__getitem__", "几个": "len",
    "全清": "clear", "清空": "clear", "是否存在": "__contains__",
    "复制": "copy", "深拷贝": "deepcopy", "长度": "len",
    # math 模块方法
    "开方": "sqrt", "开根号": "sqrt", "平方根": "sqrt",
    "正弦": "sin", "余弦": "cos", "正切": "tan",
    "反正弦": "asin", "反余弦": "acos", "反正切": "atan",
    "正弦h": "sinh", "余弦h": "cosh", "正切h": "tanh",
    "绝对值": "abs", "绝对": "abs", "向上取整": "ceil", "天花板": "ceil",
    "向下取整": "floor", "地板": "floor", "阶乘": "factorial",
    "幂": "pow", "次方": "pow", "指数": "exp", "对数": "log",
    "自然对数": "log", "常用对数": "log10", "平方": "square",
    "立方": "cube", "距离": "hypot", "角度转弧度": "radians",
    "弧度转角度": "degrees", "取整": "trunc", "最大": "max", "最小": "min",
    "求和": "sum", "圆周率": "pi", "自然常数": "e",
    # datetime 模块方法
    "now": "now", "今天": "today", "日期": "date", "时间": "time",
    "年": "year", "月": "month", "日": "day", "时": "hour",
    "分": "minute", "秒": "second", "微秒": "microsecond",
    "日期时间": "datetime", "现在": "now",
    # random 模块方法
    "随机": "random", "随机整数": "randint", "随机浮点": "uniform",
    "选择": "choice", "打乱": "shuffle", "样本": "sample",
    # os 模块方法
    "获取当前目录": "getcwd", "列出文件": "listdir", "路径存在": "exists",
    "是否为文件": "isfile", "是否为目录": "isdir", "获取环境变量": "getenv",
    # platform 模块方法
    "处理器": "processor", "系统": "system", "版本": "version",
    "机器": "machine", "平台": "platform", "发布": "release",
    # collections 模块方法
    "有序字典": "OrderedDict", "计数器": "Counter", "默认字典": "defaultdict",
    # itertools 模块方法
    "计数": "count", "循环": "cycle", "重复": "repeat",
    "链": "chain", "分组": "groupby", "islice": "islice",
    # functools 模块方法
    "偏函数": "partial", "修饰": "wraps", "约简": "reduce",
    # urllib 模块方法
    "打开": "urlopen", "请求": "Request", ".urlretrieve": "urlretrieve",
    # re 模块方法
    "编译": "compile", "匹配": "match", "搜索": "search",
    "全文搜索": "search", "替换": "sub", "分割": "split",
    "分组": "group", "分组数": "groups", "起始位置": "span",
    "匹配开始": "start", "匹配结束": "end",
    # hashlib 模块方法
    "摘要": "digest", "更新": "update", "十六进制摘要": "hexdigest",
    "复制": "copy", "摘要大小": "digest_size", "块大小": "block_size",
    "新建": "new", "sha256": "sha256", "sha1": "sha1", "md5": "md5",
    # 内置方法
    "编码": "encode", "解码": "decode", "下一个": "next", "打印": "print",
}


class NLASMClass:
    """NLASM类对象 — 支持继承和方法解析 / NLASM class object — supports inheritance and method resolution"""

    def __init__(self, name: str, bases: list[str], methods: dict[str, FuncDef], class_vars: dict[str, Any]) -> None:
        self.name = name
        self.bases = bases
        self.methods = methods
        self.class_vars = class_vars
        self._mro: list[NLASMClass] | None = None

    def resolve_method(self, name: str, interp: IRInterpreter) -> FuncDef | None:
        for cls in self.get_mro(interp):
            if name in cls.methods:
                return cls.methods[name]
        return None

    def resolve_class_var(self, name: str, interp: IRInterpreter) -> Any:
        for cls in self.get_mro(interp):
            if name in cls.class_vars:
                return cls.class_vars[name]
        return None

    def get_mro(self, interp: IRInterpreter) -> list[NLASMClass]:
        if self._mro is not None:
            return self._mro

        mro = [self]
        work_stack: list[str] = list(self.bases)
        while work_stack:
            base_name = work_stack.pop()
            base_cls = interp.symtab.get_or_none(base_name)
            if isinstance(base_cls, NLASMClass):
                if base_cls not in mro:
                    mro.append(base_cls)
                    work_stack.extend(base_cls.bases)
        self._mro = mro
        return mro

    def instantiate(self, args: list, kwargs: dict, interp: IRInterpreter) -> NLASMInstance:
        merged_methods: dict[str, FuncDef] = {}
        merged_class_vars: dict[str, Any] = {}

        for cls in reversed(self.get_mro(interp)):
            merged_methods.update(cls.methods)
            merged_class_vars.update(cls.class_vars)

        instance = NLASMInstance(cls=self, attrs=dict(merged_class_vars), methods=merged_methods)
        init_method = merged_methods.get("初始化") or merged_methods.get("__init__")
        if init_method:
            interp._call_depth += 1
            if interp._call_depth > IRInterpreter.MAX_CALL_DEPTH:
                interp._call_depth -= 1
                raise RecursionError(f"递归深度超过最大限制 {IRInterpreter.MAX_CALL_DEPTH}，请检查是否存在无限递归，或改用尾递归/循环实现")
            try:
                saved_symtab = interp.symtab
                child_symtab = SymbolTable(parent=saved_symtab)
                child_symtab.define("self", instance)
                child_symtab.define("__class__", self)
                arg_idx = 0
                positional_consumed = 0
                for pname, _ptype in init_method.params:
                    if pname == "self":
                        continue
                    if arg_idx < len(args):
                        child_symtab.define(pname, args[arg_idx])
                        arg_idx += 1
                        positional_consumed += 1
                    elif pname in kwargs:
                        child_symtab.define(pname, kwargs[pname])
                    elif init_method.defaults and pname in init_method.defaults:
                        default_val = interp._eval_expr_iter(init_method.defaults[pname])
                        child_symtab.define(pname, default_val)
                if init_method.variadic is not None:
                    child_symtab.define(init_method.variadic, list(args[positional_consumed:]))
                for k, v in kwargs.items():
                    param_names = {p[0] for p in init_method.params}
                    if k not in param_names:
                        child_symtab.define(k, v)
                interp.symtab = child_symtab
                for stmt in init_method.body:
                    result = interp._exec_stmt_iter(stmt)
                    if isinstance(result, _ReturnSentinel):
                        break
                    if isinstance(result, _TailCallSentinel):
                        break
                interp.symtab = saved_symtab
            finally:
                interp._call_depth -= 1
        return instance

    def __repr__(self) -> str:
        return f"<class {self.name}>"


class NLASMInstance:
    """NLASM类实例 / NLASM class instance"""

    def __init__(self, cls: NLASMClass, attrs: dict[str, Any] | None = None, methods: dict[str, FuncDef] | None = None) -> None:
        self.cls = cls
        self.attrs = attrs or {}
        self.methods = methods if methods is not None else cls.methods

    def __repr__(self) -> str:
        return f"<{self.cls.name} instance>"

    def get_attr(self, name: str) -> Any:
        if name in self.attrs:
            return self.attrs[name]
        if name in self.methods:
            return NLASMMethod(instance=self, method_name=name)
        raise AttributeError(f"{self.cls.name} 没有属性: {name}")

    def set_attr(self, name: str, value: Any) -> None:
        self.attrs[name] = value


class NLASMMethod:
    """NLASM方法绑定 — 支持默认参数和可变参数 / NLASM method binding — supports default and variadic params"""

    def __init__(self, instance: NLASMInstance, method_name: str, method_def: FuncDef | None = None, source_cls: NLASMClass | None = None) -> None:
        self.instance = instance
        self.method_name = method_name
        self.method_def = method_def
        self.source_cls = source_cls

    def call(self, args: list, kwargs: dict, interp: IRInterpreter) -> Any:
        return interp._invoke_method(self, args, kwargs)


class NLSuperProxy:
    """super代理 — 支持super.方法名()调用父类方法 / super proxy — supports super.method() calls to parent class"""

    def __init__(self, instance: NLASMInstance, current_cls: NLASMClass, interp: IRInterpreter) -> None:
        self.instance = instance
        self.current_cls = current_cls
        self.interp = interp

    def get_attr(self, name: str) -> Any:
        mro = self.current_cls.get_mro(self.interp)
        if len(mro) < 2:
            raise AttributeError(f"没有父类")
        parent_cls = mro[1]
        method_def = parent_cls.resolve_method(name, self.interp)
        if method_def is not None:
            return NLASMMethod(instance=self.instance, method_name=name, method_def=method_def, source_cls=parent_cls)
        class_var = parent_cls.resolve_class_var(name, self.interp)
        if class_var is not None:
            return class_var
        raise AttributeError(f"父类没有属性: {name}")


class NLASMException(Exception):
    """NLASM自定义异常 — 包装类实例为Python异常 / NLASM custom exception — wraps class instance as Python exception"""

    def __init__(self, instance: NLASMInstance) -> None:
        self.instance = instance
        msg = str(instance.attrs.get("消息", instance.attrs.get("message", str(instance))))
        super().__init__(msg)

    def is_instance_of(self, type_name: str, interp: IRInterpreter) -> bool:
        cls = self.instance.cls
        while cls is not None:
            if cls.name == type_name:
                return True
            if not cls.bases:
                break
            base_name = cls.bases[0]
            base_cls = interp.symtab.get_or_none(base_name)
            if isinstance(base_cls, NLASMClass):
                cls = base_cls
            else:
                break
        return False

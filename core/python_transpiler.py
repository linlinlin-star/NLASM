from __future__ import annotations

from .ir import (
    Add,
    And,
    Assign,
    AttributeAccess,
    Break,
    CallPython,
    ClassDef,
    Cmp,
    Continue,
    Div,
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
    Print,
    Raise,
    Return,
    StringConcat,
    Sub,
    TryExcept,
    Var,
    VarDecl,
    While,
)


class PythonTranspiler:
    """Python转译器 - 将NLASM IR节点转译为Python源代码 / Python transpiler - transpiles NLASM IR nodes to Python source code.

    将中文关键字映射为Python关键字，保持语义等价。
    Maps Chinese keywords to Python keywords, maintaining semantic equivalence.
    """

    def __init__(self) -> None:
        self._indent: int = 0  # 当前缩进级别 / Current indentation level

    def transpile(self, stmts: list) -> str:
        """转译IR语句列表为Python代码 / Transpile IR statement list to Python code"""
        lines: list[str] = []
        for stmt in stmts:
            lines.append(self._transpile_stmt(stmt))
        return "\n".join(lines) + "\n"

    def _pad(self) -> str:
        """生成缩进字符串 / Generate indentation string"""
        return "    " * self._indent

    def _transpile_stmt(self, node) -> str:
        results: list[str] = []
        work_stack: list[tuple] = [(node, False)]

        while work_stack:
            item = work_stack.pop()
            current = item[0]
            processed = item[1]

            if processed:
                marker = item[2] if len(item) > 2 else None

                if marker == "var_decl":
                    val = results.pop()
                    name = item[3]
                    type_hint = item[4]
                    if type_hint:
                        results.append(f"{self._pad()}{name}: {type_hint} = {val}")
                    else:
                        results.append(f"{self._pad()}{name} = {val}")

                elif marker == "assign":
                    val = results.pop()
                    target = item[3]
                    results.append(f"{self._pad()}{target} = {val}")

                elif marker == "index_assign":
                    val = results.pop()
                    idx = results.pop()
                    obj = results.pop()
                    results.append(f"{self._pad()}{obj}[{idx}] = {val}")

                elif marker == "func_def":
                    n_body = item[3]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    name = item[4]
                    params_str = item[5]
                    ret = item[6]
                    self._indent -= 1
                    header = f"{self._pad()}def {name}({params_str}){ret}:"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "class_def":
                    n_body = item[3]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    name = item[4]
                    bases_str = item[5]
                    self._indent -= 1
                    header = f"{self._pad()}class {name}{bases_str}:"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "if":
                    n_else = item[3]
                    n_body = item[4]
                    else_lines = results[-n_else:] if n_else > 0 else []
                    if n_else > 0:
                        del results[-n_else:]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    cond = results.pop()
                    self._indent -= 1
                    header = f"{self._pad()}if {cond}:"
                    result = header + "\n" + "\n".join(body_lines)
                    if else_lines:
                        self._indent -= 1
                        result += "\n" + f"{self._pad()}else:" + "\n" + "\n".join(else_lines)
                    results.append(result)

                elif marker == "while":
                    n_body = item[3]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    cond = results.pop()
                    self._indent -= 1
                    header = f"{self._pad()}while {cond}:"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "loop":
                    n_body = item[3]
                    count = item[4]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    self._indent -= 1
                    header = f"{self._pad()}for _ in range({count}):"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "for":
                    n_body = item[3]
                    var = item[4]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    iter_expr = results.pop()
                    self._indent -= 1
                    header = f"{self._pad()}for {var} in {iter_expr}:"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "for_range":
                    n_body = item[3]
                    var = item[4]
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    step_val = results.pop() if item[5] else None
                    stop_val = results.pop()
                    start_val = results.pop()
                    if step_val:
                        range_expr = f"range({start_val}, {stop_val}, {step_val})"
                    else:
                        range_expr = f"range({start_val}, {stop_val})"
                    self._indent -= 1
                    header = f"{self._pad()}for {var} in {range_expr}:"
                    results.append(header + "\n" + "\n".join(body_lines))

                elif marker == "match":
                    n_default = item[3]
                    n_cases = item[4]
                    all_lines = []
                    if n_default > 0:
                        default_lines = results[-n_default:]
                        del results[-n_default:]
                    else:
                        default_lines = []
                    case_data = []
                    for _ in range(n_cases):
                        n_cb = results.pop()
                        n_cv = results.pop()
                        case_data.insert(0, (n_cv, n_cb))
                    for n_cv, n_cb in case_data:
                        cb_lines = results[-n_cb:] if n_cb > 0 else [f"{self._pad()}pass"]
                        if n_cb > 0:
                            del results[-n_cb:]
                        cv = results.pop()
                        all_lines.append(f"{self._pad()}case {cv}:")
                        self._indent += 1
                        all_lines.extend(cb_lines if n_cb > 0 else [f"{self._pad()}pass"])
                        self._indent -= 1
                    val = results.pop()
                    all_lines.insert(0, f"{self._pad()}match {val}:")
                    if default_lines:
                        all_lines.append(f"{self._pad()}case _:")
                        self._indent += 1
                        all_lines.extend(default_lines if n_default > 0 else [f"{self._pad()}pass"])
                        self._indent -= 1
                    self._indent -= 1
                    results.append("\n".join(all_lines))

                elif marker == "print":
                    val = results.pop()
                    results.append(f"{self._pad()}print({val})")

                elif marker == "print_multi":
                    n = item[3]
                    args = ", ".join(results[-n:]) if n > 0 else ""
                    if n > 0:
                        del results[-n:]
                    results.append(f"{self._pad()}print({args})")

                elif marker == "return":
                    val = results.pop()
                    results.append(f"{self._pad()}return {val}")

                elif marker == "raise":
                    val = results.pop()
                    results.append(f"{self._pad()}raise {val}")

                elif marker == "try_except":
                    n_finally = item[3]
                    n_handlers = item[4]
                    n_body = item[5]
                    finally_lines = []
                    if n_finally > 0:
                        finally_lines = results[-n_finally:]
                        del results[-n_finally:]
                    handler_data = []
                    for _ in range(n_handlers):
                        n_hb = results.pop()
                        exc_name = results.pop()
                        exc_type = results.pop()
                        handler_data.insert(0, (exc_type, exc_name, n_hb))
                    body_lines = results[-n_body:] if n_body > 0 else [f"{self._pad()}pass"]
                    if n_body > 0:
                        del results[-n_body:]
                    lines = [f"{self._pad()}try:"]
                    self._indent += 1
                    lines.extend(body_lines if n_body > 0 else [f"{self._pad()}pass"])
                    self._indent -= 1
                    for exc_type, exc_name, n_hb in handler_data:
                        hb_lines = results[-n_hb:] if n_hb > 0 else [f"{self._pad()}pass"]
                        if n_hb > 0:
                            del results[-n_hb:]
                        if exc_type and exc_name:
                            lines.append(f"{self._pad()}except {exc_type} as {exc_name}:")
                        elif exc_type:
                            lines.append(f"{self._pad()}except {exc_type}:")
                        else:
                            lines.append(f"{self._pad()}except Exception:")
                        self._indent += 1
                        lines.extend(hb_lines if n_hb > 0 else [f"{self._pad()}pass"])
                        self._indent -= 1
                    if n_finally >= 0:
                        lines.append(f"{self._pad()}finally:")
                        self._indent += 1
                        lines.extend(finally_lines if finally_lines else [f"{self._pad()}pass"])
                        self._indent -= 1
                    results.append("\n".join(lines))

                elif marker == "expr_stmt":
                    val = results.pop()
                    results.append(f"{self._pad()}{val}")

                continue

            if isinstance(current, VarDecl):
                work_stack.append((current, True, "var_decl", current.name, current.type_hint))
                work_stack.append((current.value, False))
            elif isinstance(current, Assign):
                work_stack.append((current, True, "assign", current.target))
                work_stack.append((current.value, False))
            elif isinstance(current, IndexAssign):
                work_stack.append((current, True, "index_assign"))
                work_stack.append((current.value, False))
                work_stack.append((current.index, False))
                work_stack.append((current.obj, False))
            elif isinstance(current, ImportStmt):
                if current.items:
                    items = ", ".join(current.items)
                    results.append(f"{self._pad()}from {current.module} import {items}")
                else:
                    alias = f" as {current.alias}" if current.alias else ""
                    results.append(f"{self._pad()}import {current.module}{alias}")
            elif isinstance(current, FuncDef):
                params = ", ".join(p[0] + (f": {p[1]}" if p[1] else "") for p in current.params)
                ret = f" -> {current.return_type}" if current.return_type else ""
                self._indent += 1
                n_body = len(current.body)
                work_stack.append((current, True, "func_def", n_body, current.name, params, ret))
                for s in reversed(current.body):
                    work_stack.append((s, False))
            elif isinstance(current, ClassDef):
                bases_str = f"({', '.join(current.bases)})" if current.bases else ""
                self._indent += 1
                n_body = len(current.body)
                work_stack.append((current, True, "class_def", n_body, current.name, bases_str))
                for s in reversed(current.body):
                    work_stack.append((s, False))
            elif isinstance(current, If):
                self._indent += 1
                n_body = len(current.body)
                n_else = len(current.orelse)
                if n_else > 0:
                    self._indent += 1
                work_stack.append((current, True, "if", n_else, n_body))
                for s in reversed(current.orelse):
                    work_stack.append((s, False))
                for s in reversed(current.body):
                    work_stack.append((s, False))
                work_stack.append((current.condition, False))
            elif isinstance(current, While):
                self._indent += 1
                n_body = len(current.body)
                work_stack.append((current, True, "while", n_body))
                for s in reversed(current.body):
                    work_stack.append((s, False))
                work_stack.append((current.condition, False))
            elif isinstance(current, Loop):
                self._indent += 1
                n_body = len(current.body)
                work_stack.append((current, True, "loop", n_body, current.count))
                for s in reversed(current.body):
                    work_stack.append((s, False))
            elif isinstance(current, For):
                self._indent += 1
                n_body = len(current.body)
                work_stack.append((current, True, "for", n_body, current.var))
                for s in reversed(current.body):
                    work_stack.append((s, False))
                work_stack.append((current.iterable, False))
            elif isinstance(current, ForRange):
                self._indent += 1
                n_body = len(current.body)
                has_step = current.step is not None
                work_stack.append((current, True, "for_range", n_body, current.var, has_step))
                for s in reversed(current.body):
                    work_stack.append((s, False))
                if current.step:
                    work_stack.append((current.step, False))
                work_stack.append((current.stop, False))
                work_stack.append((current.start, False))
            elif isinstance(current, Break):
                results.append(f"{self._pad()}break")
            elif isinstance(current, Continue):
                results.append(f"{self._pad()}continue")
            elif isinstance(current, Match):
                self._indent += 1
                n_cases = len(current.cases)
                n_default = len(current.default) if current.default else 0
                work_stack.append((current, True, "match", n_default, n_cases))
                if current.default:
                    for s in reversed(current.default):
                        work_stack.append((s, False))
                for cv, cb in reversed(current.cases):
                    work_stack.append((Literal(len(cb)), False))
                    work_stack.append((Literal(len(cb)), False))
                    for s in reversed(cb):
                        work_stack.append((s, False))
                    work_stack.append((cv, False))
                work_stack.append((current.value, False))
            elif isinstance(current, Print):
                if current.values:
                    n = len(current.values)
                    work_stack.append((current, True, "print_multi", n))
                    for v in reversed(current.values):
                        work_stack.append((v, False))
                else:
                    work_stack.append((current, True, "print"))
                    work_stack.append((current.value, False))
            elif isinstance(current, Return):
                if current.value is not None:
                    work_stack.append((current, True, "return"))
                    work_stack.append((current.value, False))
                else:
                    results.append(f"{self._pad()}return")
            elif isinstance(current, TryExcept):
                n_body = len(current.body)
                n_handlers = len(current.handlers)
                n_finally = len(current.finally_body) if current.finally_body is not None else -1
                self._indent += 1
                work_stack.append((current, True, "try_except", n_finally, n_handlers, n_body))
                if current.finally_body is not None:
                    for s in reversed(current.finally_body):
                        work_stack.append((s, False))
                for exc_type, exc_name, handler_body in reversed(current.handlers):
                    work_stack.append((Literal(len(handler_body)), False))
                    work_stack.append((Literal(exc_name or ""), False))
                    work_stack.append((Literal(exc_type or ""), False))
                    for s in reversed(handler_body):
                        work_stack.append((s, False))
                for s in reversed(current.body):
                    work_stack.append((s, False))
            elif isinstance(current, Raise):
                if current.value is not None:
                    work_stack.append((current, True, "raise"))
                    work_stack.append((current.value, False))
                else:
                    results.append(f"{self._pad()}raise")
            elif isinstance(current, (FuncCall, CallPython)):
                work_stack.append((current, True, "expr_stmt"))
                work_stack.append((current, False))
            else:
                work_stack.append((current, True, "expr_stmt"))
                work_stack.append((current, False))

        return results[0] if results else ""

    def _transpile_expr(self, node) -> str:
        results: list[str] = []
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
                    results.append(f"({left} + {right})")
                elif marker == "sub":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} - {right})")
                elif marker == "mul":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} * {right})")
                elif marker == "div":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} // {right})")
                elif marker == "mod":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} % {right})")
                elif marker == "neg":
                    operand = results.pop()
                    results.append(f"(-{operand})")
                elif marker == "and":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} and {right})")
                elif marker == "or":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} or {right})")
                elif marker == "not":
                    operand = results.pop()
                    results.append(f"(not {operand})")
                elif marker == "cmp":
                    op = item[3]
                    right = results.pop()
                    left = results.pop()
                    results.append(f"({left} {op} {right})")
                elif marker == "index_access":
                    idx = results.pop()
                    obj = results.pop()
                    results.append(f"{obj}[{idx}]")
                elif marker == "attr_access":
                    attr = item[3]
                    obj = results.pop()
                    results.append(f"{obj}.{attr}")
                elif marker == "list_expr":
                    n = item[3]
                    items = ", ".join(results[-n:]) if n > 0 else ""
                    if n > 0:
                        del results[-n:]
                    results.append(f"[{items}]")
                elif marker == "func_call":
                    n = item[3]
                    args = ", ".join(results[-n:]) if n > 0 else ""
                    if n > 0:
                        del results[-n:]
                    name = item[4]
                    results.append(f"{name}({args})")
                elif marker == "call_python":
                    n = item[3]
                    args = ", ".join(results[-n:]) if n > 0 else ""
                    if n > 0:
                        del results[-n:]
                    module = item[4]
                    function = item[5]
                    results.append(f"{module}.{function}({args})")
                elif marker == "string_concat":
                    right = results.pop()
                    left = results.pop()
                    results.append(f"str({left}) + str({right})")
                elif marker == "literal_list":
                    n = item[3]
                    items = ", ".join(results[-n:]) if n > 0 else ""
                    if n > 0:
                        del results[-n:]
                    results.append(f"[{items}]")
                continue

            if isinstance(current, Literal):
                if isinstance(current.value, str):
                    escaped = current.value.replace("\\", "\\\\").replace('"', '\\"')
                    results.append(f'"{escaped}"')
                elif current.value is None:
                    results.append("None")
                elif isinstance(current.value, bool):
                    results.append("True" if current.value else "False")
                elif isinstance(current.value, list):
                    n = len(current.value)
                    work_stack.append((current, True, "literal_list", n))
                    for v in reversed(current.value):
                        work_stack.append((Literal(v), False))
                else:
                    results.append(str(current.value))
            elif isinstance(current, Var):
                results.append(current.name)
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
            elif isinstance(current, And):
                work_stack.append((current, True, "and"))
                work_stack.append((current.right, False))
                work_stack.append((current.left, False))
            elif isinstance(current, Or):
                work_stack.append((current, True, "or"))
                work_stack.append((current.right, False))
                work_stack.append((current.left, False))
            elif isinstance(current, Not):
                work_stack.append((current, True, "not"))
                work_stack.append((current.operand, False))
            elif isinstance(current, Cmp):
                work_stack.append((current, True, "cmp", current.op))
                work_stack.append((current.right, False))
                work_stack.append((current.left, False))
            elif isinstance(current, FuncCall):
                n = len(current.args)
                work_stack.append((current, True, "func_call", n, current.name))
                for a in reversed(current.args):
                    work_stack.append((a, False))
            elif isinstance(current, CallPython):
                n = len(current.args)
                work_stack.append((current, True, "call_python", n, current.module, current.function))
                for a in reversed(current.args):
                    work_stack.append((a, False))
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
            elif isinstance(current, StringConcat):
                work_stack.append((current, True, "string_concat"))
                work_stack.append((current.right, False))
                work_stack.append((current.left, False))
            else:
                results.append("None")

        return results[0] if results else "None"

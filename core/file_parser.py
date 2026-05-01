from __future__ import annotations

import re
from dataclasses import dataclass, field
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
    Cmp,
    Continue,
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
    Literal,
    ListExpr,
    Loop,
    Match,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    ParallelCall,
    Print,
    Return,
    Stmt,
    Sub,
    Var,
    VarDecl,
    While,
)


@dataclass(slots=True)
class SourceLocation:
    """源码位置信息 / Source code location info"""
    line: int
    col: int
    file: str = "<stdin>"


class ParseError(Exception):
    """解析错误 / Parse error"""
    def __init__(self, message: str, location: SourceLocation | None = None) -> None:
        self.location = location
        if location:
            super().__init__(f"{location.file}:{location.line}:{location.col}: {message}")
        else:
            super().__init__(message)


class NLFileParser:
    """NL文件解析器 - 将.nl源文件解析为IR节点树 / NL file parser - parses .nl source files into IR node trees.

    支持的中文关键字:
    定义函数/定义类/定义/导入/如果/当/对于/匹配/跳出/继续/返回/输出/尝试/抛出

    Supported Chinese keywords:
    定义函数/定义类/定义/导入/如果/当/对于/匹配/跳出/继续/返回/输出/尝试/抛出
    """

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._pos: int = 0
        self._file: str = "<stdin>"

    def parse_file(self, filepath: str) -> list[Stmt]:
        """解析.nl文件 / Parse a .nl file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        self._file = str(path)
        source = path.read_text(encoding="utf-8")
        return self.parse(source, self._file)

    def parse(self, source: str, filename: str = "<stdin>") -> list[Stmt]:
        """解析源代码字符串 / Parse source code string"""
        self._file = filename
        self._lines = source.split("\n")
        self._pos = 0
        stmts: list[Stmt] = []
        while self._pos < len(self._lines):
            line = self._lines[self._pos].strip()
            if not line or line.startswith("#"):
                self._pos += 1
                continue
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _current_line(self) -> str:
        """获取当前行内容 / Get current line content"""
        if self._pos < len(self._lines):
            return self._lines[self._pos].strip()
        return ""

    def _location(self) -> SourceLocation:
        """获取当前源码位置 / Get current source location"""
        line = self._current_line()
        return SourceLocation(line=self._pos + 1, col=1, file=self._file)

    def _parse_statement(self) -> Stmt | None:
        """根据行首关键字分发解析 / Dispatch parsing based on line-start keyword"""
        line = self._current_line()
        if not line or line.startswith("#"):
            self._pos += 1
            return None

        if line.startswith("定义函数"):
            return self._parse_func_def()
        if line.startswith("定义类"):
            return self._parse_class_def()
        if line.startswith("定义"):
            return self._parse_var_decl()
        if line.startswith("导入"):
            return self._parse_import()
        if line.startswith("如果"):
            return self._parse_if()
        if line.startswith("当"):
            return self._parse_while()
        if line.startswith("对于"):
            return self._parse_for()
        if line.startswith("遍历"):
            return self._parse_foreach()
        if line.startswith("重复"):
            return self._parse_repeat()
        if line.startswith("匹配"):
            return self._parse_match()
        if line == "跳出":
            self._pos += 1
            return Break()
        if line == "继续":
            self._pos += 1
            return Continue()
        if line.startswith("返回"):
            return self._parse_return()
        if line.startswith("输出"):
            return self._parse_print()
        if line.startswith("尝试"):
            return self._parse_try()
        if line.startswith("抛出"):
            return self._parse_raise()
        if line.startswith("异步"):
            return self._parse_async_call()
        if line.startswith("等待"):
            return self._parse_await()
        if line.startswith("并行"):
            return self._parse_parallel()

        return self._parse_assignment_or_expr()

    def _get_indent(self) -> int:
        """获取当前行的缩进级别 / Get indentation level of current line"""
        if self._pos >= len(self._lines):
            return 0
        line = self._lines[self._pos]
        count = 0
        for ch in line:
            if ch == " ":
                count += 1
            elif ch == "\t":
                count += 4  # Tab视为4个空格 / Tab counts as 4 spaces
            else:
                break
        return count

    def _parse_block(self, base_indent: int) -> list[Stmt]:
        """解析缩进代码块 - 读取比base_indent更深的所有行 / Parse indented block - read all lines deeper than base_indent"""
        stmts: list[Stmt] = []
        self._pos += 1
        while self._pos < len(self._lines):
            raw_line = self._lines[self._pos]
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                self._pos += 1
                continue
            current_indent = self._get_indent()
            if current_indent <= base_indent:
                break
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_func_def(self) -> FuncDef:
        """解析函数定义 - 定义函数 名(参数=默认值, *可变参数) -> 返回类型: / Parse function definition"""
        line = self._current_line()
        base_indent = self._get_indent()

        match = re.match(r'定义函数\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(\w+))?\s*:', line)
        if not match:
            raise ParseError(f"无效的函数定义: {line}", self._location())

        name = match.group(1)
        params_str = match.group(2).strip()
        return_type = match.group(3)

        params: list[tuple[str, str | None]] = []
        defaults: dict[str, Expr] | None = None
        variadic: str | None = None

        if params_str:
            for p in params_str.split(","):
                p = p.strip()
                if not p:
                    continue
                # 可变参数 *args / Variadic parameter *args
                if p.startswith("*"):
                    variadic = p[1:].strip()
                    continue
                # 默认参数 param=default / Default parameter param=default
                if "=" in p:
                    parts = p.split("=", 1)
                    param_part = parts[0].strip()
                    default_str = parts[1].strip()
                    type_parts = param_part.rsplit(":", 1)
                    if len(type_parts) == 2:
                        pname = type_parts[0].strip()
                        ptype = type_parts[1].strip()
                    else:
                        pname = param_part
                        ptype = None
                    params.append((pname, ptype))
                    if defaults is None:
                        defaults = {}
                    defaults[pname] = self._parse_expr(default_str)
                else:
                    type_parts = p.rsplit(":", 1)
                    if len(type_parts) == 2:
                        params.append((type_parts[0].strip(), type_parts[1].strip()))
                    else:
                        params.append((p, None))

        body = self._parse_block(base_indent)
        return FuncDef(name=name, params=params, body=body, return_type=return_type, defaults=defaults, variadic=variadic)

    def _parse_class_def(self):
        """解析类定义 - 定义类 名(基类): / Parse class definition"""
        from .ir import ClassDef
        line = self._current_line()
        base_indent = self._get_indent()

        match = re.match(r'定义类\s+(\w+)\s*(?:\(([^)]*)\))?\s*:', line)
        if not match:
            raise ParseError(f"无效的类定义: {line}", self._location())

        name = match.group(1)
        bases_str = match.group(2)
        bases = [b.strip() for b in bases_str.split(",") if b.strip()] if bases_str else []

        body = self._parse_block(base_indent)
        return ClassDef(name=name, bases=bases, body=body)

    def _parse_var_decl(self) -> VarDecl:
        """解析变量声明 - 定义 x = 5 或 定义 x: int = 5 / Parse variable declaration"""
        line = self._current_line()
        self._pos += 1

        match = re.match(r'定义\s+(\w+)\s*(?::\s*(\w+))?\s*=\s*(.+)', line)
        if not match:
            raise ParseError(f"无效的变量定义: {line}", self._location())

        name = match.group(1)
        type_hint = match.group(2)
        value_str = match.group(3).strip()
        value = self._parse_expr(value_str)
        return VarDecl(name=name, value=value, type_hint=type_hint)

    def _parse_import(self) -> ImportStmt:
        """解析导入语句 - 导入/从...导入 / Parse import statement"""
        line = self._current_line()
        self._pos += 1

        # 导入 math as 数学 / Import math as 数学
        match = re.match(r'导入\s+(\w+)\s*(?:as\s+(\w+))?', line)
        if match:
            module = match.group(1)
            alias = match.group(2)
            return ImportStmt(module=module, alias=alias)

        # 从 math 导入 平方, 立方 / From math import 平方, 立方
        match = re.match(r'从\s+(\w+)\s+导入\s+(.+)', line)
        if match:
            module = match.group(1)
            items_str = match.group(2).strip()
            items = [i.strip() for i in items_str.split(",") if i.strip()]
            return ImportStmt(module=module, items=items)

        raise ParseError(f"无效的导入语句: {line}", self._location())

    def _parse_inline_stmt(self, text: str) -> Stmt | None:
        text = text.strip()
        if not text:
            return None
        if text.startswith("返回"):
            m = re.match(r'返回\s+(.+)', text)
            if m:
                return Return(value=self._parse_expr(m.group(1).strip()))
            return Return()
        if text.startswith("输出"):
            m = re.match(r'输出\s*(.+)', text)
            if m:
                content = m.group(1).strip()
                if content.startswith("(") and content.endswith(")"):
                    depth = 0
                    for i, ch in enumerate(content):
                        if ch == "(":
                            depth += 1
                        elif ch == ")":
                            depth -= 1
                        if depth == 0 and i == len(content) - 1:
                            content = content[1:-1].strip()
                            break
                parts = self._split_print_args(content)
                if len(parts) == 1:
                    return Print(value=self._parse_expr(parts[0]))
                values = [self._parse_expr(p.strip()) for p in parts]
                return Print(value=Literal(None), values=values)
            return Print(value=Literal(None))
        if text == "跳出":
            return Break()
        if text == "继续":
            return Continue()
        if text.startswith("定义"):
            m = re.match(r'定义\s+(\w+)\s*=\s*(.+)', text)
            if m:
                return VarDecl(name=m.group(1), value=self._parse_expr(m.group(2).strip()))
        assign_match = re.match(r'(\w+)\s*=\s*(.+)', text)
        if assign_match:
            return Assign(target=assign_match.group(1), value=self._parse_expr(assign_match.group(2).strip()))
        return None

    def _parse_if(self) -> If:
        """解析条件语句 - 如果/否则如果/否则 / Parse conditional statement"""
        line = self._current_line()
        base_indent = self._get_indent()
        self._pos += 1

        match = re.match(r'如果\s+(.+?):\s*(.*)', line)
        if not match:
            raise ParseError(f"无效的 if 语句: {line}", self._location())

        condition = self._parse_condition_expr(match.group(1).strip())
        inline_body = match.group(2).strip()

        if inline_body:
            stmt = self._parse_inline_stmt(inline_body)
            body = [stmt] if stmt is not None else []
        else:
            body = self._parse_indented_block(base_indent)

        orelse: list[Stmt] = []
        while self._pos < len(self._lines):
            next_line = self._current_line()
            next_indent = self._get_indent()
            if next_indent != base_indent:
                break
            elif_match = re.match(r'否则如果\s+(.+?):\s*(.*)', next_line)
            if elif_match:
                elif_cond = self._parse_condition_expr(elif_match.group(1).strip())
                self._pos += 1
                elif_inline = elif_match.group(2).strip()
                if elif_inline:
                    stmt = self._parse_inline_stmt(elif_inline)
                    elif_body = [stmt] if stmt is not None else []
                else:
                    elif_body = self._parse_indented_block(base_indent)
                inner_orelse = self._parse_else_chain(base_indent)
                orelse = [If(condition=elif_cond, body=elif_body, orelse=inner_orelse)]
                break
            elif re.match(r'否则:\s*(.*)', next_line):
                else_match = re.match(r'否则:\s*(.*)', next_line)
                self._pos += 1
                else_inline = else_match.group(1).strip()
                if else_inline:
                    stmt = self._parse_inline_stmt(else_inline)
                    orelse = [stmt] if stmt is not None else []
                else:
                    orelse = self._parse_indented_block(base_indent)
                break
            else:
                break

        return If(condition=condition, body=body, orelse=orelse)

    def _parse_else_chain(self, base_indent: int) -> list[Stmt]:
        if self._pos >= len(self._lines):
            return []
        next_line = self._current_line()
        next_indent = self._get_indent()
        if next_indent != base_indent:
            return []
        elif_match = re.match(r'否则如果\s+(.+?):\s*(.*)', next_line)
        if elif_match:
            elif_cond = self._parse_condition_expr(elif_match.group(1).strip())
            self._pos += 1
            elif_inline = elif_match.group(2).strip()
            if elif_inline:
                stmt = self._parse_inline_stmt(elif_inline)
                elif_body = [stmt] if stmt is not None else []
            else:
                elif_body = self._parse_indented_block(base_indent)
            inner_orelse = self._parse_else_chain(base_indent)
            return [If(condition=elif_cond, body=elif_body, orelse=inner_orelse)]
        else_match = re.match(r'否则:\s*(.*)', next_line)
        if else_match:
            self._pos += 1
            else_inline = else_match.group(1).strip()
            if else_inline:
                stmt = self._parse_inline_stmt(else_inline)
                return [stmt] if stmt is not None else []
            else:
                return self._parse_indented_block(base_indent)
        return []

    def _parse_indented_block(self, base_indent: int) -> list[Stmt]:
        """解析缩进代码块 - 读取比base_indent更深的所有行 / Parse indented block"""
        stmts: list[Stmt] = []
        while self._pos < len(self._lines):
            raw_line = self._lines[self._pos]
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                self._pos += 1
                continue
            current_indent = self._get_indent()
            if current_indent <= base_indent:
                break
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_while(self) -> While:
        """解析当循环 - 当 条件: / Parse while loop"""
        line = self._current_line()
        base_indent = self._get_indent()

        match = re.match(r'当\s+(.+):', line)
        if not match:
            raise ParseError(f"无效的 while 语句: {line}", self._location())

        condition = self._parse_condition_expr(match.group(1).strip())
        body = self._parse_block(base_indent)
        return While(condition=condition, body=body)

    def _parse_for(self) -> For | ForRange:
        line = self._current_line()
        base_indent = self._get_indent()

        unpack_match = re.match(r'对于\s+(\w+)\s*,\s*(\w+)\s+在\s+(.+?)\s*:', line)
        if unpack_match:
            key_var = unpack_match.group(1)
            val_var = unpack_match.group(2)
            iterable_str = unpack_match.group(3).strip()
            iterable = self._parse_expr(iterable_str)
            body = self._parse_block(base_indent)
            return For(
                var=f"{key_var},{val_var}",
                iterable=iterable,
                body=body,
            )

        in_match = re.match(r'对于\s+(\w+)\s+在\s+(.+?)\s*:', line)
        if in_match:
            var_name = in_match.group(1)
            iterable_str = in_match.group(2).strip()
            iterable = self._parse_expr(iterable_str)
            body = self._parse_block(base_indent)
            return For(var=var_name, iterable=iterable, body=body)

        for_each_match = re.match(r'对于\s+(\w+)\s+中的每个元素\s+(\w+)\s*:', line)
        if for_each_match:
            arr_name = for_each_match.group(1)
            var_name = for_each_match.group(2)
            body = self._parse_block(base_indent)
            return For(var=var_name, iterable=Var(arr_name), body=body)

        range_match = re.match(r'对于\s+(\w+)\s+从\s+(.+?)\s+到\s+(.+?)(?:\s+步长\s+(.+))?\s*:', line)
        if range_match:
            var = range_match.group(1)
            start = self._parse_expr(range_match.group(2).strip())
            stop = self._parse_expr(range_match.group(3).strip())
            step = self._parse_expr(range_match.group(4).strip()) if range_match.group(4) else None
            body = self._parse_block(base_indent)
            return ForRange(var=var, start=start, stop=stop, step=step, body=body)

        kv_match = re.match(r'对于\s+(\w+)\s+中的每个键值对\s*\((\w+),\s*(\w+)\)\s*:', line)
        if kv_match:
            dict_name = kv_match.group(1)
            key_var = kv_match.group(2)
            val_var = kv_match.group(3)
            body = self._parse_block(base_indent)
            return For(
                var=f"{key_var},{val_var}",
                iterable=FuncCall(name="items", args=[Var(dict_name)]),
                body=body,
            )

        raise ParseError(f"无效的 for 语句: {line}", self._location())

    def _parse_foreach(self) -> For:
        line = self._current_line()
        base_indent = self._get_indent()

        unpack_match = re.match(r'遍历\s+(\w+)\s*,\s*(\w+)\s+在\s+(.+?)\s*:', line)
        if unpack_match:
            key_var = unpack_match.group(1)
            val_var = unpack_match.group(2)
            iterable_str = unpack_match.group(3).strip()
            iterable = self._parse_expr(iterable_str)
            body = self._parse_block(base_indent)
            return For(var=f"{key_var},{val_var}", iterable=iterable, body=body)

        match = re.match(r'遍历\s+(\w+)\s+在\s+(.+?)\s*:', line)
        if match:
            var_name = match.group(1)
            iterable_str = match.group(2).strip()
            iterable = self._parse_expr(iterable_str)
            body = self._parse_block(base_indent)
            return For(var=var_name, iterable=iterable, body=body)

        match = re.match(r'遍历\s+(.+?)\s+为\s+(\w+)\s*:', line)
        if match:
            iterable_str = match.group(1).strip()
            var_name = match.group(2)
            iterable = self._parse_expr(iterable_str)
            body = self._parse_block(base_indent)
            return For(var=var_name, iterable=iterable, body=body)

        raise ParseError(f"无效的遍历语句: {line}", self._location())

    def _parse_repeat(self) -> Loop | ForRange:
        line = self._current_line()
        base_indent = self._get_indent()

        match = re.match(r'重复\s+(.+?)\s+次\s*:', line)
        if match:
            count_expr = self._parse_expr(match.group(1).strip())
            body = self._parse_block(base_indent)
            if isinstance(count_expr, Literal) and isinstance(count_expr.value, int):
                return Loop(count=count_expr.value, body=body)
            return ForRange(var="_", start=Literal(0), stop=Sub(count_expr, Literal(1)), step=Literal(1), body=body)

        match = re.match(r'重复\s*:', line)
        if match:
            body = self._parse_block(base_indent)
            return Loop(count=1, body=body)

        raise ParseError(f"无效的重复语句: {line}", self._location())

    def _parse_match(self) -> Match:
        """解析模式匹配 - 匹配/情况/默认 / Parse pattern matching"""
        line = self._current_line()
        base_indent = self._get_indent()

        match = re.match(r'匹配\s+(.+):', line)
        if not match:
            raise ParseError(f"无效的 match 语句: {line}", self._location())

        value = self._parse_expr(match.group(1).strip())
        cases: list[tuple[Expr, list[Stmt]]] = []
        default: list[Stmt] | None = None

        self._pos += 1
        while self._pos < len(self._lines):
            next_line = self._current_line()
            if not next_line.strip():
                self._pos += 1
                continue

            next_indent = self._get_indent()
            if next_indent <= base_indent:
                break

            # 情况 值: / Case value:
            case_match = re.match(r'\s*情况\s+(.+):', next_line)
            if case_match:
                case_val = self._parse_expr(case_match.group(1).strip())
                case_body = self._collect_sub_block(next_indent)
                cases.append((case_val, case_body))
                continue

            # 默认: / Default:
            default_match = re.match(r'\s*默认:', next_line)
            if default_match:
                default = self._collect_sub_block(next_indent)
                continue

            break

        return Match(value=value, cases=cases, default=default)

    def _collect_sub_block(self, parent_indent: int) -> list[Stmt]:
        """收集子代码块（用于match的case/default） / Collect sub-block (for match case/default)"""
        stmts: list[Stmt] = []
        self._pos += 1
        while self._pos < len(self._lines):
            raw_line = self._lines[self._pos]
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                self._pos += 1
                continue
            current_indent = len(raw_line) - len(raw_line.lstrip())
            if current_indent <= parent_indent:
                break
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_return(self) -> Return:
        """解析返回语句 - 返回 值 / Parse return statement"""
        line = self._current_line()
        self._pos += 1
        match = re.match(r'返回\s+(.+)', line)
        if match:
            return Return(value=self._parse_expr(match.group(1).strip()))
        return Return()

    def _parse_print(self) -> Print:
        line = self._current_line()
        self._pos += 1
        match = re.match(r'输出\s*(.+)', line)
        if match:
            content = match.group(1).strip()
            if content.startswith("(") and content.endswith(")"):
                depth = 0
                for i, ch in enumerate(content):
                    if ch == "(":
                        depth += 1
                    elif ch == ")":
                        depth -= 1
                    if depth == 0 and i == len(content) - 1:
                        content = content[1:-1].strip()
                        break
            parts = self._split_print_args(content)
            if len(parts) == 1:
                return Print(value=self._parse_expr(parts[0]))
            values = [self._parse_expr(p.strip()) for p in parts]
            return Print(value=Literal(None), values=values)
        return Print(value=Literal(None))

    def _split_print_args(self, content: str) -> list[str]:
        """分割输出参数 - 考虑括号嵌套 / Split print arguments - respects bracket nesting"""
        parts: list[str] = []
        depth = 0
        current = []
        for ch in content:
            if ch in '([{':
                depth += 1
                current.append(ch)
            elif ch in ')]}':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append(''.join(current))
        return parts

    def _parse_try(self):
        """解析异常处理 - 尝试/捕获/最终 / Parse exception handling"""
        from .ir import TryExcept
        line = self._current_line()
        base_indent = self._get_indent()

        body = self._parse_block(base_indent)
        handlers: list[tuple[str | None, str | None, list[Stmt]]] = []
        finally_body: list[Stmt] | None = None

        while self._pos < len(self._lines):
            next_line = self._current_line()
            if not next_line.strip():
                self._pos += 1
                continue

            next_indent = self._get_indent()
            if next_indent != base_indent:
                break

            catch_match = re.match(r'捕获(?:\s+(\w+)(?:\s+as\s+(\w+))?)?\s*:', next_line)
            if catch_match:
                exc_type_or_name = catch_match.group(1)
                exc_name = catch_match.group(2)
                if exc_type_or_name is None:
                    exc_type = None
                elif exc_name is not None:
                    exc_type = exc_type_or_name
                else:
                    exc_type = None
                    exc_name = exc_type_or_name
                handler_body = self._parse_block(next_indent)
                handlers.append((exc_type, exc_name, handler_body))
                continue

            finally_match = re.match(r'最终:\s*(.*)', next_line)
            if finally_match:
                finally_body = self._parse_block(next_indent)
                continue

            break

        return TryExcept(body=body, handlers=handlers, finally_body=finally_body)

    def _parse_raise(self):
        """解析抛出异常 - 抛出 值 / Parse raise statement"""
        from .ir import Raise
        line = self._current_line()
        self._pos += 1
        match = re.match(r'抛出\s+(.+)', line)
        if match:
            return Raise(value=self._parse_expr(match.group(1).strip()))
        return Raise()

    def _parse_async_call(self) -> Stmt:
        """解析异步调用 - 异步 函数名(参数) / Parse async call"""
        line = self._current_line()
        self._pos += 1
        match = re.match(r'异步\s+(\w+)\s*\(([^)]*)\)', line)
        if match:
            name = match.group(1)
            args_str = match.group(2).strip()
            args = []
            if args_str:
                for a in args_str.split(","):
                    a = a.strip()
                    if a:
                        args.append(self._parse_expr(a))
            expr = AsyncCall(name=name, args=args)
            assign_match = re.match(r'(\w+)\s*=\s*异步\s+', line)
            if assign_match:
                return Assign(target=assign_match.group(1), value=expr)
            return Assign(target="_async_result", value=expr)
        raise ParseError(f"无效的异步调用: {line}", self._location())

    def _parse_await(self) -> Stmt:
        """解析等待 - 等待 任务 / Parse await"""
        line = self._current_line()
        self._pos += 1
        match = re.match(r'等待\s+(.+)', line)
        if match:
            task_expr = self._parse_expr(match.group(1).strip())
            return Assign(target="_await_result", value=AwaitExpr(task=task_expr))
        raise ParseError(f"无效的等待语句: {line}", self._location())

    def _parse_parallel(self) -> Stmt:
        """解析并行调用 - 并行 [f1(args), f2(args)] / Parse parallel call"""
        line = self._current_line()
        self._pos += 1
        match = re.match(r'并行\s+\[(.+)\]', line)
        if match:
            calls_str = match.group(1).strip()
            calls: list[FuncCall] = []
            depth = 0
            current = []
            for ch in calls_str:
                if ch == '(':
                    depth += 1
                    current.append(ch)
                elif ch == ')':
                    depth -= 1
                    current.append(ch)
                elif ch == ',' and depth == 0:
                    calls.append(self._parse_expr(current[0].strip() if current else ""))
                    current = []
                else:
                    current.append(ch)
            if current:
                call_expr = self._parse_expr(''.join(current).strip())
                if isinstance(call_expr, FuncCall):
                    calls.append(call_expr)
            return Assign(target="_parallel_result", value=ParallelCall(calls=calls))
        raise ParseError(f"无效的并行调用: {line}", self._location())

    def _parse_assignment_or_expr(self) -> Stmt:
        """解析赋值或表达式语句 / Parse assignment or expression statement"""
        line = self._current_line()
        self._pos += 1

        # self.xxx = 值 / self.attr = value
        attr_assign_match = re.match(r'(self\.\w+)\s*=\s*(.+)', line)
        if attr_assign_match:
            target = attr_assign_match.group(1)
            value_str = attr_assign_match.group(2).strip()
            value = self._parse_expr(value_str)
            return Assign(target=target, value=value)

        # 变量 = 值 / variable = value
        assign_match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if assign_match:
            target = assign_match.group(1)
            value_str = assign_match.group(2).strip()
            if value_str.startswith("异步"):
                async_match = re.match(r'异步\s+(\w+)\s*\(([^)]*)\)', value_str)
                if async_match:
                    name = async_match.group(1)
                    args_str = async_match.group(2).strip()
                    args = []
                    if args_str:
                        for a in args_str.split(","):
                            a = a.strip()
                            if a:
                                args.append(self._parse_expr(a))
                    return Assign(target=target, value=AsyncCall(name=name, args=args))
            if value_str.startswith("等待"):
                wait_match = re.match(r'等待\s+(.+)', value_str)
                if wait_match:
                    task_expr = self._parse_expr(wait_match.group(1).strip())
                    return Assign(target=target, value=AwaitExpr(task=task_expr))
            value = self._parse_expr(value_str)
            return Assign(target=target, value=value)

        # 数组[索引] = 值 / arr[index] = value
        idx_assign_match = re.match(r'(\w+)\[(.+)\]\s*=\s*(.+)', line)
        if idx_assign_match:
            arr_name = idx_assign_match.group(1)
            idx_str = idx_assign_match.group(2).strip()
            value_str = idx_assign_match.group(3).strip()
            idx_expr = self._parse_expr(idx_str)
            value = self._parse_expr(value_str)
            return IndexAssign(obj=Var(arr_name), index=idx_expr, value=value)

        # 独立表达式（如函数调用）/ Standalone expression (e.g. function call)
        expr = self._parse_expr(line)
        if isinstance(expr, FuncCall):
            return expr
        return Print(value=expr)

    def _parse_condition_expr(self, cond_str: str) -> Expr:
        """解析条件表达式 - 支持比较、且、或、非 / Parse condition expression - supports comparison, 且, 或, 非"""
        AND_MARKER = "__and__"
        OR_MARKER = "__or__"
        NOT_MARKER = "__not__"

        results: list[Expr] = []
        work_stack: list[tuple] = [(cond_str.strip(), False)]

        while work_stack:
            current, processed = work_stack.pop()

            if processed:
                if current == AND_MARKER:
                    right = results.pop()
                    left = results.pop()
                    results.append(And(left, right))
                elif current == OR_MARKER:
                    right = results.pop()
                    left = results.pop()
                    results.append(Or(left, right))
                elif current == NOT_MARKER:
                    expr = results.pop()
                    results.append(Not(expr))
                continue

            found_cmp = False
            for op in [">=", "<=", "!=", ">", "<", "=="]:
                idx = current.find(op)
                if idx != -1:
                    left_str = current[:idx].strip()
                    right_str = current[idx + len(op):].strip()
                    results.append(Cmp(self._parse_expr(left_str), op, self._parse_expr(right_str)))
                    found_cmp = True
                    break
            if found_cmp:
                continue

            and_idx = current.find(" 且 ")
            if and_idx != -1:
                work_stack.append((AND_MARKER, True))
                work_stack.append((current[and_idx + 3:].strip(), False))
                work_stack.append((current[:and_idx].strip(), False))
                continue

            or_idx = current.find(" 或 ")
            if or_idx != -1:
                work_stack.append((OR_MARKER, True))
                work_stack.append((current[or_idx + 3:].strip(), False))
                work_stack.append((current[:or_idx].strip(), False))
                continue

            not_match = re.match(r'非\s+(.+)', current)
            if not_match:
                work_stack.append((NOT_MARKER, True))
                work_stack.append((not_match.group(1).strip(), False))
                continue

            results.append(self._parse_expr(current))

        return results[0] if results else self._parse_expr(cond_str)

    def _parse_expr(self, expr_str: str) -> Expr:
        """委托给表达式解析器 / Delegate to expression parser"""
        return _ExprParser(expr_str).parse()


class _ExprParser:
    """递归下降表达式解析器 - 支持运算符优先级 / Recursive descent expression parser - supports operator precedence.

    优先级从低到高 / Precedence (low to high):
    或 -> 且 -> 比较 -> 加减 -> 乘除模 -> 一元 -> 后缀 -> 基本表达式
    或 -> 且 -> comparison -> add/sub -> mul/div/mod -> unary -> postfix -> primary
    """

    def __init__(self, source: str) -> None:
        self.src = source.strip()
        self.pos = 0

    def parse(self) -> Expr:
        result = self._parse_or()
        return result

    def _parse_or(self) -> Expr:
        """解析逻辑或 / Parse logical OR"""
        left = self._parse_and()
        while self._match("或"):
            self._skip_ws()
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> Expr:
        """解析逻辑与 / Parse logical AND"""
        left = self._parse_comparison()
        while self._match("且"):
            self._skip_ws()
            right = self._parse_comparison()
            left = And(left, right)
        return left

    def _parse_comparison(self) -> Expr:
        """解析比较运算 / Parse comparison"""
        left = self._parse_add()
        for op in [">=", "<=", "!=", "==", ">", "<"]:
            if self._match(op):
                right = self._parse_add()
                return Cmp(left, op, right)
        return left

    def _parse_add(self) -> Expr:
        """解析加减法 / Parse addition/subtraction"""
        left = self._parse_mul()
        while True:
            if self._match("+"):
                right = self._parse_mul()
                left = Add(left, right)
            elif self._match("-"):
                right = self._parse_mul()
                left = Sub(left, right)
            else:
                break
        return left

    def _parse_mul(self) -> Expr:
        """解析乘除取模 / Parse multiplication/division/modulo"""
        left = self._parse_unary()
        while True:
            if self._match("*"):
                right = self._parse_unary()
                left = Mul(left, right)
            elif self._match("/"):
                right = self._parse_unary()
                left = Div(left, right)
            elif self._match("%"):
                right = self._parse_unary()
                left = Mod(left, right)
            else:
                break
        return left

    def _parse_unary(self) -> Expr:
        """解析一元运算（负号、逻辑非）/ Parse unary operations (negation, logical NOT)"""
        ops: list[str] = []
        while True:
            if self._match("-"):
                ops.append("neg")
            elif self._match("非"):
                ops.append("not")
            else:
                break
        expr = self._parse_postfix()
        for op in reversed(ops):
            if op == "neg":
                expr = Neg(expr)
            else:
                expr = Not(expr)
        return expr

    def _parse_postfix(self) -> Expr:
        """解析后缀运算（索引访问、属性访问、方法调用）/ Parse postfix operations (index, attribute, method call)"""
        expr = self._parse_primary()
        while True:
            if self._match("["):
                idx = self.parse()
                self._expect("]")
                expr = IndexAccess(obj=expr, index=idx)
            elif self._match("."):
                attr = self._read_identifier()
                if self._match("("):
                    args = self._parse_call_args()
                    self._expect(")")
                    # 对于 obj.method(args)，创建特殊的函数调用 / For obj.method(args), create special function call
                    # args[0] 是对象，args[1:] 是方法参数 / args[0] is the object, args[1:] are method args
                    expr = FuncCall(name=f"_attr_{attr}", args=[expr] + args)
                else:
                    expr = AttributeAccess(obj=expr, attr=attr)
            else:
                break
        return expr

    def _parse_primary(self) -> Expr:
        """解析基本表达式 - 字面量、变量、函数调用、括号表达式 / Parse primary expression - literal, variable, function call, parenthesized"""
        self._skip_ws()

        if self.pos >= len(self.src):
            return Literal(None)

        ch = self.src[self.pos]

        # 括号表达式 / Parenthesized expression
        if ch == "(":
            self.pos += 1
            expr = self.parse()
            self._expect(")")
            return expr

        # 列表字面量 / List literal
        if ch == "[":
            return self._parse_list()

        if ch == "{":
            return self._parse_dict()

        # 字符串字面量 / String literal
        if ch == '"' or ch == "'":
            return self._parse_string()

        # 布尔字面量 / Boolean literal
        if ch == "真":
            self.pos += 1
            return Literal(True)

        if ch == "假":
            self.pos += 1
            return Literal(False)

        # 空值字面量 / None literal
        if ch == "空":
            self.pos += 1
            return Literal(None)

        # 数字字面量 / Number literal
        num = self._try_parse_number()
        if num is not None:
            return Literal(num)

        # 标识符（变量或函数调用）/ Identifier (variable or function call)
        ident = self._peek_identifier()
        if ident:
            self.pos += len(ident)
            if self._match("("):
                args = self._parse_call_args()
                self._expect(")")
                return FuncCall(name=ident, args=args)
            return Var(ident)

        # 回退：将剩余内容作为变量名 / Fallback: treat remaining content as variable name
        remaining = self.src[self.pos:]
        return Var(remaining.strip())

    def _parse_list(self) -> Expr:
        self._expect("[")
        elements: list[Expr] = []
        self._skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == "]":
            self.pos += 1
            return ListExpr(elements=elements)
        elements.append(self.parse())
        while self._match(","):
            elements.append(self.parse())
        self._expect("]")
        return ListExpr(elements=elements)

    def _parse_dict(self) -> Expr:
        from .ir import DictExpr
        self._expect("{")
        pairs: list[tuple[Expr, Expr]] = []
        self._skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == "}":
            self.pos += 1
            return DictExpr(pairs=pairs)
        key = self.parse()
        self._expect(":")
        value = self.parse()
        pairs.append((key, value))
        while self._match(","):
            self._skip_ws()
            if self.pos < len(self.src) and self.src[self.pos] == "}":
                break
            key = self.parse()
            self._expect(":")
            value = self.parse()
            pairs.append((key, value))
        self._expect("}")
        return DictExpr(pairs=pairs)

    def _parse_string(self) -> Expr:
        """解析字符串字面量 / Parse string literal"""
        quote = self.src[self.pos]
        self.pos += 1
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos] != quote:
            if self.src[self.pos] == "\\":  # 转义字符 / Escape character
                self.pos += 1
            self.pos += 1
        value = self.src[start:self.pos]
        if self.pos < len(self.src):
            self.pos += 1
        return Literal(value)

    def _parse_call_args(self) -> list[Expr]:
        """解析函数调用参数列表 / Parse function call argument list"""
        args: list[Expr] = []
        self._skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == ")":
            return args
        args.append(self.parse())
        while self._match(","):
            args.append(self.parse())
        return args

    def _try_parse_number(self) -> int | float | None:
        """尝试解析数字 / Try to parse a number"""
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isdigit() or self.src[self.pos] == "."):
            self.pos += 1
        if self.pos == start:
            return None
        num_str = self.src[start:self.pos]
        try:
            if "." in num_str:
                return float(num_str)
            return int(num_str)
        except ValueError:
            self.pos = start
            return None

    def _peek_identifier(self) -> str:
        """预读标识符（不消费）/ Peek at identifier (no consumption)"""
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == "_"):
            self.pos += 1
        ident = self.src[start:self.pos]
        self.pos = start  # 回退 / Reset position
        if ident and not ident[0].isdigit():
            return ident
        return ""

    def _read_identifier(self) -> str:
        """读取并消费标识符 / Read and consume identifier"""
        self._skip_ws()
        start = self.pos
        while self.pos < len(self.src) and (self.src[self.pos].isalnum() or self.src[self.pos] == "_"):
            self.pos += 1
        return self.src[start:self.pos]

    def _match(self, expected: str) -> bool:
        """尝试匹配指定字符串 / Try to match expected string"""
        self._skip_ws()
        if self.src[self.pos:self.pos + len(expected)] == expected:
            self.pos += len(expected)
            return True
        return False

    def _expect(self, ch: str) -> None:
        """期望并消费指定字符 / Expect and consume specified character"""
        self._skip_ws()
        if self.pos < len(self.src) and self.src[self.pos] == ch:
            self.pos += 1
        else:
            pass

    def _skip_ws(self) -> None:
        """跳过空白字符 / Skip whitespace"""
        while self.pos < len(self.src) and self.src[self.pos] in " \t":
            self.pos += 1

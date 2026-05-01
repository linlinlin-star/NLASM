from __future__ import annotations

from typing import Any

from .ir import (
    Assign,
    Break,
    Continue,
    Expr,
    For,
    ForRange,
    FuncCall,
    FuncDef,
    If,
    ImportStmt,
    Literal,
    Loop,
    Print,
    Return,
    Stmt,
    Var,
    VarDecl,
    While,
)
from .ir_interpreter import IRInterpreter, ReturnSignal, BreakSignal, ContinueSignal, _ReturnSentinel, _BreakSentinel, _ContinueSentinel, _is_control_signal
from .python_bridge import PythonBridge
from .slot_types import ArraySlot


class DebugAction:
    """调试动作常量 / Debug action constants"""
    CONTINUE = "continue"   # 继续执行 / Continue execution
    STEP = "step"           # 单步执行 / Step over
    STEP_IN = "step_in"     # 步入函数 / Step into
    STEP_OUT = "step_out"   # 步出函数 / Step out
    STOP = "stop"           # 停止执行 / Stop execution


class Breakpoint:
    """断点 - 在指定行号暂停执行 / Breakpoint - pause execution at specified line"""
    def __init__(self, line: int, condition: str | None = None) -> None:
        self.line = line
        self.condition = condition  # 条件断点表达式 / Conditional breakpoint expression
        self.hit_count = 0          # 命中次数 / Hit count
        self.enabled = True         # 是否启用 / Whether enabled


class DebugFrame:
    """调试栈帧 - 记录函数调用信息 / Debug stack frame - records function call info"""
    def __init__(self, name: str, env: dict[str, Any], stmt_index: int) -> None:
        self.name = name            # 函数名 / Function name
        self.env = dict(env)        # 环境变量快照 / Environment variable snapshot
        self.stmt_index = stmt_index  # 语句索引 / Statement index


class DebugEvent:
    """调试事件 - 记录调试过程中的事件 / Debug event - records events during debugging"""
    def __init__(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        self.event_type = event_type  # 事件类型 / Event type
        self.data = data or {}        # 事件数据 / Event data


class NLASMDebugger:
    """NLASM调试器 - 支持断点、单步、变量查看 / NLASM debugger - supports breakpoints, stepping, variable inspection.

    调试功能:
    - 设置/移除/启用/禁用断点（含条件断点）
    - 单步执行(Step)、步入(Step In)、步出(Step Out)、继续(Continue)
    - 查看当前变量、调用栈、输出
    - 在断点处求值表达式

    Debug features:
    - Set/remove/enable/disable breakpoints (with conditional breakpoints)
    - Step, Step In, Step Out, Continue execution
    - Inspect current variables, call stack, outputs
    - Evaluate expressions at breakpoints
    """

    def __init__(self) -> None:
        self.interp = IRInterpreter()
        self.breakpoints: dict[int, Breakpoint] = {}  # 行号 -> 断点 / Line number -> breakpoint
        self.call_stack: list[DebugFrame] = []  # 调用栈 / Call stack
        self.current_line: int = 0
        self.action: str = DebugAction.CONTINUE
        self.events: list[DebugEvent] = []  # 调试事件列表 / Debug event list
        self.step_depth: int = 0
        self._stmt_counter: int = 0
        self._paused: bool = False

    def set_breakpoint(self, line: int, condition: str | None = None) -> None:
        """设置断点 / Set breakpoint"""
        self.breakpoints[line] = Breakpoint(line=line, condition=condition)

    def remove_breakpoint(self, line: int) -> None:
        """移除断点 / Remove breakpoint"""
        self.breakpoints.pop(line, None)

    def enable_breakpoint(self, line: int, enabled: bool = True) -> None:
        """启用/禁用断点 / Enable/disable breakpoint"""
        if line in self.breakpoints:
            self.breakpoints[line].enabled = enabled

    def run(self, stmts: list[Stmt], source_lines: list[str] | None = None) -> Any:
        """启动调试执行 / Start debug execution"""
        self._source_lines = source_lines or []
        self._stmt_counter = 0
        self._paused = False
        self.action = DebugAction.CONTINUE
        try:
            result = self._run_stmts(stmts)
            self._emit("terminated", {"result": result})
            return result
        except ReturnSignal as ret:
            self._emit("terminated", {"result": ret.value})
            return ret.value

    def step(self) -> None:
        """单步执行 / Step over"""
        self.action = DebugAction.STEP
        self._paused = False

    def step_in(self) -> None:
        """步入函数 / Step into function"""
        self.action = DebugAction.STEP_IN
        self._paused = False

    def step_out(self) -> None:
        """步出函数 / Step out of function"""
        self.action = DebugAction.STEP_OUT
        self.step_depth = len(self.call_stack) - 1
        self._paused = False

    def continue_exec(self) -> None:
        """继续执行 / Continue execution"""
        self.action = DebugAction.CONTINUE
        self._paused = False

    def stop(self) -> None:
        """停止执行 / Stop execution"""
        self.action = DebugAction.STOP
        self._paused = False

    def get_variables(self) -> dict[str, Any]:
        """获取当前所有变量 / Get all current variables"""
        return dict(self.interp.env)

    def get_call_stack(self) -> list[DebugFrame]:
        """获取调用栈 / Get call stack"""
        return list(self.call_stack)

    def get_outputs(self) -> list[Any]:
        """获取所有输出 / Get all outputs"""
        return list(self.interp.outputs)

    def evaluate(self, expr_str: str) -> Any:
        """在当前上下文中求值表达式 / Evaluate expression in current context"""
        from .file_parser import _ExprParser
        expr = _ExprParser(expr_str).parse()
        return self.interp._eval_expr(expr)

    def _emit(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """发射调试事件 / Emit debug event"""
        self.events.append(DebugEvent(event_type, data))

    def _check_breakpoint(self) -> bool:
        """检查是否需要暂停 / Check if execution should pause"""
        if self.action == DebugAction.STOP:
            return True

        if self.action == DebugAction.STEP:
            return True

        if self.action == DebugAction.STEP_IN:
            return True

        if self.action == DebugAction.STEP_OUT:
            if len(self.call_stack) <= self.step_depth:
                return True
            return False

        # 检查行断点 / Check line breakpoint
        if self.current_line in self.breakpoints:
            bp = self.breakpoints[self.current_line]
            if bp.enabled:
                bp.hit_count += 1
                if bp.condition is None:
                    return True
                try:
                    result = self.evaluate(bp.condition)
                    return bool(result)
                except Exception:
                    return False

        return False

    def _pause(self, stmt: Stmt) -> None:
        """暂停执行并发射事件 / Pause execution and emit event"""
        self._paused = True
        self._emit("stopped", {
            "line": self.current_line,
            "stmt": type(stmt).__name__,
            "variables": dict(self.interp.env),
        })

    def _run_stmts(self, stmts: list[Stmt]) -> Any:
        """执行语句列表（含断点检查）/ Execute statement list (with breakpoint checking)"""
        result = None
        work_stack: list[tuple[list[Stmt], int]] = [(stmts, 0)]

        while work_stack:
            current_stmts, idx = work_stack[-1]

            if idx >= len(current_stmts):
                work_stack.pop()
                continue

            work_stack[-1] = (current_stmts, idx + 1)
            stmt = current_stmts[idx]

            self._stmt_counter += 1
            self.current_line = self._stmt_counter

            if self._check_breakpoint():
                self._pause(stmt)
                self._emit("breakpoint", {"line": self.current_line})

            if self.action == DebugAction.STOP:
                return result

            if isinstance(stmt, If):
                cond = self.interp._eval_expr(stmt.condition)
                branch = stmt.body if cond else stmt.orelse
                if branch:
                    work_stack.append((branch, 0))
                continue

            result = self._exec_debug_stmt(stmt)

        return result

    def _exec_debug_stmt(self, node: Stmt) -> Any:
        """执行单条语句（含调试逻辑）/ Execute single statement (with debug logic)"""
        if isinstance(node, FuncDef):
            self.call_stack.append(DebugFrame(
                name=node.name,
                env=dict(self.interp.env),
                stmt_index=0,
            ))
            result = self.interp._exec_stmt(node)
            return result

        if isinstance(node, While):
            while self.interp._eval_expr(node.condition):
                for stmt in node.body:
                    self._stmt_counter += 1
                    self.current_line = self._stmt_counter
                    if self._check_breakpoint():
                        self._pause(stmt)
                    if self.action == DebugAction.STOP:
                        return None
                    try:
                        self.interp._exec_stmt(stmt)
                    except BreakSignal:
                        break
                    except ContinueSignal:
                        continue
            return None

        if isinstance(node, For):
            iterable = self.interp._eval_expr(node.iterable)
            for item in iterable:
                self.interp.env[node.var] = item
                for stmt in node.body:
                    self._stmt_counter += 1
                    self.current_line = self._stmt_counter
                    if self._check_breakpoint():
                        self._pause(stmt)
                    if self.action == DebugAction.STOP:
                        return None
                    try:
                        self.interp._exec_stmt(stmt)
                    except BreakSignal:
                        break
                    except ContinueSignal:
                        continue
            return None

        if isinstance(node, ForRange):
            start = int(self.interp._eval_expr(node.start))
            stop = int(self.interp._eval_expr(node.stop))
            step = int(self.interp._eval_expr(node.step)) if node.step else 1
            for i in range(start, stop, step):
                self.interp.env[node.var] = i
                for stmt in node.body:
                    self._stmt_counter += 1
                    self.current_line = self._stmt_counter
                    if self._check_breakpoint():
                        self._pause(stmt)
                    if self.action == DebugAction.STOP:
                        return None
                    try:
                        self.interp._exec_stmt(stmt)
                    except BreakSignal:
                        break
                    except ContinueSignal:
                        continue
            return None

        return self.interp._exec_stmt(node)

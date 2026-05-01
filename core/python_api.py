from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .ir import (
    Add,
    And,
    Assign,
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
    IndexAssign,
    Literal,
    ListExpr,
    Loop,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    Print,
    Return,
    Stmt,
    Sub,
    Var,
    VarDecl,
    While,
)
from .ir_interpreter import IRInterpreter
from .python_bridge import PythonBridge
from .slot_types import ArraySlot


@dataclass(slots=True)
class FuncParam:
    """函数参数定义 / Function parameter definition"""
    name: str
    type_name: str


@dataclass(slots=True)
class FuncSignature:
    """函数签名 - 描述函数的参数类型和返回类型 / Function signature - describes parameter types and return type"""
    params: list[FuncParam] = field(default_factory=list)
    return_type: str = "int64"


class NLASMFunction:
    """NLASM可调用函数 - 将IR节点包装为Python可调用对象 / NLASM callable function - wraps IR nodes as Python callable.

    通过IR解释器执行IR节点，支持参数绑定和类型转换。
    Executes IR nodes through IR interpreter, supports argument binding and type conversion.
    """

    def __init__(self, ir_nodes: list[Stmt], signature: FuncSignature, bridge: PythonBridge | None = None) -> None:
        self.ir_nodes = ir_nodes
        self.signature = signature
        self._bridge = bridge or PythonBridge()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """执行NLASM函数 / Execute NLASM function"""
        arr_slot = self._build_arr_slot(args)
        interp = IRInterpreter(arr_slot=arr_slot, bridge=self._bridge)
        self._bind_args(interp, args, kwargs)
        result = interp.run(self.ir_nodes)
        if result is not None:
            return self._convert_result(result)
        if interp.outputs:
            return self._convert_result(interp.outputs[-1])
        return None

    def _build_arr_slot(self, args: tuple) -> ArraySlot | None:
        """从参数中构建数组槽位 / Build array slot from arguments"""
        for i, (arg, param) in enumerate(zip(args, self.signature.params)):
            if param.type_name.startswith("array"):
                import numpy as np
                arr = np.asarray(arg)
                return ArraySlot(name=param.name, values=arr.tolist())
        return None

    def _bind_args(self, interp: IRInterpreter, args: tuple, kwargs: dict[str, Any]) -> None:
        """绑定函数参数到解释器环境 / Bind function arguments to interpreter environment"""
        for i, (arg, param) in enumerate(zip(args, self.signature.params)):
            interp.env[param.name] = self._convert_arg(arg, param.type_name)
        for key, val in kwargs.items():
            interp.env[key] = val

    def _convert_arg(self, arg: Any, type_name: str) -> Any:
        """转换参数类型 / Convert argument type"""
        if type_name.startswith("array"):
            import numpy as np
            arr = np.asarray(arg)
            return arr.tolist()
        if type_name == "int64":
            return int(arg)
        if type_name == "float64":
            return float(arg)
        return arg

    def _convert_result(self, result: Any) -> Any:
        """转换返回值类型 / Convert return value type"""
        if isinstance(result, int):
            return result
        if isinstance(result, float):
            return result
        if isinstance(result, list):
            return result
        return result


def _parse_function_signature(code: str) -> tuple[str, list[tuple[str, str]], str]:
    """解析函数签名 - 从.nl代码中提取函数名、参数和返回类型 / Parse function signature - extract name, params, and return type from .nl code"""
    import re

    func_match = re.search(r'定义函数\s+(\w+)\s*\(([^)]*)\)', code)
    if not func_match:
        raise SyntaxError("无法解析函数定义")

    func_name = func_match.group(1)
    params_str = func_match.group(2).strip()

    params: list[tuple[str, str]] = []
    if params_str:
        for p in params_str.split(","):
            p = p.strip()
            if not p:
                continue
            parts = p.rsplit(":", 1)
            if len(parts) == 2:
                pname = parts[0].strip()
                ptype = parts[1].strip()
            else:
                pname = p
                ptype = "int64"
            params.append((pname, ptype))

    return func_name, params, "int64"


def _compile_nlasm_code(code: str) -> tuple[list[Stmt], FuncSignature]:
    """编译.nl代码为IR节点和函数签名 / Compile .nl code to IR nodes and function signature"""
    import re

    func_name, param_list, return_type = _parse_function_signature(code)

    signature = FuncSignature(
        params=[FuncParam(name=n, type_name=t) for n, t in param_list],
        return_type=return_type,
    )

    body_code = code[code.index(")") + 1:]
    ir_nodes = _compile_body(body_code, param_list)
    return ir_nodes, signature


def _compile_body(body_code: str, param_list: list[tuple[str, str]]) -> list[Stmt]:
    """编译函数体为IR语句列表 / Compile function body to IR statement list"""
    import re

    nodes: list[Stmt] = []
    lines = body_code.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        import_match = re.match(r'导入\s+(\w+)\s*(?:as\s+(\w+))?', line)
        if import_match:
            module = import_match.group(1)
            alias = import_match.group(2)
            nodes.append(ImportStmt(module=module, alias=alias))
            continue

        assign_match = re.match(r'定义\s+(\w+)\s*=\s*(.+)', line)
        if assign_match:
            target = assign_match.group(1)
            value_str = assign_match.group(2).strip()
            value = _parse_expr(value_str)
            nodes.append(Assign(target=target, value=value))
            continue

        return_match = re.match(r'返回\s+(.+)', line)
        if return_match:
            value_str = return_match.group(1).strip()
            value = _parse_expr(value_str)
            nodes.append(Return(value=value))
            continue

        print_match = re.match(r'输出\s+(.+)', line)
        if print_match:
            value_str = print_match.group(1).strip()
            value = _parse_expr(value_str)
            nodes.append(Print(value=value))
            continue

        if_match = re.match(r'如果\s+(.+):', line)
        if if_match:
            cond_str = if_match.group(1).strip()
            cond = _parse_condition(cond_str)
            nodes.append(If(condition=cond, body=[], orelse=[]))
            continue

        for_match = re.match(r'对于\s+(\w+)\s+中的每个元素\s+(\w+):', line)
        if for_match:
            arr_name = for_match.group(1)
            elem_name = for_match.group(2)
            nodes.append(For(var=elem_name, iterable=Var(arr_name), body=[]))
            continue

    return nodes


def _parse_expr(expr_str: str) -> Expr:
    """解析表达式字符串为IR表达式节点 / Parse expression string to IR expression node"""
    import re

    results: list[Expr] = []
    work_stack: list[tuple] = [("parse", expr_str.strip())]

    while work_stack:
        item = work_stack.pop()
        action = item[0]

        if action == "combine":
            marker = item[1]
            if marker == "add":
                right = results.pop()
                left = results.pop()
                results.append(Add(left, right))
            elif marker == "sub":
                right = results.pop()
                left = results.pop()
                results.append(Sub(left, right))
            elif marker == "mul":
                right = results.pop()
                left = results.pop()
                results.append(Mul(left, right))
            elif marker == "call":
                module = item[2]
                func = item[3]
                num_args = item[4]
                args = results[-num_args:]
                del results[-num_args:]
                results.append(CallPython(module=module, function=func, args=args))
            continue

        current = item[1]

        list_match = re.match(r'\[([^\]]*)\]', current)
        if list_match:
            inner = list_match.group(1).strip()
            if inner:
                values = [int(x.strip()) for x in inner.split(",") if x.strip()]
            else:
                values = []
            results.append(Literal(values))
            continue

        try:
            results.append(Literal(int(current)))
            continue
        except ValueError:
            pass

        try:
            results.append(Literal(float(current)))
            continue
        except ValueError:
            pass

        call_match = re.match(r'(\w+)\.(\w+)\(([^)]*)\)', current)
        if call_match:
            module = call_match.group(1)
            func = call_match.group(2)
            args_str = call_match.group(3).strip()
            if args_str:
                arg_strs = [a.strip() for a in args_str.split(",")]
                work_stack.append(("combine", "call", module, func, len(arg_strs)))
                for a in reversed(arg_strs):
                    work_stack.append(("parse", a.strip()))
            else:
                results.append(CallPython(module=module, function=func, args=[]))
            continue

        add_match = re.match(r'(.+?)\s*\+\s*(.+)', current)
        if add_match:
            work_stack.append(("combine", "add"))
            work_stack.append(("parse", add_match.group(2).strip()))
            work_stack.append(("parse", add_match.group(1).strip()))
            continue

        sub_match = re.match(r'(.+?)\s*-\s*(.+)', current)
        if sub_match:
            work_stack.append(("combine", "sub"))
            work_stack.append(("parse", sub_match.group(2).strip()))
            work_stack.append(("parse", sub_match.group(1).strip()))
            continue

        mul_match = re.match(r'(.+?)\s*\*\s*(.+)', current)
        if mul_match:
            work_stack.append(("combine", "mul"))
            work_stack.append(("parse", mul_match.group(2).strip()))
            work_stack.append(("parse", mul_match.group(1).strip()))
            continue

        results.append(Var(current))

    return results[0] if results else Var(expr_str)


def _parse_condition(cond_str: str) -> Expr:
    """解析条件表达式 / Parse condition expression"""
    import re

    for op in [">=", "<=", "!=", ">", "<", "=="]:
        pattern = re.escape(op)
        match = re.search(pattern, cond_str)
        if match:
            left_str = cond_str[:match.start()].strip()
            right_str = cond_str[match.end():].strip()
            return Cmp(_parse_expr(left_str), op, _parse_expr(right_str))

    raise SyntaxError(f"无法解析条件: {cond_str}")


def compile(code: str) -> NLASMFunction:
    """编译.nl代码为可调用的NLASMFunction / Compile .nl code to callable NLASMFunction"""
    ir_nodes, signature = _compile_nlasm_code(code)
    return NLASMFunction(ir_nodes=ir_nodes, signature=signature)

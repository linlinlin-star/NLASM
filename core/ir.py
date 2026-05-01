from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class IRNode:
    """所有IR节点的基类 / Base class for all IR (Intermediate Representation) nodes."""


class Expr(IRNode):
    """表达式节点基类 - 表示可求值的计算单元 / Base class for expression nodes - represents evaluable computation units."""


class Stmt(IRNode):
    """语句节点基类 - 表示可执行的操作 / Base class for statement nodes - represents executable operations."""


@dataclass(slots=True)
class Literal(Expr):
    """字面量 - 整数、浮点数、字符串、布尔值等常量 / Literal constant - int, float, str, bool, etc."""
    value: int | float | str | bool | list[Any] | None


@dataclass(slots=True)
class Var(Expr):
    """变量引用 / Variable reference"""
    name: str


@dataclass(slots=True)
class Add(Expr):
    """加法运算 / Addition operation"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Sub(Expr):
    """减法运算 / Subtraction operation"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Mul(Expr):
    """乘法运算 / Multiplication operation"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Div(Expr):
    """除法运算（整数除法向下取整）/ Division operation (integer division floors)"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Mod(Expr):
    """取模运算 / Modulo operation"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Neg(Expr):
    """负号运算 / Negation operation"""
    operand: Expr


@dataclass(slots=True)
class And(Expr):
    """逻辑与（中文关键字"且"）/ Logical AND (Chinese keyword "且")"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Or(Expr):
    """逻辑或（中文关键字"或"）/ Logical OR (Chinese keyword "或")"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class Not(Expr):
    """逻辑非（中文关键字"非"）/ Logical NOT (Chinese keyword "非")"""
    operand: Expr


@dataclass(slots=True)
class Cmp(Expr):
    """比较运算 - 支持 >, >=, <, <=, ==, != / Comparison operation - supports >, >=, <, <=, ==, !="""
    left: Expr
    op: str
    right: Expr


@dataclass(slots=True)
class CallPython(Expr):
    """调用Python外部函数 / Call an external Python function"""
    module: str
    function: str
    args: list[Expr] = field(default_factory=list)
    kwargs: dict[str, Expr] = field(default_factory=dict)


@dataclass(slots=True)
class FuncCall(Expr):
    """函数调用 - 包括普通函数和属性方法调用（_attr_前缀）/ Function call - includes normal functions and attribute method calls (_attr_ prefix)"""
    name: str
    args: list[Expr] = field(default_factory=list)
    kwargs: dict[str, Expr] = field(default_factory=dict)


@dataclass(slots=True)
class IndexAccess(Expr):
    """索引访问 - 如 arr[0] / Index access - e.g. arr[0]"""
    obj: Expr
    index: Expr


@dataclass(slots=True)
class AttributeAccess(Expr):
    """属性访问 - 如 math.平方 / Attribute access - e.g. math.平方"""
    obj: Expr
    attr: str


@dataclass(slots=True)
class ListExpr(Expr):
    """列表字面量 - 如 [1, 2, 3] / List literal - e.g. [1, 2, 3]"""
    elements: list[Expr] = field(default_factory=list)


@dataclass(slots=True)
class DictExpr(Expr):
    """字典字面量 - 如 {"键": "值"} / Dict literal - e.g. {"键": "值"}"""
    pairs: list[tuple[Expr, Expr]] = field(default_factory=list)


@dataclass(slots=True)
class StringConcat(Expr):
    """字符串拼接 - 输出语句中逗号分隔的多个值 / String concatenation - comma-separated values in print statements"""
    left: Expr
    right: Expr


@dataclass(slots=True)
class AsyncCall(Expr):
    """异步调用 - 启动异步任务，返回AsyncTask / Async call - starts async task, returns AsyncTask"""
    name: str
    args: list[Expr] = field(default_factory=list)
    kwargs: dict[str, Expr] = field(default_factory=dict)


@dataclass(slots=True)
class AwaitExpr(Expr):
    """等待异步任务完成，获取结果 / Await async task completion, get result"""
    task: Expr


@dataclass(slots=True)
class ParallelCall(Expr):
    """并行调用多个函数，返回结果列表 / Call multiple functions in parallel, return results list"""
    calls: list[FuncCall] = field(default_factory=list)


@dataclass(slots=True)
class Assign(Stmt):
    """赋值语句 - 如 x = 10 / Assignment statement - e.g. x = 10"""
    target: str
    value: Expr


@dataclass(slots=True)
class IndexAssign(Stmt):
    """索引赋值 - 如 arr[0] = 10 / Index assignment - e.g. arr[0] = 10"""
    obj: Expr
    index: Expr
    value: Expr


@dataclass(slots=True)
class VarDecl(Stmt):
    """变量声明 - 如 定义 x = 5 或 定义 x: int = 5 / Variable declaration - e.g. 定义 x = 5 or 定义 x: int = 5"""
    name: str
    value: Expr
    type_hint: str | None = None


@dataclass(slots=True)
class ImportStmt(Stmt):
    """导入语句 - 支持 导入/从...导入 / Import statement - supports 导入/从...导入"""
    module: str
    alias: str | None = None
    items: list[str] | None = None


@dataclass(slots=True)
class FuncDef(Stmt):
    """函数定义 - 如 定义函数 加法(a, b=10, *args): / Function definition - e.g. 定义函数 加法(a, b=10, *args):
    
    params: [(参数名, 类型注解)] — 不含默认值信息
    defaults: {参数名: 默认值表达式} — 有默认值的参数
    variadic: 可变参数名 — 如 *args 中的 args，None表示无可变参数
    """
    name: str
    params: list[tuple[str, str | None]] = field(default_factory=list)
    body: list[Stmt] = field(default_factory=list)
    return_type: str | None = None
    defaults: dict[str, Expr] | None = None
    variadic: str | None = None


@dataclass(slots=True)
class If(Stmt):
    """条件语句 - 如果/否则如果/否则 / Conditional statement - 如果/否则如果/否则"""
    condition: Expr
    body: list[Stmt] = field(default_factory=list)
    orelse: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class While(Stmt):
    """当循环 - 当 条件: / While loop - 当 condition:"""
    condition: Expr
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class Loop(Stmt):
    """计数循环 - IR层固定次数循环 / Counted loop - fixed-iteration loop at IR level"""
    count: int
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class For(Stmt):
    """For-each循环 - 对于 x 在 可迭代对象: / For-each loop - 对于 x 在 iterable:"""
    var: str
    iterable: Expr
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class ForRange(Stmt):
    """范围循环 - 对于 i 从 1 到 10 步长 2: / Range loop - 对于 i 从 1 到 10 步长 2:"""
    var: str
    start: Expr
    stop: Expr
    step: Expr | None = None
    body: list[Stmt] = field(default_factory=list)


@dataclass(slots=True)
class Break(Stmt):
    """跳出循环（break）/ Break out of loop"""


@dataclass(slots=True)
class Continue(Stmt):
    """继续下一次循环（continue）/ Continue to next loop iteration"""


@dataclass(slots=True)
class Match(Stmt):
    """模式匹配 - 匹配/情况/默认 / Pattern matching - 匹配/情况/默认"""
    value: Expr
    cases: list[tuple[Expr, list[Stmt]]] = field(default_factory=list)
    default: list[Stmt] | None = None


@dataclass(slots=True)
class Print(Stmt):
    value: Expr
    values: list[Expr] = field(default_factory=list)


@dataclass(slots=True)
class Return(Stmt):
    """返回语句 - 返回 值 / Return statement - 返回 value"""
    value: Expr | None = None


@dataclass(slots=True)
class TryExcept(Stmt):
    """异常处理 - 尝试/捕获/最终 / Exception handling - 尝试/捕获/最终"""
    body: list[Stmt] = field(default_factory=list)
    handlers: list[tuple[str | None, str | None, list[Stmt]]] = field(default_factory=list)
    finally_body: list[Stmt] | None = None


@dataclass(slots=True)
class Raise(Stmt):
    """抛出异常 / Raise an exception"""
    value: Expr | None = None


@dataclass(slots=True)
class ClassDef(Stmt):
    """类定义 - 定义类 类名(基类): / Class definition - 定义类 ClassName(BaseClass):"""
    name: str
    bases: list[str] = field(default_factory=list)
    body: list[Stmt] = field(default_factory=list)

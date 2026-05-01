from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class NLType:
    """NLASM类型系统基类 / NLASM type system base class"""


@dataclass(slots=True, eq=True)
class IntType(NLType):
    """整数类型 / Integer type"""
    def __repr__(self) -> str:
        return "int"


@dataclass(slots=True, eq=True)
class FloatType(NLType):
    """浮点数类型 / Float type"""
    def __repr__(self) -> str:
        return "float"


@dataclass(slots=True, eq=True)
class StrType(NLType):
    """字符串类型 / String type"""
    def __repr__(self) -> str:
        return "str"


@dataclass(slots=True, eq=True)
class BoolType(NLType):
    """布尔类型 / Boolean type"""
    def __repr__(self) -> str:
        return "bool"


@dataclass(slots=True, eq=True)
class NoneType(NLType):
    """空值类型 / None type"""
    def __repr__(self) -> str:
        return "none"


@dataclass(slots=True)
class ArrayType(NLType):
    """数组类型 - 带元素类型参数 / Array type - with element type parameter"""
    element_type: NLType

    def __repr__(self) -> str:
        return f"list[{self.element_type}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ArrayType):
            return self.element_type == other.element_type
        return False


@dataclass(slots=True)
class DictType(NLType):
    """字典类型 - 带键值类型参数 / Dict type - with key-value type parameters"""
    key_type: NLType
    value_type: NLType

    def __repr__(self) -> str:
        return f"dict[{self.key_type}, {self.value_type}]"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DictType):
            return self.key_type == other.key_type and self.value_type == other.value_type
        return False


@dataclass(slots=True)
class FuncType(NLType):
    """函数类型 - 带参数类型和返回类型 / Function type - with parameter types and return type"""
    param_types: list[NLType]
    return_type: NLType

    def __repr__(self) -> str:
        params = ", ".join(str(p) for p in self.param_types)
        return f"({params}) -> {self.return_type}"


@dataclass(slots=True)
class AnyType(NLType):
    """任意类型 - 兼容所有类型 / Any type - compatible with all types"""
    def __repr__(self) -> str:
        return "any"


@dataclass(slots=True, eq=True)
class UnionType(NLType):
    """联合类型 - 多种类型之一 / Union type - one of multiple types"""
    types: list[NLType]

    def __repr__(self) -> str:
        return " | ".join(str(t) for t in self.types)


# 预定义类型单例 / Predefined type singletons
INT = IntType()
FLOAT = FloatType()
STR = StrType()
BOOL = BoolType()
NONE = NoneType()
ANY = AnyType()

# 类型名字符串到NLType的映射 / Type name string to NLType mapping
_TYPE_MAP: dict[str, NLType] = {
    "int": INT,
    "int64": INT,
    "float": FLOAT,
    "float64": FLOAT,
    "str": STR,
    "string": STR,
    "bool": BOOL,
    "none": NONE,
    "any": ANY,
}


def parse_type(type_str: str) -> NLType:
    """解析类型字符串为NLType对象（迭代式）/ Parse type string to NLType object (iterative).

    支持: int, float, str, bool, none, any, list[T], dict[K, V]
    """
    stack: list[str] = [type_str.strip()]
    results: list[NLType] = []
    while stack:
        s = stack.pop()
        if s in _TYPE_MAP:
            results.append(_TYPE_MAP[s])
        elif s.startswith("list[") and s.endswith("]"):
            inner = s[5:-1]
            stack.append("__array__")
            stack.append(inner)
        elif s.startswith("dict[") and s.endswith("]"):
            inner = s[5:-1]
            parts = inner.split(",", 1)
            if len(parts) == 2:
                stack.append("__dict__")
                stack.append(parts[1])
                stack.append(parts[0])
            else:
                results.append(DictType(key_type=ANY, value_type=ANY))
        elif s == "__array__":
            elem = results.pop()
            results.append(ArrayType(element_type=elem))
        elif s == "__dict__":
            val = results.pop()
            key = results.pop()
            results.append(DictType(key_type=key, value_type=val))
        else:
            results.append(ANY)
    return results[0] if results else ANY


def infer_type(value: Any) -> NLType:
    """从Python值推断NLType（迭代式）/ Infer NLType from Python value (iterative)"""
    stack: list[Any] = [value]
    results: list[NLType] = []
    while stack:
        v = stack.pop()
        if v is None:
            results.append(NONE)
        elif isinstance(v, bool):
            results.append(BOOL)
        elif isinstance(v, int):
            results.append(INT)
        elif isinstance(v, float):
            results.append(FLOAT)
        elif isinstance(v, str):
            results.append(STR)
        elif isinstance(v, list):
            if not v:
                results.append(ArrayType(element_type=ANY))
            else:
                stack.append(("__list__", len(v)))
                for item in reversed(v):
                    stack.append(item)
        elif isinstance(v, dict):
            if not v:
                results.append(DictType(key_type=ANY, value_type=ANY))
            else:
                k0 = next(iter(v))
                stack.append("__dict__")
                stack.append(v[k0])
                stack.append(k0)
        elif isinstance(v, tuple) and len(v) == 2 and v[0] == "__list__":
            n = v[1]
            if n == 0:
                results.append(ArrayType(element_type=ANY))
            else:
                items = results[-n:]
                del results[-n:]
                elem_type = items[0]
                for it in items[1:]:
                    if it != elem_type:
                        elem_type = ANY
                        break
                results.append(ArrayType(element_type=elem_type))
        elif v == "__dict__":
            val_type = results.pop()
            key_type = results.pop()
            results.append(DictType(key_type=key_type, value_type=val_type))
        else:
            results.append(ANY)
    return results[0] if results else ANY


class TypeChecker:
    """类型检查器 - 验证类型兼容性 / Type checker - validates type compatibility"""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def check(self, expected: NLType, actual: NLType, context: str = "") -> bool:
        """检查实际类型是否匹配期望类型（迭代式）/ Check if actual type matches expected type (iterative)"""
        work_stack: list[NLType] = [expected]

        while work_stack:
            current = work_stack.pop()
            if isinstance(current, AnyType) or isinstance(actual, AnyType):
                return True
            if current == actual:
                return True
            if isinstance(current, UnionType):
                work_stack.extend(current.types)
                continue
            msg = f"类型不匹配: 期望 {current}, 实际 {actual}"
            if context:
                msg += f" ({context})"
            self.errors.append(msg)
            return False

        return False

    def unify(self, t1: NLType, t2: NLType) -> NLType:
        """类型统一 - 推导两个类型的公共类型 / Type unification - derive common type of two types.

        规则: any统一为对方, int+float->float, 相同类型->自身, 其他->any
        Rules: any unifies to other, int+float->float, same type->self, else->any
        """
        if isinstance(t1, AnyType):
            return t2
        if isinstance(t2, AnyType):
            return t1
        if t1 == t2:
            return t1
        if isinstance(t1, IntType) and isinstance(t2, FloatType):
            return FLOAT
        if isinstance(t1, FloatType) and isinstance(t2, IntType):
            return FLOAT
        return ANY

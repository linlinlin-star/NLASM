from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PredicateOp = Literal[">", ">=", "<", "<=", "==", "!="]  # 谓词比较运算符类型 / Predicate comparison operator type


@dataclass(slots=True)
class ArraySlot:
    """数组槽位 - 存储数组名称和值 / Array slot - stores array name and values"""
    name: str
    values: list[int] | None = None


@dataclass(slots=True)
class PredicateSlot:
    """谓词槽位 - 存储比较运算符和阈值 / Predicate slot - stores comparison operator and threshold value"""
    op: PredicateOp
    value: int


@dataclass(slots=True)
class ScalarSlot:
    """标量槽位 - 存储单个整数值 / Scalar slot - stores a single integer value"""
    value: int

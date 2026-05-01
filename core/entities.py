from __future__ import annotations

from dataclasses import dataclass
from typing import Any

ENTITY_VAR = "VAR"       # 变量实体 / Variable entity
ENTITY_NUM = "NUM"       # 数字实体 / Number entity
ENTITY_ARRAY = "ARRAY"   # 数组实体 / Array entity
ENTITY_OP = "OP"         # 运算符实体 / Operator entity
ENTITY_INTENT = "INTENT" # 意图实体 / Intent entity


@dataclass(slots=True)
class Entity:
    """实体 - 从自然语言中提取的结构化信息 / Entity - structured information extracted from natural language.

    每个实体包含标签（类型）、值和在原文中的位置范围。
    Each entity contains a label (type), value, and position range in the original text.
    """
    label: str   # 实体类型标签 / Entity type label
    value: Any   # 实体值 / Entity value
    start: int   # 在原文中的起始位置 / Start position in original text
    end: int     # 在原文中的结束位置 / End position in original text

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < 0:
            raise ValueError("Entity start/end must be non-negative")
        if self.start > self.end:
            raise ValueError("Entity start must be <= end")

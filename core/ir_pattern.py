from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .ir import IRNode


@dataclass
class IRPattern:
    """IR模式定义 - 描述一种可识别的编程意图及其IR生成方式 / IR Pattern definition - describes a recognizable programming intent and its IR generation method.

    每个Pattern包含:
    - name: 唯一标识名 / Unique identifier name
    - description: 语义描述（用于向量编码）/ Semantic description (for vector encoding)
    - examples: 示例句子列表（用于向量编码）/ Example sentence list (for vector encoding)
    - slots: 槽位定义 {槽位名: 槽位类型} / Slot definitions {slot_name: slot_type}
    - constraints: 约束条件（如可向量化、可并行化）/ Constraints (e.g. vectorizable, parallelizable)
    - ir_builder: IR生成回调函数 / IR generation callback function
    - vector: 语义向量（由PatternMatcher计算）/ Semantic vector (computed by PatternMatcher)
    """
    name: str
    description: str
    examples: list[str]
    slots: dict[str, str]
    constraints: dict[str, Any] = field(default_factory=dict)
    ir_builder: Callable[[dict[str, Any]], list[IRNode]] | None = None
    vector: np.ndarray | None = None

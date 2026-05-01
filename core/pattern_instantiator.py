from __future__ import annotations

from typing import Any

from .ir import IRNode
from .ir_pattern import IRPattern


class PatternInstantiator:
    """Pattern实例化器 - 调用Pattern的ir_builder生成IR节点 / Pattern instantiator - calls Pattern's ir_builder to generate IR nodes.

    将填充好的槽位值传入ir_builder回调函数，生成可执行的IR节点列表。
    Passes filled slot values to the ir_builder callback, generating executable IR node list.
    """

    def instantiate(self, pattern: IRPattern, slots: dict[str, Any]) -> list[IRNode]:
        """实例化Pattern - 用槽位值生成IR节点 / Instantiate pattern - generate IR nodes with slot values"""
        if pattern.ir_builder is None:
            raise ValueError(f"Pattern {pattern.name} 缺少 ir_builder")
        result = pattern.ir_builder(slots)
        if not isinstance(result, list):
            raise TypeError(f"Pattern {pattern.name} 的 ir_builder 返回值不是列表")
        return result

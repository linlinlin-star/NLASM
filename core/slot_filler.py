from __future__ import annotations

import re
from typing import Any

from .entities import ENTITY_ARRAY, ENTITY_NUM, Entity
from .frontend import IntentPacket
from .ir_pattern import IRPattern
from .semantic_operators import SemanticOperatorMatcher
from .slot_types import ArraySlot, PredicateSlot


class SlotFiller:
    """槽位填充器 - 从IntentPacket中提取值填充Pattern的槽位 / Slot filler - extracts values from IntentPacket to fill Pattern slots.

    支持的槽位类型 / Supported slot types:
    - array: 数组槽位 / Array slot
    - predicate: 谓词槽位（比较运算）/ Predicate slot (comparison operation)
    - length: 数组长度（自动推导）/ Array length (auto-derived)
    - int: 整数槽位 / Integer slot

    如果提供 embedder，谓词提取优先使用语义匹配（FAISS），
    否则回退到正则匹配 / If embedder is provided, predicate extraction
    prefers semantic matching (FAISS), otherwise falls back to regex.
    """

    def __init__(self, embedder: Any | None = None) -> None:
        self._op_matcher: SemanticOperatorMatcher | None = None
        if embedder is not None:
            self._op_matcher = SemanticOperatorMatcher(embedder)

    def fill(self, pattern: IRPattern, packet: IntentPacket) -> dict[str, object]:
        """填充Pattern的所有槽位 / Fill all slots of a Pattern"""
        slots: dict[str, object] = {}
        needs_predicate = "predicate" in pattern.slots.values()
        needs_array = "array" in pattern.slots.values()

        array_entity = self._find_array(packet) if needs_array else None
        predicate = self._extract_predicate(packet.normalized) if needs_predicate else None

        used_ints: set[int] = set()
        if predicate is not None:
            used_ints.add(predicate.value)

        # 尝试提取特殊整数槽位 / Try to extract special integer slots
        repeat_count = self._extract_repeat_count(packet.normalized)
        factor = self._extract_factor(packet.normalized)

        for slot_name, slot_type in pattern.slots.items():
            if slot_type == "array":
                if array_entity is None:
                    raise ValueError("缺少数组槽位")
                slots[slot_name] = array_entity
            elif slot_type == "predicate":
                if predicate is None:
                    raise ValueError("缺少 predicate 槽位")
                slots[slot_name] = predicate
            elif slot_type == "length":
                # 从数组推导长度 / Derive length from array
                if array_entity is None:
                    raise ValueError("缺少数组槽位，无法推导长度")
                slots[slot_name] = self._infer_length(array_entity)
            elif slot_type == "int":
                if slot_name == "repeat_count" and repeat_count is not None:
                    slots[slot_name] = repeat_count
                    used_ints.add(repeat_count)
                elif slot_name == "factor" and factor is not None:
                    slots[slot_name] = factor
                    used_ints.add(factor)
                elif slot_name == "start":
                    start_val = self._extract_start(packet.normalized)
                    if start_val is not None:
                        slots[slot_name] = start_val
                        used_ints.add(start_val)
                    else:
                        slots[slot_name] = self._next_int(packet, used_ints)
                        used_ints.add(slots[slot_name])
                else:
                    slots[slot_name] = self._next_int(packet, used_ints)
                    used_ints.add(slots[slot_name])
            else:
                raise ValueError(f"未知槽位类型: {slot_type}")

        return slots

    def _find_array(self, packet: IntentPacket) -> ArraySlot:
        """从实体中查找数组 / Find array from entities"""
        for entity in packet.entities:
            if entity.label == ENTITY_ARRAY:
                name = "arr"
                return ArraySlot(name=name, values=entity.value)
        raise ValueError("缺少数组槽位 arr")

    def _extract_predicate(self, text: str) -> PredicateSlot | None:
        if self._op_matcher is not None:
            result = self._extract_predicate_semantic(text)
            if result is not None:
                return result
        return self._extract_predicate_regex(text)

    def _extract_predicate_semantic(self, text: str) -> PredicateSlot | None:
        """基于语义的操作符匹配 / Semantic-based operator matching.

        提取数字及其前文上下文，用 embedding 匹配最相似的操作符。
        """
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return None

        for num in numbers:
            # 排除常见介词"中"、"的"，提取2-6个字符作为操作符上下文
            match = re.search(rf'([^中的\d\s]{{2,6}})\s*{num}', text)
            if match:
                context = match.group(1).strip()
                if not context:
                    continue
                op = self._op_matcher.match(context)
                if op:
                    return PredicateSlot(op=op, value=int(num))

        return None

    def _extract_predicate_regex(self, text: str) -> PredicateSlot | None:
        """基于正则的操作符匹配（回退方案）/ Regex-based operator matching (fallback)."""
        patterns = [
            (r"大于等于\s*(\d+)", ">="),
            (r"小于等于\s*(\d+)", "<="),
            (r"不等于\s*(\d+)", "!="),
            (r"大于\s*(\d+)", ">"),
            (r"小于\s*(\d+)", "<"),
            (r"等于\s*(\d+)", "=="),
        ]
        for pat, op in patterns:
            match = re.search(pat, text)
            if match:
                return PredicateSlot(op=op, value=int(match.group(1)))
        return None

    def _extract_repeat_count(self, text: str) -> int | None:
        """提取重复次数 / Extract repeat count"""
        match = re.search(r"重复\s*(\d+)\s*次", text)
        if match:
            return int(match.group(1))
        match = re.search(r"循环\s*(\d+)\s*次", text)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s*次", text)
        if match:
            val = int(match.group(1))
            if val > 1:
                return val
        return None

    def _extract_factor(self, text: str) -> int | None:
        """提取乘法因子 / Extract multiplication factor"""
        match = re.search(r"乘以\s*(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _extract_start(self, text: str) -> int | None:
        """提取起始值 / Extract start value"""
        match = re.search(r"从\s*(\d+)", text)
        if match:
            return int(match.group(1))
        return None

    def _infer_length(self, array_slot: ArraySlot) -> int:
        """从数组槽位推导长度 / Infer length from array slot"""
        if array_slot.values is not None:
            return len(array_slot.values)
        raise ValueError("无法推导数组长度")

    def _next_int(self, packet: IntentPacket, exclude: set[int] | None = None) -> int:
        """获取下一个可用的整数实体 / Get next available integer entity"""
        exclude = exclude or set()
        for entity in packet.entities:
            if entity.label == ENTITY_NUM and entity.value not in exclude:
                return int(entity.value)
        raise ValueError("缺少 int 槽位")

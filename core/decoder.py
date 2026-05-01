from __future__ import annotations

from .frontend import IntentPacket
from .ir import Add, Assign, Literal, Loop, Print, Var
from .slot_types import ArraySlot


class SemanticDecoder:
    """语义解码器 - Pattern匹配失败时的降级路径 / Semantic decoder - fallback path when Pattern matching fails.

    根据提取的实体（数组、意图）直接生成IR节点。
    Generates IR nodes directly based on extracted entities (array, intent).
    """

    def decode(self, packet: IntentPacket) -> list:
        """解码IntentPacket为IR节点列表 / Decode IntentPacket to IR node list"""
        from .entities import ENTITY_ARRAY, ENTITY_INTENT

        array_entity = None
        intent_value = None

        for entity in packet.entities:
            if entity.label == ENTITY_ARRAY and array_entity is None:
                array_entity = entity
            if entity.label == ENTITY_INTENT and intent_value is None:
                intent_value = entity.value

        # 如果有数组且意图是求和，生成循环求和IR / If array present and intent is sum, generate loop-sum IR
        if array_entity is not None and array_entity.value:
            arr_name = "arr"
            n = len(array_entity.value)
            if intent_value == "sum":
                return [
                    Assign("acc", Literal(0)),
                    Assign("i", Literal(0)),
                    Loop(
                        count=n,
                        body=[
                            Assign("acc", Add(Var("acc"), Var(f"{arr_name}[i]"))),
                            Assign("i", Add(Var("i"), Literal(1))),
                        ],
                    ),
                    Print(Var("acc")),
                ]

        # 无法识别时返回默认输出 / Return default output when unrecognized
        return [Print(Literal(0))]

from __future__ import annotations

from typing import Any

from .ir import Add, Assign, Cmp, If, Literal, Loop, Mul, Print, Var
from .ir_pattern import IRPattern
from .slot_types import ArraySlot, PredicateSlot


def build_sum_array_ir(slots: dict[str, Any]) -> list:
    """构建数组求和IR - 遍历数组累加所有元素 / Build array sum IR - iterate and accumulate all elements"""
    arr: ArraySlot = slots["arr"]
    n: int = slots["N"]
    arr_name = arr.name

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


def build_filter_sum_ir(slots: dict[str, Any]) -> list:
    """构建过滤求和IR - 仅累加满足谓词条件的元素 / Build filter-sum IR - only accumulate elements satisfying predicate"""
    arr: ArraySlot = slots["arr"]
    pred: PredicateSlot = slots["predicate"]
    n: int = slots["N"]
    arr_name = arr.name

    return [
        Assign("acc", Literal(0)),
        Assign("i", Literal(0)),
        Loop(
            count=n,
            body=[
                If(
                    condition=Cmp(Var(f"{arr_name}[i]"), pred.op, Literal(pred.value)),
                    body=[
                        Assign("acc", Add(Var("acc"), Var(f"{arr_name}[i]"))),
                    ],
                ),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var("acc")),
    ]


def build_map_double_ir(slots: dict[str, Any]) -> list:
    """构建映射翻倍IR - 将每个元素乘以2 / Build map-double IR - multiply each element by 2"""
    arr: ArraySlot = slots["arr"]
    n: int = slots["N"]
    arr_name = arr.name

    return [
        Assign("i", Literal(0)),
        Loop(
            count=n,
            body=[
                Assign(f"{arr_name}[i]", Mul(Var(f"{arr_name}[i]"), Literal(2))),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var(arr_name)),
    ]


def build_repeat_multiply_ir(slots: dict[str, Any]) -> list:
    """构建重复乘法IR - 对每个元素重复乘以因子 / Build repeat-multiply IR - multiply each element by factor repeatedly"""
    arr: ArraySlot = slots["arr"]
    n: int = slots["N"]
    repeat: int = slots["repeat_count"]
    factor: int = slots.get("factor", 2)
    arr_name = arr.name

    return [
        Assign("i", Literal(0)),
        Loop(
            count=n,
            body=[
                Assign("k", Literal(0)),
                Loop(
                    count=repeat,
                    body=[
                        Assign(f"{arr_name}[i]", Mul(Var(f"{arr_name}[i]"), Literal(factor))),
                        Assign("k", Add(Var("k"), Literal(1))),
                    ],
                ),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var(arr_name)),
    ]


def build_count_loop_ir(slots: dict[str, Any]) -> list:
    """构建计数循环IR - 从起始值循环累加 / Build count-loop IR - loop and accumulate from start value"""
    repeat: int = slots["repeat_count"]
    start: int = slots.get("start", 1)

    return [
        Assign("acc", Literal(0)),
        Assign("i", Literal(start)),
        Loop(
            count=repeat,
            body=[
                Assign("acc", Add(Var("acc"), Var("i"))),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var("acc")),
    ]


# ============================================================
# Pattern数据库 - 预定义的5个IR模式 / Pattern database - 5 predefined IR patterns
# ============================================================
PATTERN_DB: list[IRPattern] = [
    IRPattern(
        name="sum_array",
        description="求数组所有元素的和",
        examples=[
            "求数组的总和",
            "计算列表所有元素的和",
            "sum all elements in an array",
            "数组求和",
            "列表元素累加",
        ],
        slots={"arr": "array", "N": "length"},
        constraints={"reduction": True, "pure": True},  # 可归约、纯函数 / Reducible, pure function
        ir_builder=build_sum_array_ir,
    ),
    IRPattern(
        name="filter_sum",
        description="过滤数组元素并求和",
        examples=[
            "计算数组中大于10的元素和",
            "求所有大于阈值的数之和",
            "sum values larger than threshold in an array",
            "过滤满足条件的元素并求和",
            "列表中满足条件的值累加",
            "大于某个数的元素求和",
        ],
        slots={"arr": "array", "predicate": "predicate", "N": "length"},
        constraints={"vectorizable": True, "reduction": True},  # 可向量化、可归约 / Vectorizable, reducible
        ir_builder=build_filter_sum_ir,
    ),
    IRPattern(
        name="map_double",
        description="将数组每个元素翻倍",
        examples=[
            "把数组每个元素翻倍",
            "将列表中所有值乘以2",
            "double every element in the array",
            "数组元素乘以二",
            "列表每个值翻倍",
        ],
        slots={"arr": "array", "N": "length"},
        constraints={"parallelizable": True, "pure": True},  # 可并行化、纯函数 / Parallelizable, pure function
        ir_builder=build_map_double_ir,
    ),
    IRPattern(
        name="repeat_multiply",
        description="对数组每个元素重复乘以某个数",
        examples=[
            "把数组每个元素乘以2重复3次",
            "将列表中所有值重复翻倍",
            "repeat multiply every element in the array",
            "数组元素循环乘法",
            "列表每个值重复乘",
        ],
        slots={"arr": "array", "N": "length", "repeat_count": "int", "factor": "int"},
        constraints={"parallelizable": True, "pure": True},
        ir_builder=build_repeat_multiply_ir,
    ),
    IRPattern(
        name="count_loop",
        description="从某个数开始循环累加",
        examples=[
            "从1循环累加5次",
            "循环10次求和",
            "loop and sum from 1 to N",
            "循环累加",
            "重复计数求和",
        ],
        slots={"repeat_count": "int", "start": "int"},
        constraints={"reduction": True, "pure": True},
        ir_builder=build_count_loop_ir,
    ),
]

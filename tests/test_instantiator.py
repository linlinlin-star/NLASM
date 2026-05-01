import pytest

from core.ir_pattern import IRPattern
from core.pattern_instantiator import PatternInstantiator
from core.slot_types import ArraySlot, PredicateSlot


def test_instantiate_normal_pattern() -> None:
    from core.pattern_database import build_sum_array_ir

    pattern = IRPattern(
        name="sum_array",
        description="",
        examples=[],
        slots={"arr": "array", "N": "length"},
        ir_builder=build_sum_array_ir,
    )
    instantiator = PatternInstantiator()
    slots = {"arr": ArraySlot(name="arr", values=[1, 2, 3]), "N": 3}
    nodes = instantiator.instantiate(pattern, slots)
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_instantiate_missing_builder_raises() -> None:
    pattern = IRPattern(
        name="no_builder",
        description="",
        examples=[],
        slots={},
        ir_builder=None,
    )
    instantiator = PatternInstantiator()
    with pytest.raises(ValueError, match="ir_builder"):
        instantiator.instantiate(pattern, {})


def test_instantiate_filter_sum() -> None:
    from core.pattern_database import build_filter_sum_ir

    pattern = IRPattern(
        name="filter_sum",
        description="",
        examples=[],
        slots={"arr": "array", "predicate": "predicate", "N": "length"},
        ir_builder=build_filter_sum_ir,
    )
    instantiator = PatternInstantiator()
    slots = {
        "arr": ArraySlot(name="arr", values=[1, 5, 12, 20]),
        "predicate": PredicateSlot(op=">", value=10),
        "N": 4,
    }
    nodes = instantiator.instantiate(pattern, slots)
    assert isinstance(nodes, list)
    assert len(nodes) > 0

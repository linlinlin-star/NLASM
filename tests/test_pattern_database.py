from core.pattern_database import PATTERN_DB, build_filter_sum_ir, build_map_double_ir, build_sum_array_ir
from core.slot_types import ArraySlot, PredicateSlot


def test_pattern_db_names_unique() -> None:
    names = [p.name for p in PATTERN_DB]
    assert len(names) == len(set(names))


def test_pattern_db_has_three_patterns() -> None:
    name_set = {p.name for p in PATTERN_DB}
    assert "sum_array" in name_set
    assert "filter_sum" in name_set
    assert "map_double" in name_set


def test_pattern_db_each_has_examples() -> None:
    for p in PATTERN_DB:
        assert len(p.examples) >= 3, f"{p.name} 需要至少3条examples"


def test_build_sum_array_ir() -> None:
    slots = {"arr": ArraySlot(name="arr", values=[1, 2, 3, 4]), "N": 4}
    nodes = build_sum_array_ir(slots)
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_build_filter_sum_ir() -> None:
    slots = {
        "arr": ArraySlot(name="arr", values=[1, 5, 12, 20]),
        "predicate": PredicateSlot(op=">", value=10),
        "N": 4,
    }
    nodes = build_filter_sum_ir(slots)
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_build_map_double_ir() -> None:
    slots = {"arr": ArraySlot(name="arr", values=[1, 2, 3]), "N": 3}
    nodes = build_map_double_ir(slots)
    assert isinstance(nodes, list)
    assert len(nodes) > 0

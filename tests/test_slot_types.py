from core.slot_types import ArraySlot, PredicateSlot, ScalarSlot


def test_array_slot_accepts_optional_values() -> None:
    slot = ArraySlot(name="arr", values=[1, 2, 3])
    assert slot.name == "arr"
    assert slot.values == [1, 2, 3]


def test_predicate_slot_is_structured() -> None:
    slot = PredicateSlot(op=">", value=10)
    assert slot.op == ">"
    assert slot.value == 10


def test_scalar_slot_stores_int() -> None:
    slot = ScalarSlot(value=7)
    assert slot.value == 7

import pytest

from core.entities import ENTITY_ARRAY, Entity


def test_entity_stores_values() -> None:
    entity = Entity(label=ENTITY_ARRAY, value=[1, 2, 3], start=2, end=9)
    assert entity.label == ENTITY_ARRAY
    assert entity.value == [1, 2, 3]
    assert entity.start == 2
    assert entity.end == 9


def test_entity_rejects_invalid_range() -> None:
    with pytest.raises(ValueError):
        Entity(label=ENTITY_ARRAY, value=[], start=5, end=1)

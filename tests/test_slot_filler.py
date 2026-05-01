import numpy as np
import pytest

from core.entities import ENTITY_ARRAY, ENTITY_NUM, ENTITY_OP, Entity
from core.frontend import Frontend, IntentPacket, RuleEntityExtractor
from core.ir_pattern import IRPattern
from core.slot_filler import SlotFiller
from core.slot_types import ArraySlot, PredicateSlot


def _make_packet(text: str) -> IntentPacket:
    class FakeEmbedder:
        def encode(self, t, normalize_embeddings=True):
            if isinstance(t, list):
                return np.asarray([[1.0, 0.0, 0.0] for _ in t], dtype=np.float32)
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    frontend = Frontend(embedder=FakeEmbedder())
    return frontend.process(text)


def test_find_array_extracts_array() -> None:
    filler = SlotFiller()
    packet = _make_packet("计算列表里大于10的元素和, 列表是[1,5,12,20]")
    arr = filler._find_array(packet)
    assert isinstance(arr, ArraySlot)
    assert arr.values == [1, 5, 12, 20]


def test_extract_predicate_long_first() -> None:
    filler = SlotFiller()
    pred = filler._extract_predicate("大于等于10")
    assert pred.op == ">="
    assert pred.value == 10


def test_extract_predicate_short() -> None:
    filler = SlotFiller()
    pred = filler._extract_predicate("大于5")
    assert pred.op == ">"
    assert pred.value == 5


def test_extract_predicate_fails_without_predicate() -> None:
    filler = SlotFiller()
    result = filler._extract_predicate("没有比较词的句子")
    assert result is None


def test_infer_length() -> None:
    filler = SlotFiller()
    arr = ArraySlot(name="arr", values=[1, 2, 3])
    assert filler._infer_length(arr) == 3


def test_infer_length_fails_without_values() -> None:
    filler = SlotFiller()
    arr = ArraySlot(name="arr", values=None)
    with pytest.raises(ValueError, match="长度"):
        filler._infer_length(arr)


def test_fill_filter_sum_pattern() -> None:
    filler = SlotFiller()
    packet = _make_packet("计算列表里大于10的元素和, 列表是[1,5,12,20]")
    pattern = IRPattern(
        name="filter_sum",
        description="",
        examples=[],
        slots={"arr": "array", "predicate": "predicate", "N": "length"},
    )
    slots = filler.fill(pattern, packet)
    assert isinstance(slots["arr"], ArraySlot)
    assert isinstance(slots["predicate"], PredicateSlot)
    assert slots["N"] == 4


def test_fill_missing_array_raises() -> None:
    filler = SlotFiller()
    packet = _make_packet("没有数组的句子")
    pattern = IRPattern(
        name="test",
        description="",
        examples=[],
        slots={"arr": "array"},
    )
    with pytest.raises(ValueError, match="数组"):
        filler.fill(pattern, packet)

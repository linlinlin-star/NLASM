from core.inline_cache import InlineCache, build_cache_key, normalize_type_signature
from core.slot_types import ArraySlot, PredicateSlot


def test_normalize_type_signature() -> None:
    slots = {
        "arr": ArraySlot(name="arr", values=[1, 2]),
        "predicate": PredicateSlot(op=">", value=10),
        "N": 4,
    }
    sig = normalize_type_signature(slots)
    assert "array<int64>" in sig
    assert "predicate<int64>" in sig
    assert "int64" in sig


def test_normalize_type_signature_stable() -> None:
    slots1 = {"arr": ArraySlot(name="arr", values=[1, 2]), "N": 4}
    slots2 = {"arr": ArraySlot(name="arr", values=[5, 6]), "N": 3}
    assert normalize_type_signature(slots1) == normalize_type_signature(slots2)


def test_build_cache_key() -> None:
    sig = ("array<int64>", "int64")
    key = build_cache_key("filter_sum", sig)
    assert key == "filter_sum::array<int64>|int64"


def test_inline_cache_hit_miss() -> None:
    cache = InlineCache()
    assert cache.lookup("nonexistent") is None
    assert cache.stats()["miss"] == 1

    cache.update("key1", "value1")
    assert cache.lookup("key1") == "value1"
    assert cache.stats()["hit"] == 1


def test_inline_cache_different_keys() -> None:
    cache = InlineCache()
    cache.update("key1", "value1")
    assert cache.lookup("key2") is None

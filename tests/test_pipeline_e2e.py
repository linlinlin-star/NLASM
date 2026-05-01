import numpy as np
import pytest

from core.decoder import SemanticDecoder
from core.frontend import Frontend, IntentPacket
from core.inline_cache import InlineCache, build_cache_key, normalize_type_signature
from core.ir_interpreter import IRInterpreter
from core.ir_pattern import IRPattern
from core.pattern_instantiator import PatternInstantiator
from core.pattern_matcher import PatternMatcher
from core.pipeline import PipelineV08
from core.slot_filler import SlotFiller
from core.slot_types import ArraySlot, PredicateSlot


class FakeEmbedder:
    def __init__(self, dim: int = 3):
        self.dim = dim
        self._rng = np.random.RandomState(42)

    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            result = self._rng.randn(len(text), self.dim).astype(np.float32)
        else:
            result = self._rng.randn(1, self.dim).astype(np.float32).flatten()
        norm = np.linalg.norm(result, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        result = result / norm
        return result


class ConstantEmbedder:
    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.asarray([[1.0, 0.0, 0.0] for _ in text], dtype=np.float32)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


def _build_pipeline() -> PipelineV08:
    from core.pattern_database import PATTERN_DB

    embedder = FakeEmbedder()
    frontend = Frontend(embedder=embedder)
    matcher = PatternMatcher(embedder=embedder, patterns=PATTERN_DB)
    filler = SlotFiller()
    instantiator = PatternInstantiator()
    decoder = SemanticDecoder()
    cache = InlineCache()

    return PipelineV08(
        frontend=frontend,
        matcher=matcher,
        filler=filler,
        instantiator=instantiator,
        decoder=decoder,
        cache=cache,
    )


def _build_constant_pipeline() -> PipelineV08:
    from core.pattern_database import PATTERN_DB

    embedder = ConstantEmbedder()
    frontend = Frontend(embedder=embedder)
    matcher = PatternMatcher(embedder=embedder, patterns=PATTERN_DB)
    filler = SlotFiller()
    instantiator = PatternInstantiator()
    decoder = SemanticDecoder()
    cache = InlineCache()

    return PipelineV08(
        frontend=frontend,
        matcher=matcher,
        filler=filler,
        instantiator=instantiator,
        decoder=decoder,
        cache=cache,
    )


def test_filter_sum_interpreter_direct() -> None:
    from core.pattern_database import build_filter_sum_ir

    arr = ArraySlot(name="arr", values=[1, 5, 12, 20])
    pred = PredicateSlot(op=">", value=10)
    slots = {"arr": arr, "predicate": pred, "N": 4}
    nodes = build_filter_sum_ir(slots)

    interp = IRInterpreter(arr_slot=arr)
    interp.run(nodes)
    assert interp.outputs[-1] == 32


def test_sum_array_interpreter_direct() -> None:
    from core.pattern_database import build_sum_array_ir

    arr = ArraySlot(name="arr", values=[1, 2, 3, 4])
    slots = {"arr": arr, "N": 4}
    nodes = build_sum_array_ir(slots)

    interp = IRInterpreter(arr_slot=arr)
    interp.run(nodes)
    assert interp.outputs[-1] == 10


def test_map_double_interpreter_direct() -> None:
    from core.pattern_database import build_map_double_ir

    arr = ArraySlot(name="arr", values=[1, 2, 3])
    slots = {"arr": arr, "N": 3}
    nodes = build_map_double_ir(slots)

    interp = IRInterpreter(arr_slot=arr)
    interp.run(nodes)
    assert interp.outputs[-1] == [2, 4, 6]


def test_fallback_path_no_crash() -> None:
    pipeline = _build_pipeline()
    result = pipeline.compile_and_run("把数组里满足条件的值做处理")
    assert result is not None


def test_cache_hit_on_second_call() -> None:
    cache = InlineCache()
    slots1 = {"arr": ArraySlot(name="arr", values=[1, 5, 12, 20]), "N": 4}
    slots2 = {"arr": ArraySlot(name="arr", values=[2, 8, 10, 20]), "N": 4}

    sig1 = normalize_type_signature(slots1)
    sig2 = normalize_type_signature(slots2)
    assert sig1 == sig2

    key1 = build_cache_key("sum_array", sig1)
    key2 = build_cache_key("sum_array", sig2)
    assert key1 == key2

    cache.update(key1, "compiled_entry")
    assert cache.lookup(key2) is not None
    assert cache.stats()["hit"] >= 1


def test_compile_only_returns_ir() -> None:
    pipeline = _build_pipeline()
    nodes = pipeline.compile_only("计算列表里大于10的元素和, 列表是[1,5,12,20]")
    assert isinstance(nodes, list)
    assert len(nodes) > 0


def test_pipeline_runs_without_error() -> None:
    pipeline = _build_pipeline()
    for text in [
        "计算列表里大于10的元素和, 列表是[1,5,12,20]",
        "求数组[1,2,3,4]的求和",
        "把数组[1,2,3]每个元素翻倍",
    ]:
        result = pipeline.compile_and_run(text)
        assert result is not None

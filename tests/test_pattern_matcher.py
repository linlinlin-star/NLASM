import numpy as np
import pytest

from core.ir_pattern import IRPattern
from core.pattern_matcher import PatternMatcher


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.asarray([[1.0, 0.0, 0.0] for _ in text], dtype=np.float32)
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)


class LowScoreEmbedder:
    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.asarray([[0.0, 1.0, 0.0] for _ in text], dtype=np.float32)
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float32)


def _make_patterns():
    from core.pattern_database import PATTERN_DB
    return PATTERN_DB


def test_matcher_rejects_empty_patterns() -> None:
    with pytest.raises(ValueError):
        PatternMatcher(embedder=FakeEmbedder(), patterns=[])


def test_matcher_returns_top_k() -> None:
    patterns = _make_patterns()
    matcher = PatternMatcher(embedder=FakeEmbedder(), patterns=patterns)
    results = matcher.match("求数组中大于10的元素和", top_k=3)
    assert len(results) <= 3
    assert len(results) > 0
    for p, score in results:
        assert isinstance(p, IRPattern)
        assert isinstance(score, float)


def test_match_best_returns_none_below_threshold() -> None:
    patterns = _make_patterns()
    matcher = PatternMatcher(embedder=LowScoreEmbedder(), patterns=patterns)
    result = matcher.match_best("完全不相关的输入xyz", threshold=0.99)
    assert result is None


def test_matcher_builds_faiss_index() -> None:
    patterns = _make_patterns()
    matcher = PatternMatcher(embedder=FakeEmbedder(), patterns=patterns)
    assert matcher.index is not None
    assert matcher.index.ntotal == len(patterns)

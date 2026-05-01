import numpy as np

from core.entities import ENTITY_ARRAY, ENTITY_INTENT, ENTITY_NUM, ENTITY_OP
from core.frontend import Frontend, RuleEntityExtractor


class FakeEmbedder:
    def encode(self, text, normalize_embeddings=True):
        if isinstance(text, list):
            return np.asarray([[1.0, 0.0, 0.0] for _ in text], dtype=np.float32)
        return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)


def test_normalize_rewrites_and_strips_noise() -> None:
    frontend = Frontend(embedder=FakeEmbedder())
    normalized = frontend._normalize("请帮我 计算数组里超过10的总和")
    assert normalized == "计算数组里大于10的求和"


def test_rule_extractor_extracts_array_num_intent_and_op() -> None:
    extractor = RuleEntityExtractor()
    entities = extractor.extract("计算列表里大于10的元素和, 列表是[1,5,12,20]")
    labels = [entity.label for entity in entities]

    assert ENTITY_ARRAY in labels
    assert ENTITY_NUM in labels
    assert ENTITY_OP in labels
    assert any(entity.label == ENTITY_INTENT and entity.value == "sum" for entity in entities)


def test_process_builds_intent_packet() -> None:
    frontend = Frontend(embedder=FakeEmbedder())
    packet = frontend.process("计算列表里大于10的元素和，列表是[1,5,12,20]")
    assert packet.raw.startswith("计算列表")
    assert packet.normalized.startswith("计算列表")
    assert packet.vector.shape == (3,)
    assert packet.metadata["entity_count"] >= 3
    assert any(entity.label == ENTITY_ARRAY for entity in packet.entities)

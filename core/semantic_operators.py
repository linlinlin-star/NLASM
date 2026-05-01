from __future__ import annotations

from typing import Any

import faiss
import numpy as np


OPERATOR_SEMANTICS: dict[str, list[str]] = {
    ">": ["大于", "超过", "多于", "高于", "比...大", "超出"],
    "<": ["小于", "少于", "低于", "不足", "比...小", "低于"],
    ">=": ["大于等于", "不小于", "至少", "最少"],
    "<=": ["小于等于", "不大于", "至多", "最多"],
    "==": ["等于", "是", "相等", "一样"],
    "!=": ["不等于", "不是", "不同", "不一样"],
}


class SemanticOperatorMatcher:
    """基于 FAISS 的语义操作符匹配器 / FAISS-based semantic operator matcher.

    每个操作符用其同义词的 embedding 质心向量表示，
    查询时用内积（等价余弦）检索最相似的操作符。
    """

    def __init__(self, embedder: Any) -> None:
        self.embedder = embedder
        self.operators: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self._build_index()

    def _build_index(self) -> None:
        vectors: list[np.ndarray] = []
        for op, synonyms in OPERATOR_SEMANTICS.items():
            embs = self.embedder.encode(synonyms, normalize_embeddings=True)
            embs = np.asarray(embs, dtype=np.float32)
            centroid = embs.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            vectors.append(centroid)
            self.operators.append(op)

        data = np.asarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(data.shape[1])
        self.index.add(data)

    def match(self, text: str, threshold: float = 0.6) -> str | None:
        """匹配操作符，返回 >, <, == 等 / Match operator, return >, <, ==, etc."""
        query = self.embedder.encode(text, normalize_embeddings=True)
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)

        scores, indices = self.index.search(query, 1)
        if scores[0][0] >= threshold:
            return self.operators[indices[0][0]]
        return None

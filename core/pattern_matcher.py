from __future__ import annotations

import faiss
import numpy as np

from .ir_pattern import IRPattern


class PatternMatcher:
    """基于FAISS的Pattern匹配器 / FAISS-based Pattern matcher.

    使用内积相似度（IP）进行语义匹配，每个Pattern用其描述和示例的向量质心表示。
    Uses inner product similarity for semantic matching, each Pattern represented by the centroid of its description and example vectors.
    """

    QUERY_PREFIX = "编程操作意图: "  # 查询前缀 - 提升匹配质量 / Query prefix - improves match quality

    def __init__(self, embedder, patterns: list[IRPattern]):
        if not patterns:
            raise ValueError("Pattern 数据库不能为空")
        self.embedder = embedder
        self.patterns = patterns
        self.index: faiss.IndexFlatIP | None = None
        self._build_faiss_index()

    def _build_faiss_index(self) -> None:
        """构建FAISS内积索引 / Build FAISS inner product index.

        对每个Pattern，编码其描述和示例，计算质心向量作为索引表示。
        For each Pattern, encode its description and examples, compute centroid vector as index representation.
        """
        vectors = []
        for pattern in self.patterns:
            texts = [pattern.description, *pattern.examples]
            embs = self.embedder.encode(texts, normalize_embeddings=True)
            centroid = np.asarray(embs, dtype=np.float32).mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # 归一化质心 / Normalize centroid
            pattern.vector = centroid.astype(np.float32)
            vectors.append(pattern.vector)

        data = np.asarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(data.shape[1])  # 内积索引 / Inner product index
        self.index.add(data)

    def match(self, text: str, top_k: int = 3) -> list[tuple[IRPattern, float]]:
        """匹配最相似的Pattern / Match most similar patterns.

        返回top_k个(pattern, score)对，score为内积相似度。
        Returns top_k (pattern, score) pairs, score is inner product similarity.
        """
        query = self._encode_query(text)
        k = min(top_k, len(self.patterns))
        scores, indices = self.index.search(query, k)
        return [
            (self.patterns[int(i)], float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]

    def match_best(self, text: str, threshold: float = 0.70) -> IRPattern | None:
        """匹配最佳Pattern，低于阈值返回None / Match best pattern, return None if below threshold"""
        results = self.match(text, top_k=1)
        if not results:
            return None
        pattern, score = results[0]
        if score >= threshold:
            return pattern
        return None

    def _encode_query(self, text: str) -> np.ndarray:
        """编码查询文本为向量 / Encode query text to vector"""
        vec = self.embedder.encode(
            self.QUERY_PREFIX + text,
            normalize_embeddings=True,
        )
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

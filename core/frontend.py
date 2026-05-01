from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import numpy as np

from .entities import (
    ENTITY_ARRAY,
    ENTITY_INTENT,
    ENTITY_NUM,
    ENTITY_OP,
    Entity,
)


# 噪声词模式 - 在规范化时移除 / Noise patterns - removed during normalization
NOISE_PATTERNS = [
    "请帮我",
    "帮我",
    "麻烦你",
    "帮我算一下",
    "请",
]

# 同义词重写规则 - 统一语义表达 / Synonym rewrite rules - unify semantic expressions
NORMALIZE_REWRITES = [
    ("超过", "大于"),
    ("高于", "大于"),
    ("低于", "小于"),
    ("不少于", "大于等于"),
    ("不多于", "小于等于"),
    ("求总和", "求和"),
    ("总和", "求和"),
]

# 意图关键词映射 / Intent keyword mapping
INTENT_KEYWORDS = {
    "求和": "sum",
    "总和": "sum",
    "元素和": "sum",
    "之和": "sum",
    "翻倍": "double",
    "计数": "count",
    "过滤": "filter",
    "排序": "sort",
    "最大值": "max",
    "最小值": "min",
    "平均值": "avg",
    "反转": "reverse",
    "去重": "unique",
    "合并": "merge",
    "查找": "find",
    "包含": "contains",
    "删除": "remove",
    "添加": "append",
    "插入": "insert",
    "替换": "replace",
    "分组": "group",
    "映射": "map",
    "累加": "accumulate",
    "乘积": "product",
    "差集": "difference",
    "交集": "intersection",
    "并集": "union",
}

# 比较运算符模式 / Comparison operator patterns
OP_PATTERNS = [
    ("大于等于", ">="),
    ("小于等于", "<="),
    ("不等于", "!="),
    ("大于", ">"),
    ("小于", "<"),
    ("等于", "=="),
]

_KEYWORD_HIT_SET: set[str] = set(INTENT_KEYWORDS.keys())
_OP_KEYWORD_SET: set[str] = {kw for kw, _ in OP_PATTERNS}
_FULL_KEYWORD_SET = _KEYWORD_HIT_SET | _OP_KEYWORD_SET

ARRAY_PATTERN = re.compile(r"\[(.*?)\]")  # 数组模式 / Array pattern
NUM_PATTERN = re.compile(r"-?\d+")  # 数字模式 / Number pattern


@dataclass(slots=True)
class IntentPacket:
    """意图数据包 - 前端处理的输出 / Intent packet - output of frontend processing.

    包含规范化文本、语义向量、提取的实体和语义骨架。
    Contains normalized text, semantic vector, extracted entities, and semantic skeleton.
    """
    raw: str  # 原始输入 / Raw input
    normalized: str  # 规范化文本 / Normalized text
    vector: np.ndarray  # 语义向量 / Semantic vector
    entities: list[Entity] = field(default_factory=list)  # 提取的实体 / Extracted entities
    metadata: dict[str, Any] = field(default_factory=dict)  # 元数据 / Metadata
    semantic_skeleton: str = ""  # 语义骨架（去除数组后）/ Semantic skeleton (arrays removed)


class RuleEntityExtractor:
    """基于规则的实体提取器 - 中文MVP路径 / Rule-based entity extractor for Chinese MVP path.

    提取四类实体: 数组(ARRAY)、数字(NUM)、意图(INTENT)、运算符(OP)
    Extracts four entity types: array(ARRAY), number(NUM), intent(INTENT), operator(OP)
    """

    def extract(self, text: str) -> list[Entity]:
        """从文本中提取所有实体 / Extract all entities from text"""
        entities = []
        arrays = self._extract_arrays(text)
        entities.extend(arrays)
        # 提取数字时排除数组内的数字 / Exclude numbers inside arrays when extracting
        entities.extend(self._extract_numbers(text, arrays))
        entities.extend(self._extract_intents(text))
        entities.extend(self._extract_ops(text))
        return sorted(entities, key=lambda item: (item.start, item.end))

    def _extract_arrays(self, text: str) -> list[Entity]:
        """提取数组实体 - 如 [1, 2, 3] / Extract array entities"""
        result: list[Entity] = []
        for match in ARRAY_PATTERN.finditer(text):
            raw = match.group(1).strip()
            if not raw:
                values: list[int] = []
            else:
                values = [int(x.strip()) for x in raw.split(",") if x.strip()]
            result.append(
                Entity(
                    label=ENTITY_ARRAY,
                    value=values,
                    start=match.start(),
                    end=match.end(),
                )
            )
        return result

    def _extract_numbers(self, text: str, arrays: list[Entity]) -> list[Entity]:
        """提取数字实体 - 排除数组内的数字 / Extract number entities - exclude numbers inside arrays"""
        excluded_ranges = [(entity.start, entity.end) for entity in arrays]
        result: list[Entity] = []
        for match in NUM_PATTERN.finditer(text):
            start, end = match.start(), match.end()
            if any(range_start <= start and end <= range_end for range_start, range_end in excluded_ranges):
                continue
            result.append(
                Entity(
                    label=ENTITY_NUM,
                    value=int(match.group(0)),
                    start=start,
                    end=end,
                )
            )
        return result

    def _extract_intents(self, text: str) -> list[Entity]:
        """提取意图实体 - 如 求和、翻倍 / Extract intent entities"""
        result: list[Entity] = []
        for keyword, value in INTENT_KEYWORDS.items():
            for match in re.finditer(re.escape(keyword), text):
                result.append(
                    Entity(
                        label=ENTITY_INTENT,
                        value=value,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return result

    def _extract_ops(self, text: str) -> list[Entity]:
        """提取运算符实体 - 如 大于、小于 / Extract operator entities"""
        result: list[Entity] = []
        for keyword, op in OP_PATTERNS:
            for match in re.finditer(re.escape(keyword), text):
                result.append(
                    Entity(
                        label=ENTITY_OP,
                        value=op,
                        start=match.start(),
                        end=match.end(),
                    )
                )
        return result


class Frontend:
    """前端处理器 - 规范化、实体抽取、向量编码 / Frontend processor - normalization, entity extraction, vector encoding.

    处理流水线 / Processing pipeline:
    原始文本 -> 规范化 -> 实体抽取 -> 语义骨架构建 -> 向量编码 -> IntentPacket
    Raw text -> Normalize -> Extract entities -> Build skeleton -> Encode vector -> IntentPacket
    """

    def __init__(self, embedder: Any = None, entity_extractor: RuleEntityExtractor | None = None):
        self.embedder = embedder
        self.entity_extractor = entity_extractor or RuleEntityExtractor()

    def _keyword_hit(self, text: str) -> bool:
        for kw in _FULL_KEYWORD_SET:
            if kw in text:
                return True
        return False

    def process(self, text: str) -> IntentPacket:
        normalized = self._normalize(text)
        entities = self.entity_extractor.extract(normalized)
        skeleton = self._build_skeleton(normalized, entities)

        if self._keyword_hit(normalized):
            vector = np.zeros(384, dtype=np.float32)
            metadata = {"entity_count": len(entities), "keyword_hit": True}
        elif self.embedder is not None:
            vector = np.asarray(
                self.embedder.encode(skeleton, normalize_embeddings=True),
                dtype=np.float32,
            )
            metadata = {"entity_count": len(entities), "keyword_hit": False}
        else:
            vector = np.zeros(384, dtype=np.float32)
            metadata = {"entity_count": len(entities), "keyword_hit": False, "no_embedder": True}

        return IntentPacket(
            raw=text,
            normalized=normalized,
            vector=vector,
            entities=entities,
            metadata=metadata,
            semantic_skeleton=skeleton,
        )

    def _build_skeleton(self, text: str, entities: list[Entity]) -> str:
        """构建语义骨架 - 移除数组实体后的文本 / Build semantic skeleton - text with array entities removed"""
        skeleton = text
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            if entity.label == ENTITY_ARRAY:
                skeleton = skeleton[:entity.start] + skeleton[entity.end:]
        skeleton = re.sub(r"\s+", " ", skeleton).strip()
        skeleton = re.sub(r"[,，]\s*$", "", skeleton)
        return skeleton

    def _normalize(self, text: str) -> str:
        """规范化文本 - 全角转半角、去噪、同义词统一 / Normalize text - full-to-half width, denoise, synonym unification"""
        normalized = self._to_half_width(text)
        normalized = self._strip_noise(normalized)
        for src, dst in NORMALIZE_REWRITES:
            normalized = normalized.replace(src, dst)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _to_half_width(self, text: str) -> str:
        """全角字符转半角 / Convert full-width characters to half-width"""
        chars = []
        for ch in text:
            code = ord(ch)
            if code == 0x3000:  # 全角空格 / Full-width space
                chars.append(" ")
            elif 0xFF01 <= code <= 0xFF5E:  # 全角ASCII / Full-width ASCII
                chars.append(chr(code - 0xFEE0))
            else:
                chars.append(ch)
        return "".join(chars)

    def _strip_noise(self, text: str) -> str:
        """移除噪声词 / Remove noise words"""
        cleaned = text
        for pattern in NOISE_PATTERNS:
            cleaned = cleaned.replace(pattern, "")
        return cleaned.strip(" ，,")

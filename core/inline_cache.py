from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

from .slot_types import ArraySlot, PredicateSlot


MAX_CACHE_ENTRIES = 512  # 增大缓存容量 8x / Increased cache capacity 8x


def normalize_type_signature(slots: dict[str, object]) -> tuple[str, ...]:
    """规范化类型签名 / Normalize type signature"""
    sig: list[str] = []
    for key in sorted(slots.keys()):
        value = slots[key]
        if isinstance(value, ArraySlot):
            sig.append("array<int64>" if value.values else "array<any>")
        elif isinstance(value, PredicateSlot):
            sig.append("predicate<int64>")
        elif isinstance(value, int):
            sig.append("int64")
        else:
            sig.append(type(value).__name__)
    return tuple(sig)


def build_cache_key(pattern_name: str, type_sig: tuple[str, ...]) -> str:
    """构建缓存键 — 使用xxhash快速哈希 / Build cache key — using fast hash.

    对长键使用SHA256截断，对短键直接拼接，兼顾速度和唯一性。
    For long keys use SHA256 truncation, for short keys direct concatenation,
    balancing speed and uniqueness.
    """
    raw = f"{pattern_name}::{'|'.join(type_sig)}"
    if len(raw) > 64:
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    return raw


class InlineCache:
    """多态内联缓存（PIC）— 高性能版 / Polymorphic Inline Cache — high-performance version.

    优化点 / Optimizations:
    1. OrderedDict LRU — O(1) 淘汰，替代 list.remove() 的 O(n)
    2. 512条目容量 — 8x扩容，减少淘汰频率
    3. 快速哈希键 — 长键SHA256截断，短键直接拼接
    4. 预热接口 — warmup() 预注册常用函数
    5. 多态限制 — 每个调用点最多4种类型（V8策略），超出降级为megamorphic

    1. OrderedDict LRU — O(1) eviction, replaces O(n) list.remove()
    2. 512 entries — 8x capacity, reduces eviction frequency
    3. Fast hash key — SHA256 truncation for long keys, direct concat for short
    4. Warmup interface — warmup() pre-registers common functions
    5. Polymorphism limit — max 4 types per call site (V8 strategy), megamorphic beyond
    """

    MAX_POLYMORPHIC = 4  # 多态上限 / Polymorphism limit per call site

    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES) -> None:
        self.cache: OrderedDict[str, Any] = OrderedDict()  # LRU缓存 / LRU cache
        self.max_entries = max_entries
        self.hit: int = 0
        self.miss: int = 0
        self._call_site_types: dict[str, set[str]] = {}  # 调用点类型追踪 / Call site type tracking

    def lookup(self, cache_key: str) -> Any | None:
        """查找缓存 — O(1) LRU更新 / Look up cache — O(1) LRU update"""
        if cache_key in self.cache:
            self.hit += 1
            self.cache.move_to_end(cache_key)  # O(1) 移到最近 / O(1) move to most recent
            return self.cache[cache_key]
        self.miss += 1
        return None

    def update(self, cache_key: str, compiled_entry: Any) -> None:
        """更新缓存 — O(1) LRU淘汰 / Update cache — O(1) LRU eviction"""
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            self.cache[cache_key] = compiled_entry
            return

        # 超过容量淘汰最旧 / Evict oldest when over capacity
        while len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)  # O(1) 淘汰最旧 / O(1) evict oldest

        self.cache[cache_key] = compiled_entry

    def warmup(self, entries: dict[str, Any]) -> None:
        """预热缓存 — 批量注册常用编译结果 / Warmup cache — batch register common compiled results.

        在启动时预加载标准库函数的编译结果，避免首次调用的冷启动延迟。
        Preload compiled results for stdlib functions at startup,
        avoiding cold-start latency on first call.
        """
        for key, value in entries.items():
            if key not in self.cache:
                if len(self.cache) >= self.max_entries:
                    self.cache.popitem(last=False)
                self.cache[key] = value

    def record_call_site_type(self, call_site: str, type_sig: str) -> bool:
        """记录调用点的类型签名 — 用于多态检测 / Record call site type signature — for polymorphism detection.

        返回True表示仍在多态范围内，False表示已降级为megamorphic。
        Returns True if still within polymorphic limit, False if degraded to megamorphic.
        """
        if call_site not in self._call_site_types:
            self._call_site_types[call_site] = set()
        types = self._call_site_types[call_site]
        types.add(type_sig)
        return len(types) <= self.MAX_POLYMORPHIC

    def is_megamorphic(self, call_site: str) -> bool:
        """检查调用点是否已降级为megamorphic / Check if call site has degraded to megamorphic"""
        types = self._call_site_types.get(call_site, set())
        return len(types) > self.MAX_POLYMORPHIC

    def stats(self) -> dict[str, Any]:
        """获取缓存统计信息 / Get cache statistics"""
        total = self.hit + self.miss
        hit_rate = (self.hit / total * 100) if total > 0 else 0.0
        return {
            "hit": self.hit,
            "miss": self.miss,
            "hit_rate": f"{hit_rate:.1f}%",
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "megamorphic_sites": sum(
                1 for types in self._call_site_types.values()
                if len(types) > self.MAX_POLYMORPHIC
            ),
        }

    def clear(self) -> None:
        """清空缓存 / Clear cache"""
        self.cache.clear()
        self._call_site_types.clear()
        self.hit = 0
        self.miss = 0

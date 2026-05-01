from __future__ import annotations

from typing import Any, NamedTuple


class SymbolEntry(NamedTuple):
    """符号条目 - 用命名元组替代字典，减少内存和GC压力 / Symbol entry - use NamedTuple instead of dict, reduce memory and GC pressure"""
    value: Any
    type_hint: str | None = None
    kind: str = "var"


class SymbolTable:
    """符号表 - 管理变量作用域和名称绑定 / Symbol table - manages variable scopes and name bindings.

    优化点:
    1. SymbolEntry 用 NamedTuple 替代 dict，每个符号节省 ~60% 内存
    2. get_or_none 合并 has()+get_value() 双重查找为单次查找
    3. try_get_value 避免 NameError 异常开销

    Optimizations:
    1. SymbolEntry uses NamedTuple instead of dict, ~60% memory savings per symbol
    2. get_or_none merges has()+get_value() double lookup into single lookup
    3. try_get_value avoids NameError exception overhead
    """

    __slots__ = ('symbols', 'parent')

    def __init__(self, parent: SymbolTable | None = None) -> None:
        self.symbols: dict[str, SymbolEntry] = {}  # 名称 -> SymbolEntry / Name -> SymbolEntry
        self.parent = parent  # 父作用域 / Parent scope

    def define(self, name: str, value: Any, type_hint: str | None = None, kind: str = "var") -> None:
        """定义新符号 / Define a new symbol"""
        self.symbols[name] = SymbolEntry(value=value, type_hint=type_hint, kind=kind)

    def lookup(self, name: str) -> SymbolEntry | None:
        """查找符号 - 沿作用域链向上搜索（迭代式）/ Look up symbol - search up the scope chain (iterative)"""
        table: SymbolTable | None = self
        while table is not None:
            if name in table.symbols:
                return table.symbols[name]
            table = table.parent
        return None

    def get_value(self, name: str) -> Any:
        """获取符号值 / Get symbol value"""
        entry = self.symbols.get(name)
        if entry is not None:
            return entry.value
        table: SymbolTable | None = self.parent
        while table is not None:
            entry = table.symbols.get(name)
            if entry is not None:
                return entry.value
            table = table.parent
        raise NameError(f"未定义变量: {name}")

    def get_or_none(self, name: str) -> Any:
        """获取符号值或None — 合并 has()+get_value() 为单次查找 / Get symbol value or None — merges has()+get_value() into single lookup"""
        entry = self.symbols.get(name)
        if entry is not None:
            return entry.value
        table: SymbolTable | None = self.parent
        while table is not None:
            entry = table.symbols.get(name)
            if entry is not None:
                return entry.value
            table = table.parent
        return None

    def set_value(self, name: str, value: Any) -> None:
        """设置符号值 - 沿作用域链向上查找（迭代式）/ Set symbol value - search up the scope chain (iterative)"""
        table: SymbolTable | None = self
        while table is not None:
            if name in table.symbols:
                old = table.symbols[name]
                table.symbols[name] = SymbolEntry(value=value, type_hint=old.type_hint, kind=old.kind)
                return
            table = table.parent
        raise NameError(f"未定义变量: {name}")

    def has(self, name: str) -> bool:
        """检查符号是否存在（迭代式）/ Check if symbol exists (iterative)"""
        table: SymbolTable | None = self
        while table is not None:
            if name in table.symbols:
                return True
            table = table.parent
        return False

    def define_or_set(self, name: str, value: Any, type_hint: str | None = None, kind: str = "var") -> None:
        """定义或更新符号 — 合并 has()+define/set_value 为单次查找 / Define or update symbol — merges has()+define/set_value into single lookup"""
        if name in self.symbols:
            old = self.symbols[name]
            self.symbols[name] = SymbolEntry(value=value, type_hint=old.type_hint, kind=old.kind)
        else:
            self.symbols[name] = SymbolEntry(value=value, type_hint=type_hint, kind=kind)

    def enter_scope(self) -> SymbolTable:
        """进入子作用域 / Enter child scope"""
        return SymbolTable(parent=self)

    def exit_scope(self) -> SymbolTable | None:
        """退出当前作用域，返回父作用域 / Exit current scope, return parent scope"""
        return self.parent

    def all_names(self) -> set[str]:
        """获取所有可见的符号名（含父作用域，迭代式）/ Get all visible symbol names (including parent scopes, iterative)"""
        names = set(self.symbols.keys())
        table: SymbolTable | None = self.parent
        while table is not None:
            names |= table.symbols.keys()
            table = table.parent
        return names

    def local_names(self) -> set[str]:
        """获取当前作用域的符号名 / Get symbol names in current scope only"""
        return set(self.symbols.keys())

    def snapshot(self) -> SymbolTable:
        """创建独立快照 — 将所有可见符号展平到新表中，断开父链 / Create independent snapshot — flatten all visible symbols into new table, detach parent chain.

        用于并发场景：子线程需要访问父作用域变量，但不能共享可变状态。
        Used in concurrency: child threads need parent scope variables, but cannot share mutable state.
        """
        flat = SymbolTable()
        chain: list[SymbolTable] = []
        table: SymbolTable | None = self
        while table is not None:
            chain.append(table)
            table = table.parent
        for scope in reversed(chain):
            for name, entry in scope.symbols.items():
                flat.symbols[name] = entry
        return flat

    def snapshot(self) -> SymbolTable:
        """创建符号表的快照（深拷贝）/ Create a snapshot of the symbol table (deep copy)"""
        snapshot = SymbolTable(parent=None)
        for name, entry in self.symbols.items():
            snapshot.symbols[name] = entry
        return snapshot

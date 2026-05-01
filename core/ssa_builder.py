from __future__ import annotations

from dataclasses import dataclass, field

from .cfg import BasicBlock, CFG
from .dominator import DominatorTree
from .ir import Assign, Var


@dataclass
class PhiInstruction:
    """Phi函数指令 - SSA中合并不同控制流路径的变量版本 / Phi function instruction - merges variable versions from different control flow paths in SSA.

    格式: target = phi(block1: ver1, block2: ver2, ...)
    Format: target = phi(block1: ver1, block2: ver2, ...)
    """
    target: str
    sources: dict[str, str] = field(default_factory=dict)  # 前驱块名 -> 变量版本 / Predecessor block name -> variable version

    def __repr__(self) -> str:
        pairs = ", ".join(f"{blk}: {ver}" for blk, ver in self.sources.items())
        return f"{self.target} = phi({pairs})"


class SSABuilder:
    """SSA构建器 - 将CFG转换为静态单赋值形式 / SSA builder - converts CFG to Static Single Assignment form.

    使用Cytron算法:
    1. 收集变量定义点
    2. 在支配前沿插入Phi节点
    3. 重命名变量（添加版本号）

    Uses Cytron's algorithm:
    1. Collect variable definition sites
    2. Insert Phi nodes at dominance frontiers
    3. Rename variables (add version numbers)
    """

    def __init__(self) -> None:
        self.counters: dict[str, int] = {}  # 变量版本计数器 / Variable version counters
        self.stacks: dict[str, list[str]] = {}  # 变量版本栈 / Variable version stacks

    def build(self, cfg: CFG) -> CFG:
        """构建SSA形式 / Build SSA form"""
        dom_tree = DominatorTree()
        dom_tree.build(cfg)

        def_sites = self._collect_def_sites(cfg)
        self._insert_phi_nodes(cfg, dom_tree, def_sites)
        self._rename_variables(cfg, dom_tree)

        return cfg

    def _collect_def_sites(self, cfg: CFG) -> dict[str, set[BasicBlock]]:
        """收集每个变量的定义点（哪些基本块定义了该变量）/ Collect definition sites for each variable (which blocks define it)"""
        sites: dict[str, set[BasicBlock]] = {}
        for block in cfg.blocks:
            for instr in block.instructions:
                if isinstance(instr, Assign):
                    name = instr.target
                    sites.setdefault(name, set()).add(block)
        return sites

    def _insert_phi_nodes(
        self,
        cfg: CFG,
        dom_tree: DominatorTree,
        def_sites: dict[str, set[BasicBlock]],
    ) -> None:
        """在支配前沿插入Phi节点 / Insert Phi nodes at dominance frontiers.

        对每个变量，从其定义点出发，在支配前沿放置Phi节点。
        For each variable, starting from its definition sites, place Phi nodes at dominance frontiers.
        """
        for var, sites_set in def_sites.items():
            worklist = list(sites_set)
            placed: set[BasicBlock] = set()
            while worklist:
                block = worklist.pop()
                for frontier_block in dom_tree.frontier.get(block, set()):
                    if frontier_block not in placed:
                        phi = PhiInstruction(target=var)
                        frontier_block.instructions.insert(0, phi)
                        placed.add(frontier_block)
                        if frontier_block not in sites_set:
                            worklist.append(frontier_block)

    def _collect_all_vars(self, cfg: CFG) -> set[str]:
        """收集CFG中所有变量名 / Collect all variable names in CFG"""
        vars_set: set[str] = set()
        for block in cfg.blocks:
            for instr in block.instructions:
                if isinstance(instr, Assign):
                    vars_set.add(instr.target)
                if isinstance(instr, PhiInstruction):
                    vars_set.add(instr.target)
        return vars_set

    def _rename_variables(self, cfg: CFG, dom_tree: DominatorTree) -> None:
        """重命名变量 - 添加SSA版本号 / Rename variables - add SSA version numbers.

        使用Cytron的栈式重命名算法，遍历支配树（迭代式）。
        Uses Cytron's stack-based renaming algorithm, traversing the dominator tree (iterative).
        """
        self.counters = {}
        self.stacks = {}

        for var in self._collect_all_vars(cfg):
            self.counters[var] = 0
            self.stacks[var] = [var]

        work_stack: list[tuple[BasicBlock, bool, dict[str, int]]] = [(cfg.entry, False, {})]
        while work_stack:
            block, children_processed, old_stack_sizes = work_stack.pop()
            if not children_processed:
                old_stack_sizes = {}
                for instr in block.instructions:
                    if isinstance(instr, PhiInstruction):
                        ver = self._new_version(instr.target)
                        old_stack_sizes.setdefault(instr.target, len(self.stacks[instr.target]))
                        self.stacks[instr.target].append(ver)
                        instr.target = ver
                    if isinstance(instr, Assign):
                        self._rewrite_uses(instr, block)
                        ver = self._new_version(instr.target)
                        old_stack_sizes.setdefault(instr.target, len(self.stacks[instr.target]))
                        self.stacks[instr.target].append(ver)
                        instr.target = ver
                for succ in block.succs:
                    for instr in succ.instructions:
                        if isinstance(instr, PhiInstruction):
                            for pred in succ.preds:
                                if pred is block:
                                    orig_name = instr.target.rsplit("_", 1)[0] if "_" in instr.target else instr.target
                                    current = self.stacks.get(orig_name, [orig_name])
                                    instr.sources[block.name] = current[-1]
                work_stack.append((block, True, old_stack_sizes))
                children = [child for child in cfg.blocks if dom_tree.idom.get(child) is block]
                for child in reversed(children):
                    work_stack.append((child, False, {}))
            else:
                for var, old_size in old_stack_sizes.items():
                    self.stacks[var] = self.stacks[var][:old_size]

    def _new_version(self, name: str) -> str:
        """生成新的SSA版本号 / Generate new SSA version number"""
        base = name.rsplit("_", 1)[0] if "_" in name and name.rsplit("_", 1)[1].isdigit() else name
        self.counters.setdefault(base, 0)
        self.counters[base] += 1
        return f"{base}_{self.counters[base]}"

    def _rewrite_uses(self, instr: Assign, block: BasicBlock) -> None:
        """重写赋值指令中的变量使用 / Rewrite variable uses in assignment instruction"""
        if isinstance(instr.value, Var):
            current = self.stacks.get(instr.value.name, [instr.value.name])
            instr.value = Var(current[-1])

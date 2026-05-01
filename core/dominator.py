from __future__ import annotations

from .cfg import BasicBlock, CFG


class DominatorTree:
    """支配树 - 计算CFG的支配关系、直接支配者和支配前沿 / Dominator tree - computes dominance relations, immediate dominators, and dominance frontiers of CFG.

    支配(Dom): 若从entry到B的每条路径都经过A，则A支配B。
    直接支配(IDom): 支配B且最接近B的节点。
    支配前沿(DF): A的支配前沿中的块B，A支配B的某个前驱但不严格支配B。

    Dom: A dominates B if every path from entry to B passes through A.
    IDom: The closest dominator of B.
    DF: Block B is in A's dominance frontier if A dominates a predecessor of B but does not strictly dominate B.
    """

    def __init__(self) -> None:
        self.dom: dict[BasicBlock, set[BasicBlock]] = {}  # 支配集合 / Dominator sets
        self.idom: dict[BasicBlock, BasicBlock | None] = {}  # 直接支配者 / Immediate dominators
        self.frontier: dict[BasicBlock, set[BasicBlock]] = {}  # 支配前沿 / Dominance frontiers

    def build(self, cfg: CFG) -> None:
        """构建支配树 / Build dominator tree"""
        self._compute_dominators(cfg)
        self._compute_idom(cfg)
        self._compute_frontier(cfg)

    def _compute_dominators(self, cfg: CFG) -> None:
        """计算支配集合 - 迭代数据流算法 / Compute dominator sets - iterative dataflow algorithm.

        初始化: entry仅被自身支配，其他块被所有块支配。
        不动点迭代直到收敛。
        Initialize: entry dominated only by itself, others by all blocks.
        Iterate to fixed point.
        """
        all_blocks = set(cfg.blocks)
        self.dom = {}
        for block in cfg.blocks:
            if block is cfg.entry:
                self.dom[block] = {block}
            else:
                self.dom[block] = set(all_blocks)

        changed = True
        while changed:
            changed = False
            for block in cfg.blocks:
                if block is cfg.entry:
                    continue
                # 新支配集 = 所有前驱支配集的交集 ∪ 自身 / New dom = intersection of all predecessors' dom ∪ self
                if block.preds:
                    new_dom = set.intersection(*(self.dom[p] for p in block.preds))
                else:
                    new_dom = set(all_blocks)
                new_dom = new_dom | {block}
                if new_dom != self.dom[block]:
                    self.dom[block] = new_dom
                    changed = True

    def _compute_idom(self, cfg: CFG) -> None:
        """计算直接支配者 / Compute immediate dominators.

        IDom(B)是支配B且不支配其他支配B的节点的那个节点。
        IDom(B) is the dominator of B that doesn't dominate any other dominator of B.
        """
        self.idom = {}
        for block in cfg.blocks:
            if block is cfg.entry:
                self.idom[block] = None
                continue
            doms = self.dom[block] - {block}
            idom = None
            for d in doms:
                # d是IDom当且仅当d不支配doms中的其他节点 / d is IDom iff d doesn't dominate other nodes in doms
                is_idom = True
                for other in doms:
                    if other is not d and d in self.dom[other]:
                        is_idom = False
                        break
                if is_idom:
                    idom = d
                    break
            self.idom[block] = idom

    def _compute_frontier(self, cfg: CFG) -> None:
        """计算支配前沿 / Compute dominance frontiers.

        对于有多个前驱的块B，从每个前驱沿IDom链向上直到B的IDom，
        将B加入路径上所有节点的支配前沿。
        For block B with multiple predecessors, walk up the IDom chain from each predecessor
        until reaching B's IDom, adding B to the frontier of all nodes along the path.
        """
        self.frontier = {block: set() for block in cfg.blocks}
        for block in cfg.blocks:
            if len(block.preds) >= 2:
                for pred in block.preds:
                    runner = pred
                    while runner is not None and runner != self.idom.get(block):
                        self.frontier[runner].add(block)
                        runner = self.idom.get(runner)

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BasicBlock:
    """基本块 - CFG的基本单元 / Basic block - fundamental unit of CFG.

    包含一组顺序执行的指令，以及前驱和后继基本块的引用。
    Contains a group of sequentially executed instructions, and references to predecessor and successor blocks.
    """
    name: str
    instructions: list = field(default_factory=list)  # 指令列表 / Instruction list
    preds: list[BasicBlock] = field(default_factory=list)  # 前驱块列表 / Predecessor block list
    succs: list[BasicBlock] = field(default_factory=list)  # 后继块列表 / Successor block list

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


class CFG:
    """控制流图 - 表示程序的控制流结构 / Control Flow Graph - represents program control flow structure.

    由基本块和块间的控制流边组成，entry为入口基本块。
    Composed of basic blocks and control flow edges between them, entry is the entry block.
    """

    def __init__(self, entry: BasicBlock) -> None:
        self.entry = entry
        self.blocks: list[BasicBlock] = [entry]

    def add_block(self, block: BasicBlock) -> None:
        """添加基本块到CFG / Add basic block to CFG"""
        self.blocks.append(block)

    def connect(self, src: BasicBlock, dst: BasicBlock) -> None:
        """连接两个基本块 - 添加控制流边 / Connect two basic blocks - add control flow edge"""
        if dst not in src.succs:
            src.succs.append(dst)
        if src not in dst.preds:
            dst.preds.append(src)

    def dump(self) -> str:
        """导出CFG为文本格式 / Dump CFG to text format"""
        lines = []
        for block in self.blocks:
            preds = [b.name for b in block.preds]
            succs = [b.name for b in block.succs]
            lines.append(f"{block.name}: preds={preds} succs={succs}")
            for instr in block.instructions:
                lines.append(f"  {instr}")
        return "\n".join(lines)

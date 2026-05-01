from __future__ import annotations

from .cfg import BasicBlock, CFG
from .ir import Assign, If, Loop, Print, Return, Stmt


class CFGBuilder:
    """控制流图构建器 - 将IR节点转换为CFG / Control Flow Graph builder - converts IR nodes to CFG.

    将线性IR节点序列转换为包含基本块和控制流边的CFG，
    支持If分支和Loop循环的CFG结构化。
    Converts linear IR node sequences into CFG with basic blocks and control flow edges,
    supporting If branching and Loop cycle CFG structuring.
    """

    def __init__(self) -> None:
        self._counter: int = 0  # 基本块命名计数器 / Basic block naming counter

    def build(self, nodes: list) -> CFG:
        """从IR节点列表构建CFG / Build CFG from IR node list"""
        entry = BasicBlock(name="entry")
        cfg = CFG(entry)
        worklist: list[tuple[list, BasicBlock, BasicBlock | None]] = [(nodes, entry, None)]
        while worklist:
            stmts, current, after_block = worklist.pop()
            i = 0
            while i < len(stmts):
                stmt = stmts[i]
                if isinstance(stmt, (Assign, Print, Return)):
                    current.instructions.append(stmt)
                    i += 1
                elif isinstance(stmt, If):
                    self._counter += 1
                    then_block = BasicBlock(name=f"then_{self._counter}")
                    else_block = BasicBlock(name=f"else_{self._counter}")
                    join_block = BasicBlock(name=f"join_{self._counter}")
                    cfg.add_block(then_block)
                    cfg.add_block(else_block)
                    cfg.add_block(join_block)
                    current.instructions.append(stmt)
                    cfg.connect(current, then_block)
                    cfg.connect(current, else_block)
                    remaining = stmts[i + 1:]
                    if remaining:
                        worklist.append((remaining, join_block, after_block))
                    elif after_block is not None:
                        cfg.connect(join_block, after_block)
                    if stmt.orelse:
                        worklist.append((stmt.orelse, else_block, join_block))
                    else:
                        cfg.connect(else_block, join_block)
                    if stmt.body:
                        worklist.append((stmt.body, then_block, join_block))
                    else:
                        cfg.connect(then_block, join_block)
                    break
                elif isinstance(stmt, Loop):
                    self._counter += 1
                    loop_header = BasicBlock(name=f"loop_header_{self._counter}")
                    loop_body = BasicBlock(name=f"loop_body_{self._counter}")
                    loop_exit = BasicBlock(name=f"loop_exit_{self._counter}")
                    cfg.add_block(loop_header)
                    cfg.add_block(loop_body)
                    cfg.add_block(loop_exit)
                    loop_header.instructions.append(stmt)
                    cfg.connect(current, loop_header)
                    cfg.connect(loop_header, loop_body)
                    cfg.connect(loop_header, loop_exit)
                    cfg.connect(loop_body, loop_header)
                    if stmt.body:
                        worklist.append((stmt.body, loop_body, loop_header))
                    else:
                        cfg.connect(loop_body, loop_header)
                    remaining = stmts[i + 1:]
                    if remaining:
                        worklist.append((remaining, loop_exit, after_block))
                    elif after_block is not None:
                        cfg.connect(loop_exit, after_block)
                    break
                else:
                    current.instructions.append(stmt)
                    i += 1
        return cfg

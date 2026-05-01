from core.cfg import BasicBlock, CFG
from core.ir import Assign, Literal, Var
from core.ssa_builder import SSABuilder, PhiInstruction


def test_ssa_inserts_phi_at_join() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("x", Literal(1)))

    then_b = BasicBlock(name="then")
    then_b.instructions.append(Assign("x", Literal(2)))

    else_b = BasicBlock(name="else")
    else_b.instructions.append(Assign("x", Literal(3)))

    join = BasicBlock(name="join")

    cfg = CFG(entry)
    cfg.add_block(then_b)
    cfg.add_block(else_b)
    cfg.add_block(join)
    cfg.connect(entry, then_b)
    cfg.connect(entry, else_b)
    cfg.connect(then_b, join)
    cfg.connect(else_b, join)

    ssa = SSABuilder()
    ssa.build(cfg)

    has_phi = any(isinstance(i, PhiInstruction) for i in join.instructions)
    assert has_phi


def test_ssa_renames_variables() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("acc", Literal(0)))

    loop_header = BasicBlock(name="loop_header")
    loop_body = BasicBlock(name="loop_body")
    loop_body.instructions.append(Assign("acc", Var("acc")))

    loop_exit = BasicBlock(name="loop_exit")

    cfg = CFG(entry)
    cfg.add_block(loop_header)
    cfg.add_block(loop_body)
    cfg.add_block(loop_exit)
    cfg.connect(entry, loop_header)
    cfg.connect(loop_header, loop_body)
    cfg.connect(loop_header, loop_exit)
    cfg.connect(loop_body, loop_header)

    ssa = SSABuilder()
    ssa.build(cfg)

    targets = set()
    for block in cfg.blocks:
        for instr in block.instructions:
            if isinstance(instr, Assign):
                targets.add(instr.target)
    assert any("_" in t for t in targets)

from core.cfg import BasicBlock, CFG
from core.cfg_builder import CFGBuilder
from core.ir import Add, Assign, Cmp, If, Literal, Loop, Print, Var


def test_cfg_builder_linear() -> None:
    nodes = [
        Assign("x", Literal(1)),
        Assign("y", Literal(2)),
        Print(Var("x")),
    ]
    builder = CFGBuilder()
    cfg = builder.build(nodes)
    assert len(cfg.blocks) >= 1
    assert cfg.entry is not None


def test_cfg_builder_if_creates_branches() -> None:
    nodes = [
        If(
            condition=Cmp(Var("x"), ">", Literal(0)),
            body=[Assign("y", Literal(1))],
            orelse=[Assign("y", Literal(0))],
        ),
    ]
    builder = CFGBuilder()
    cfg = builder.build(nodes)
    block_names = [b.name for b in cfg.blocks]
    has_then = any("then" in n for n in block_names)
    has_else = any("else" in n for n in block_names)
    has_join = any("join" in n for n in block_names)
    assert has_then
    assert has_else
    assert has_join


def test_cfg_builder_loop_has_back_edge() -> None:
    nodes = [
        Loop(
            count=3,
            body=[
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
    ]
    builder = CFGBuilder()
    cfg = builder.build(nodes)
    block_names = [b.name for b in cfg.blocks]
    has_header = any("loop_header" in n for n in block_names)
    has_body = any("loop_body" in n for n in block_names)
    has_exit = any("loop_exit" in n for n in block_names)
    assert has_header
    assert has_body
    assert has_exit

    header_block = None
    body_block = None
    for b in cfg.blocks:
        if "loop_header" in b.name:
            header_block = b
        if "loop_body" in b.name:
            body_block = b
    assert header_block is not None and body_block is not None
    assert header_block in body_block.succs

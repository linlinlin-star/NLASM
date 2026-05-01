from core.cfg import BasicBlock, CFG
from core.dominator import DominatorTree


def test_linear_cfg_dominators() -> None:
    entry = BasicBlock(name="entry")
    b1 = BasicBlock(name="b1")
    b2 = BasicBlock(name="b2")
    cfg = CFG(entry)
    cfg.add_block(b1)
    cfg.add_block(b2)
    cfg.connect(entry, b1)
    cfg.connect(b1, b2)

    dt = DominatorTree()
    dt.build(cfg)

    assert entry in dt.dom[b1]
    assert entry in dt.dom[b2]
    assert b1 in dt.dom[b2]


def test_branch_cfg_frontier() -> None:
    entry = BasicBlock(name="entry")
    then_b = BasicBlock(name="then")
    else_b = BasicBlock(name="else")
    join = BasicBlock(name="join")
    cfg = CFG(entry)
    cfg.add_block(then_b)
    cfg.add_block(else_b)
    cfg.add_block(join)
    cfg.connect(entry, then_b)
    cfg.connect(entry, else_b)
    cfg.connect(then_b, join)
    cfg.connect(else_b, join)

    dt = DominatorTree()
    dt.build(cfg)

    assert join in dt.frontier.get(then_b, set()) or join in dt.frontier.get(else_b, set())


def test_entry_dominates_itself() -> None:
    entry = BasicBlock(name="entry")
    cfg = CFG(entry)
    dt = DominatorTree()
    dt.build(cfg)
    assert entry in dt.dom[entry]

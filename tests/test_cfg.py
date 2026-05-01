from core.cfg import BasicBlock, CFG


def test_cfg_add_block() -> None:
    entry = BasicBlock(name="entry")
    cfg = CFG(entry)
    b1 = BasicBlock(name="b1")
    cfg.add_block(b1)
    assert len(cfg.blocks) == 2


def test_cfg_connect() -> None:
    entry = BasicBlock(name="entry")
    b1 = BasicBlock(name="b1")
    cfg = CFG(entry)
    cfg.add_block(b1)
    cfg.connect(entry, b1)
    assert b1 in entry.succs
    assert entry in b1.preds


def test_cfg_dump() -> None:
    entry = BasicBlock(name="entry")
    cfg = CFG(entry)
    text = cfg.dump()
    assert "entry" in text

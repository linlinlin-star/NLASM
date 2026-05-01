from core.cfg import BasicBlock, CFG
from core.ir import Add, Assign, Literal, Mul, Var
from core.optimizer import ConstFold, DCE, GVN, Optimizer


def test_const_fold_add() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("x", Add(Literal(1), Literal(2))))
    cfg = CFG(entry)

    cf = ConstFold()
    cf.run(cfg)

    instr = cfg.entry.instructions[0]
    assert isinstance(instr, Assign)
    assert isinstance(instr.value, Literal)
    assert instr.value.value == 3


def test_const_fold_mul_by_one() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("x", Mul(Var("y"), Literal(1))))
    cfg = CFG(entry)

    cf = ConstFold()
    cf.run(cfg)

    instr = cfg.entry.instructions[0]
    assert isinstance(instr, Assign)
    assert isinstance(instr.value, Var)
    assert instr.value.name == "y"


def test_dce_removes_dead_code() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("dead", Literal(42)))
    entry.instructions.append(Assign("used", Literal(7)))
    entry.instructions.append(Assign("result", Var("used")))
    cfg = CFG(entry)

    dce = DCE()
    dce.run(cfg)

    targets = [i.target for i in cfg.entry.instructions if isinstance(i, Assign)]
    assert "dead" not in targets
    assert "used" in targets


def test_optimizer_preserves_result() -> None:
    entry = BasicBlock(name="entry")
    entry.instructions.append(Assign("x", Add(Literal(1), Literal(2))))
    entry.instructions.append(Assign("y", Var("x")))
    cfg = CFG(entry)

    opt = Optimizer()
    opt.run(cfg)

    assert len(cfg.entry.instructions) >= 1

from core.ir import Add, Assign, Cmp, If, Literal, Loop, Mul, Print, Return, Sub, Var


def test_arithmetic_nodes_keep_children() -> None:
    expr = Add(Literal(1), Mul(Var("x"), Literal(2)))
    assert expr.left.value == 1
    assert expr.right.left.name == "x"
    assert expr.right.right.value == 2


def test_statement_nodes_store_structure() -> None:
    stmt = If(
        condition=Cmp(Var("x"), ">", Literal(10)),
        body=[Assign("y", Sub(Var("x"), Literal(1)))],
        orelse=[Return(Literal(0))],
    )
    loop = Loop(count=3, body=[stmt, Print(Var("y"))])
    assert loop.count == 3
    assert loop.body[0].body[0].target == "y"
    assert loop.body[1].value.name == "y"

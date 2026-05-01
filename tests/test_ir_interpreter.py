from core.ir import Add, Assign, Cmp, If, Literal, Loop, Mul, Print, Var
from core.ir_interpreter import IRInterpreter
from core.slot_types import ArraySlot


def test_filter_sum_interpreter() -> None:
    arr = ArraySlot(name="arr", values=[1, 5, 12, 20])
    interp = IRInterpreter(arr_slot=arr)

    nodes = [
        Assign("acc", Literal(0)),
        Assign("i", Literal(0)),
        Loop(
            count=4,
            body=[
                If(
                    condition=Cmp(Var("arr[i]"), ">", Literal(10)),
                    body=[
                        Assign("acc", Add(Var("acc"), Var("arr[i]"))),
                    ],
                ),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var("acc")),
    ]

    interp.run(nodes)
    assert interp.outputs[-1] == 32


def test_sum_array_interpreter() -> None:
    arr = ArraySlot(name="arr", values=[1, 2, 3, 4])
    interp = IRInterpreter(arr_slot=arr)

    nodes = [
        Assign("acc", Literal(0)),
        Assign("i", Literal(0)),
        Loop(
            count=4,
            body=[
                Assign("acc", Add(Var("acc"), Var("arr[i]"))),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var("acc")),
    ]

    interp.run(nodes)
    assert interp.outputs[-1] == 10


def test_map_double_interpreter() -> None:
    arr = ArraySlot(name="arr", values=[1, 2, 3])
    interp = IRInterpreter(arr_slot=arr)

    nodes = [
        Assign("i", Literal(0)),
        Loop(
            count=3,
            body=[
                Assign("arr[i]", Mul(Var("arr[i]"), Literal(2))),
                Assign("i", Add(Var("i"), Literal(1))),
            ],
        ),
        Print(Var("arr")),
    ]

    interp.run(nodes)
    assert interp.outputs[-1] == [2, 4, 6]


def test_eval_arithmetic() -> None:
    interp = IRInterpreter()
    result = interp._eval_expr(Add(Literal(3), Literal(4)))
    assert result == 7


def test_compare_operations() -> None:
    interp = IRInterpreter()
    assert interp._compare(5, ">", 3) is True
    assert interp._compare(3, ">=", 3) is True
    assert interp._compare(2, "<", 5) is True
    assert interp._compare(5, "==", 5) is True
    assert interp._compare(5, "!=", 3) is True

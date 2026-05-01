from core.ir_pattern import IRPattern


def test_ir_pattern_constructs() -> None:
    p = IRPattern(
        name="test_pattern",
        description="测试模式",
        examples=["样例1", "样例2"],
        slots={"arr": "array", "N": "length"},
    )
    assert p.name == "test_pattern"
    assert p.description == "测试模式"
    assert len(p.examples) == 2
    assert p.ir_builder is None
    assert p.vector is None


def test_ir_pattern_with_builder() -> None:
    def builder(slots):
        return []

    p = IRPattern(
        name="with_builder",
        description="有builder的模式",
        examples=["样例"],
        slots={"arr": "array"},
        ir_builder=builder,
    )
    assert p.ir_builder is not None
    assert p.ir_builder({}) == []

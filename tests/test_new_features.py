from core.ir import (
    Add,
    Assign,
    ClassDef,
    Cmp,
    Div,
    For,
    FuncCall,
    FuncDef,
    If,
    Literal,
    Print,
    Raise,
    Return,
    Sub,
    TryExcept,
    Var,
    VarDecl,
)
from core.ir_interpreter import IRInterpreter, NLASMClass, NLASMException, NLASMInstance


# ============================================================
# 面向对象测试 / Object-Oriented Tests
# ============================================================


def test_class_basic():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="动物",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("名字", None)],
                    body=[
                        Assign(target="self.名字", value=Var("名字")),
                    ],
                ),
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Literal("我是"), Var("self.名字"))),
                    ],
                ),
            ],
        ),
        Assign(target="猫", value=FuncCall(name="动物", args=[Literal("小花")])),
        Assign(target="结果", value=FuncCall(name="_attr_说话", args=[Var("猫")])),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "我是小花"


def test_class_inheritance():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="动物",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("名字", None)],
                    body=[
                        Assign(target="self.名字", value=Var("名字")),
                    ],
                ),
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Literal("我是"), Var("self.名字"))),
                    ],
                ),
            ],
        ),
        ClassDef(
            name="猫",
            bases=["动物"],
            body=[
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Var("self.名字"), Literal("喵喵"))),
                    ],
                ),
            ],
        ),
        Assign(target="小猫", value=FuncCall(name="猫", args=[Literal("咪咪")])),
        Assign(target="结果", value=FuncCall(name="_attr_说话", args=[Var("小猫")])),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "咪咪喵喵"


def test_class_inheritance_inherited_method():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="动物",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("名字", None)],
                    body=[
                        Assign(target="self.名字", value=Var("名字")),
                    ],
                ),
                FuncDef(
                    name="获取名字",
                    params=[("self", None)],
                    body=[
                        Return(Var("self.名字")),
                    ],
                ),
            ],
        ),
        ClassDef(
            name="狗",
            bases=["动物"],
            body=[
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Var("self.名字"), Literal("汪汪"))),
                    ],
                ),
            ],
        ),
        Assign(target="旺财", value=FuncCall(name="狗", args=[Literal("旺财")])),
        Assign(target="名字", value=FuncCall(name="_attr_获取名字", args=[Var("旺财")])),
        Print(Var("名字")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "旺财"


def test_class_super_call():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="动物",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("名字", None)],
                    body=[
                        Assign(target="self.名字", value=Var("名字")),
                    ],
                ),
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Literal("动物:"), Var("self.名字"))),
                    ],
                ),
            ],
        ),
        ClassDef(
            name="猫",
            bases=["动物"],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("名字", None), ("颜色", None)],
                    body=[
                        FuncCall(name="super", args=[Var("名字")]),
                        Assign(target="self.颜色", value=Var("颜色")),
                    ],
                ),
                FuncDef(
                    name="说话",
                    params=[("self", None)],
                    body=[
                        Return(Add(Var("self.名字"), Add(Literal("("), Add(Var("self.颜色"), Literal(") 喵喵"))))),
                    ],
                ),
            ],
        ),
        Assign(target="猫猫", value=FuncCall(name="猫", args=[Literal("咪咪"), Literal("白色")])),
        Assign(target="结果", value=FuncCall(name="_attr_说话", args=[Var("猫猫")])),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "咪咪(白色) 喵喵"


def test_class_class_vars():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="计数器",
            bases=[],
            body=[
                Assign(target="计数", value=Literal(0)),
                FuncDef(
                    name="初始化",
                    params=[("self", None)],
                    body=[
                        Assign(target="self.计数", value=Literal(0)),
                    ],
                ),
                FuncDef(
                    name="增加",
                    params=[("self", None)],
                    body=[
                        Assign(target="self.计数", value=Add(Var("self.计数"), Literal(1))),
                        Return(Var("self.计数")),
                    ],
                ),
            ],
        ),
        Assign(target="c", value=FuncCall(name="计数器", args=[])),
        FuncCall(name="_attr_增加", args=[Var("c")]),
        FuncCall(name="_attr_增加", args=[Var("c")]),
        Assign(target="结果", value=FuncCall(name="_attr_增加", args=[Var("c")])),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == 3


# ============================================================
# 异常处理测试 / Exception Handling Tests
# ============================================================


def test_try_except_basic():
    interp = IRInterpreter()
    nodes = [
        TryExcept(
            body=[
                Assign(target="x", value=Div(Literal(10), Literal(0))),
            ],
            handlers=[
                (None, "e", [
                    Assign(target="结果", value=Literal("捕获到异常")),
                ]),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "捕获到异常"


def test_try_except_type_match():
    interp = IRInterpreter()
    nodes = [
        TryExcept(
            body=[
                Raise(value=Literal("测试错误")),
            ],
            handlers=[
                ("RuntimeError", "e", [
                    Assign(target="结果", value=Literal("匹配RuntimeError")),
                ]),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "匹配RuntimeError"


def test_try_except_finally():
    interp = IRInterpreter()
    nodes = [
        Assign(target="结果", value=Literal("")),
        TryExcept(
            body=[
                Assign(target="结果", value=Add(Var("结果"), Literal("try"))),
            ],
            handlers=[
                (None, "e", [
                    Assign(target="结果", value=Add(Var("结果"), Literal("catch"))),
                ]),
            ],
            finally_body=[
                Assign(target="结果", value=Add(Var("结果"), Literal("finally"))),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "tryfinally"


def test_try_except_finally_with_exception():
    interp = IRInterpreter()
    nodes = [
        Assign(target="结果", value=Literal("")),
        TryExcept(
            body=[
                Raise(value=Literal("错误")),
            ],
            handlers=[
                (None, "e", [
                    Assign(target="结果", value=Add(Var("结果"), Literal("catch"))),
                ]),
            ],
            finally_body=[
                Assign(target="结果", value=Add(Var("结果"), Literal("finally"))),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "catchfinally"


def test_raise_custom_exception_class():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="我的错误",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("消息", None)],
                    body=[
                        Assign(target="self.消息", value=Var("消息")),
                    ],
                ),
            ],
        ),
        TryExcept(
            body=[
                Assign(target="err", value=FuncCall(name="我的错误", args=[Literal("自定义错误消息")])),
                Raise(value=Var("err")),
            ],
            handlers=[
                ("我的错误", "e", [
                    Assign(target="结果", value=Literal("捕获自定义异常")),
                ]),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "捕获自定义异常"


def test_exception_inheritance_catch():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="基础错误",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("消息", None)],
                    body=[
                        Assign(target="self.消息", value=Var("消息")),
                    ],
                ),
            ],
        ),
        ClassDef(
            name="网络错误",
            bases=["基础错误"],
            body=[],
        ),
        TryExcept(
            body=[
                Assign(target="err", value=FuncCall(name="网络错误", args=[Literal("连接超时")])),
                Raise(value=Var("err")),
            ],
            handlers=[
                ("基础错误", "e", [
                    Assign(target="结果", value=Literal("捕获基类异常")),
                ]),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "捕获基类异常"


# ============================================================
# 默认参数测试 / Default Parameter Tests
# ============================================================


def test_default_params_basic():
    interp = IRInterpreter()
    nodes = [
        FuncDef(
            name="问候",
            params=[("名字", None), ("问候语", None)],
            defaults={"问候语": Literal("你好")},
            body=[
                Return(Add(Var("问候语"), Add(Literal(" "), Var("名字")))),
            ],
        ),
        Assign(target="r1", value=FuncCall(name="问候", args=[Literal("小明")])),
        Assign(target="r2", value=FuncCall(name="问候", args=[Literal("小明"), Literal("早上好")])),
        Print(Var("r1")),
        Print(Var("r2")),
    ]
    interp.run(nodes)
    assert interp.outputs[-2] == "你好 小明"
    assert interp.outputs[-1] == "早上好 小明"


def test_default_params_multiple():
    interp = IRInterpreter()
    nodes = [
        FuncDef(
            name="计算",
            params=[("a", None), ("b", None), ("c", None)],
            defaults={"b": Literal(10), "c": Literal(1)},
            body=[
                Return(Add(Var("a"), Add(Var("b"), Var("c")))),
            ],
        ),
        Assign(target="r1", value=FuncCall(name="计算", args=[Literal(5)])),
        Assign(target="r2", value=FuncCall(name="计算", args=[Literal(5), Literal(20)])),
        Assign(target="r3", value=FuncCall(name="计算", args=[Literal(5), Literal(20), Literal(100)])),
        Print(Var("r1")),
        Print(Var("r2")),
        Print(Var("r3")),
    ]
    interp.run(nodes)
    assert interp.outputs[-3] == 16
    assert interp.outputs[-2] == 26
    assert interp.outputs[-1] == 125


def test_default_params_in_method():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="计算器",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("初始值", None)],
                    defaults={"初始值": Literal(0)},
                    body=[
                        Assign(target="self.值", value=Var("初始值")),
                    ],
                ),
                FuncDef(
                    name="增加",
                    params=[("self", None), ("数量", None)],
                    defaults={"数量": Literal(1)},
                    body=[
                        Assign(target="self.值", value=Add(Var("self.值"), Var("数量"))),
                        Return(Var("self.值")),
                    ],
                ),
            ],
        ),
        Assign(target="c1", value=FuncCall(name="计算器", args=[])),
        Assign(target="c2", value=FuncCall(name="计算器", args=[Literal(100)])),
        Assign(target="r1", value=FuncCall(name="_attr_增加", args=[Var("c1")])),
        Assign(target="r2", value=FuncCall(name="_attr_增加", args=[Var("c2"), Literal(50)])),
        Print(Var("r1")),
        Print(Var("r2")),
    ]
    interp.run(nodes)
    assert interp.outputs[-2] == 1
    assert interp.outputs[-1] == 150


# ============================================================
# 可变参数测试 / Variadic Parameter Tests
# ============================================================


def test_variadic_params_basic():
    interp = IRInterpreter()
    nodes = [
        FuncDef(
            name="求和",
            params=[],
            variadic="数字",
            body=[
                Assign(target="结果", value=Literal(0)),
                For(
                    var="n",
                    iterable=Var("数字"),
                    body=[
                        Assign(target="结果", value=Add(Var("结果"), Var("n"))),
                    ],
                ),
                Return(Var("结果")),
            ],
        ),
        Assign(target="r1", value=FuncCall(name="求和", args=[Literal(1), Literal(2), Literal(3)])),
        Assign(target="r2", value=FuncCall(name="求和", args=[Literal(10), Literal(20), Literal(30), Literal(40)])),
        Print(Var("r1")),
        Print(Var("r2")),
    ]
    interp.run(nodes)
    assert interp.outputs[-2] == 6
    assert interp.outputs[-1] == 100


def test_variadic_params_with_regular():
    interp = IRInterpreter()
    nodes = [
        FuncDef(
            name="问候",
            params=[("前缀", None)],
            variadic="名字们",
            body=[
                Assign(target="结果", value=Var("前缀")),
                For(
                    var="名字",
                    iterable=Var("名字们"),
                    body=[
                        Assign(target="结果", value=Add(Var("结果"), Add(Literal(" "), Var("名字")))),
                    ],
                ),
                Return(Var("结果")),
            ],
        ),
        Assign(target="r", value=FuncCall(name="问候", args=[Literal("你好"), Literal("小明"), Literal("小红")])),
        Print(Var("r")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "你好 小明 小红"


def test_variadic_params_empty():
    interp = IRInterpreter()
    nodes = [
        FuncDef(
            name="计数",
            params=[("前缀", None)],
            variadic="项目",
            body=[
                Assign(target="结果", value=Add(Var("前缀"), Add(Literal(":"), Literal(0)))),
                Return(Var("结果")),
            ],
        ),
        Assign(target="r", value=FuncCall(name="计数", args=[Literal("总计")])),
        Print(Var("r")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "总计:0"


# ============================================================
# 解析器测试 / Parser Tests
# ============================================================


def test_parse_default_params():
    from core.file_parser import NLFileParser
    parser = NLFileParser()
    source = """定义函数 问候(名字, 问候语="你好"):
    返回 问候语"""
    stmts = parser.parse(source)
    assert len(stmts) == 1
    func = stmts[0]
    assert isinstance(func, FuncDef)
    assert func.name == "问候"
    assert len(func.params) == 2
    assert func.defaults is not None
    assert "问候语" in func.defaults


def test_parse_variadic_params():
    from core.file_parser import NLFileParser
    parser = NLFileParser()
    source = """定义函数 求和(*数字):
    返回 0"""
    stmts = parser.parse(source)
    assert len(stmts) == 1
    func = stmts[0]
    assert isinstance(func, FuncDef)
    assert func.variadic == "数字"


def test_parse_class_inheritance():
    from core.file_parser import NLFileParser
    parser = NLFileParser()
    source = """定义类 猫(动物):
    定义函数 说话(self):
        返回 "喵" """
    stmts = parser.parse(source)
    assert len(stmts) == 1
    cls = stmts[0]
    assert isinstance(cls, ClassDef)
    assert cls.name == "猫"
    assert cls.bases == ["动物"]


def test_parse_try_except():
    from core.file_parser import NLFileParser
    parser = NLFileParser()
    source = """尝试:
    x = 10 / 0
捕获 RuntimeError as e:
    结果 = "错误"
最终:
    输出 "完成" """
    stmts = parser.parse(source)
    assert len(stmts) == 1
    try_node = stmts[0]
    assert isinstance(try_node, TryExcept)
    assert len(try_node.handlers) == 1
    assert try_node.handlers[0][0] == "RuntimeError"
    assert try_node.handlers[0][1] == "e"
    assert try_node.finally_body is not None


def test_parse_raise():
    from core.file_parser import NLFileParser
    parser = NLFileParser()
    source = '抛出 "出错了"'
    stmts = parser.parse(source)
    assert len(stmts) == 1
    assert isinstance(stmts[0], Raise)


# ============================================================
# 综合测试 / Integration Tests
# ============================================================


def test_class_with_default_and_variadic():
    interp = IRInterpreter()
    nodes = [
        ClassDef(
            name="日志器",
            bases=[],
            body=[
                FuncDef(
                    name="初始化",
                    params=[("self", None), ("级别", None)],
                    defaults={"级别": Literal("INFO")},
                    body=[
                        Assign(target="self.级别", value=Var("级别")),
                        Assign(target="self.消息们", value=Literal([])),
                    ],
                ),
                FuncDef(
                    name="记录",
                    params=[("self", None)],
                    variadic="消息",
                    body=[
                        Assign(target="self.消息们", value=FuncCall(name="_attr_扩展", args=[Var("self.消息们"), Var("消息")])),
                        Return(Var("self.级别")),
                    ],
                ),
            ],
        ),
        Assign(target="log", value=FuncCall(name="日志器", args=[])),
        Assign(target="r1", value=FuncCall(name="_attr_记录", args=[Var("log"), Literal("hello"), Literal("world")])),
        Print(Var("r1")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "INFO"


def test_nested_try_except():
    interp = IRInterpreter()
    nodes = [
        TryExcept(
            body=[
                TryExcept(
                    body=[
                        Raise(value=Literal("内部错误")),
                    ],
                    handlers=[
                        (None, "e", [
                            Raise(value=Literal("外部错误")),
                        ]),
                    ],
                ),
            ],
            handlers=[
                (None, "e2", [
                    Assign(target="结果", value=Literal("最外层捕获")),
                ]),
            ],
        ),
        Print(Var("结果")),
    ]
    interp.run(nodes)
    assert interp.outputs[-1] == "最外层捕获"

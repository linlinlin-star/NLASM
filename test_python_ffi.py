import sys
sys.path.insert(0, ".")

from core.ir_interpreter import IRInterpreter
from core.file_parser import NLFileParser
from core.python_bridge import PythonBridge

print("=== 测试 Python FFI ===\n")

bridge = PythonBridge()
interp = IRInterpreter(bridge=bridge)
p = NLFileParser()

# 测试1: stdlib json (功能有限)
code1 = """
导入 json
定义 s = json.序列化({"name": "test"})
"""
interp.run(p.parse(code1))
result = interp.symtab.lookup('s').value
print(f"测试1 - json.序列化: {result}")

# 测试2: platform (无 .nl 版本 - 回退到 Python)
code2 = """
导入 platform as plat
定义 p = plat.处理器()
"""
interp.run(p.parse(code2))
result = interp.symtab.lookup('p').value
print(f"测试2 - platform.处理器(): {result}")

# 测试3: collections (无 .nl 版本)
code3 = """
导入 collections as coll
定义 d = coll.有序字典()
"""
interp.run(p.parse(code3))
result = interp.symtab.lookup('d').value
print(f"测试3 - collections.有序字典(): {type(result).__name__}")

# 测试4: re (无 .nl 版本)
code4 = """
导入 re
定义 pattern = re.编译("(\\d+)-(\\d+)")
定义 m = pattern.匹配("123-456")
定义 g1 = m.分组(1) if m else ""
"""
interp.run(p.parse(code4))
result = interp.symtab.lookup('g1').value
print(f"测试4 - re.分组(1): {result}")

# 测试5: hashlib
code5 = """
导入 hashlib as hl
定义 data = "test"
定义 h = hl.sha256()
h.更新(data.编码())
定义 result = h.十六进制摘要()
"""
interp.run(p.parse(code5))
result = interp.symtab.lookup('result').value
print(f"测试5 - hashlib.十六进制摘要(): {result[:16]}...")

# 测试6: itertools (无 .nl 版本)
code6 = """
导入 itertools as it
定义 cycle = it.循环([1, 2, 3])
定义 first3 = []
定义 i = 0
当 i < 3:
    定义 first3 = first3 + [next(cycle)]
    i = i + 1
"""
interp.run(p.parse(code6))
result = interp.symtab.lookup('first3').value
print(f"测试6 - itertools.循环: {result}")

# 测试7: functools
code7 = """
导入 functools as ft
定义 f = ft.偏函数(打印, "prefix:")
打印(f("world"))
"""
try:
    interp.run(p.parse(code7))
    print("测试7 - functools.偏函数: OK")
except Exception as e:
    print(f"测试7 - functools.偏函数: {e}")

# 测试8: urllib (Python 3)
code8 = """
导入 urllib.request as ur
"""
interp.run(p.parse(code8))
print("测试8 - urllib.request: OK")

# 测试9: json (用 Python 的)
code9 = """
导入 json as j
定义 obj = j.loads('{"x": 1}')
定义 x = obj["x"]
"""
interp.run(p.parse(code9))
result = interp.symtab.lookup('x').value
print(f"测试9 - json.loads: {result}")

# 测试10: datetime.datetime (用 Python)
code10 = """
导入 datetime as dt
定义 n = dt.日期时间.now()
定义 year = n.year
定义 month = n.month
"""
interp.run(p.parse(code10))
result = interp.symtab.lookup('year').value
print(f"测试10 - datetime.now().year: {result}")

print("\n=== Python FFI 测试完成 ===")
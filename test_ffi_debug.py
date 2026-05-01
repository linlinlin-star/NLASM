import sys
sys.path.insert(0, ".")

from core.ir_interpreter import IRInterpreter
from core.file_parser import NLFileParser
from core.python_bridge import PythonBridge
from core.module_system import ModuleLoader

print("=== 调试 Python FFI ===\n")

loader = ModuleLoader(search_paths=[".", "./stdlib"])

try:
    mod = loader.import_module("platform")
    print(f"platform exports: {list(mod.exports.keys())[:10]}")
    print(f"platform.processor: {mod.exports.get('processor', 'NOT FOUND')}")
except Exception as e:
    print(f"platform failed: {e}")

try:
    mod = loader.import_module("collections")
    print(f"collections exports: {list(mod.exports.keys())[:10]}")
except Exception as e:
    print(f"collections failed: {e}")

print("\n=== Done ===")
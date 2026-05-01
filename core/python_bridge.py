from __future__ import annotations

import importlib
from typing import Any


class PythonBridge:
    """Python互操作桥 - 允许NLASM调用Python模块和函数 / Python interop bridge - allows NLASM to call Python modules and functions.

    提供模块导入、函数调用、对象创建和属性访问的桥接能力。
    Provides bridging capabilities for module import, function call, object creation, and attribute access.
    """

    def __init__(self) -> None:
        self.imported_modules: dict[str, Any] = {}  # 已导入的Python模块 / Imported Python modules

    def import_module(self, module_name: str, alias: str | None = None) -> Any:
        """导入Python模块 / Import a Python module"""
        try:
            mod = importlib.import_module(module_name)
            name = alias or module_name
            self.imported_modules[name] = mod
            return mod
        except ImportError as e:
            raise RuntimeError(f"无法导入 Python 模块 {module_name}: {e}") from e

    def call_function(
        self,
        module_name: str,
        func_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """调用Python模块中的函数 / Call a function from a Python module"""
        mod = self._resolve_module(module_name)
        parts = func_name.split(".")
        obj: Any = mod
        for part in parts:
            obj = getattr(obj, part)
        if not callable(obj):
            raise TypeError(f"{module_name}.{func_name} 不是可调用对象")
        return obj(*args, **kwargs)

    def create_object(
        self,
        module_name: str,
        class_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """创建Python对象实例 / Create a Python object instance"""
        mod = self._resolve_module(module_name)
        cls = getattr(mod, class_name)
        return cls(*args, **kwargs)

    def get_attribute(self, module_name: str, attr_name: str) -> Any:
        """获取Python模块的属性 / Get attribute from a Python module"""
        mod = self._resolve_module(module_name)
        parts = attr_name.split(".")
        obj: Any = mod
        for part in parts:
            obj = getattr(obj, part)
        return obj

    def _resolve_module(self, module_name: str) -> Any:
        """解析已导入的模块 / Resolve an imported module"""
        if module_name not in self.imported_modules:
            raise RuntimeError(f"模块 {module_name} 未导入，请先使用 import 语句导入")
        return self.imported_modules[module_name]

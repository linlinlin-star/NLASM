from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .file_parser import NLFileParser
from .ir import FuncDef, ImportStmt, Stmt
from .package_manager import MANIFEST_FILE, PACKAGES_DIR


class Module:
    """NLASM模块 - 表示一个已加载的.nl文件模块 / NLASM module - represents a loaded .nl file module.

    模块通过解析.nl文件获得IR节点，将FuncDef导出为模块公共API。
    Modules obtain IR nodes by parsing .nl files, exporting FuncDefs as public API.
    """

    def __init__(self, name: str, filepath: str | None = None) -> None:
        self.name = name
        self.filepath = filepath
        self.exports: dict[str, Any] = {}  # 导出字典（函数名 -> FuncDef）/ Export dictionary (function name -> FuncDef)
        self.ir_nodes: list[Stmt] = []     # 解析后的IR节点列表 / Parsed IR node list
        self.loaded: bool = False
        self._py_module: Any = None  # 原始 Python 模块引用 / Original Python module reference

    def load(self) -> None:
        """加载模块 - 解析.nl文件并提取导出 / Load module - parse .nl file and extract exports"""
        if self.loaded:
            return
        if self.filepath and Path(self.filepath).exists():
            parser = NLFileParser()
            self.ir_nodes = parser.parse_file(self.filepath)
            # 将所有函数定义导出 / Export all function definitions
            for node in self.ir_nodes:
                if isinstance(node, FuncDef):
                    self.exports[node.name] = node
        self.loaded = True

    def get_export(self, name: str) -> Any:
        """获取指定名称的导出 / Get export by name"""
        if name in self.exports:
            return self.exports[name]
        raise ImportError(f"模块 {self.name} 没有导出: {name}")


class ModuleLoader:
    """模块加载器 - 搜索并加载.nl模块 / Module loader - searches and loads .nl modules.

    搜索路径优先级: 当前目录 -> ./lib -> ./stdlib -> .nlasm包目录
    如果.nl文件未找到，回退到Python importlib导入。
    Search path priority: current dir -> ./lib -> ./stdlib -> .nlasm packages
    Falls back to Python importlib if .nl file not found.
    """

    def __init__(self, search_paths: list[str] | None = None, project_dir: str | None = None) -> None:
        self.loaded_modules: dict[str, Module] = {}
        self.search_paths = search_paths or [".", "./lib", "./stdlib"]
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self._add_package_search_paths()

    def _add_package_search_paths(self) -> None:
        packages_dir = self.project_dir / PACKAGES_DIR
        if packages_dir.exists():
            for pkg_dir in sorted(packages_dir.iterdir()):
                if pkg_dir.is_dir():
                    pkg_path = str(pkg_dir)
                    if pkg_path not in self.search_paths:
                        self.search_paths.append(pkg_path)
                    manifest_path = pkg_dir / MANIFEST_FILE
                    if manifest_path.exists():
                        try:
                            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                            main_file = manifest_data.get("main", "")
                            if main_file:
                                main_dir = str((pkg_dir / main_file).parent)
                                if main_dir not in self.search_paths:
                                    self.search_paths.append(main_dir)
                        except (json.JSONDecodeError, OSError):
                            pass

    def import_module(self, name: str) -> Module:
        if name in self.loaded_modules:
            return self.loaded_modules[name]

        for path in self.search_paths:
            filepath = Path(path) / f"{name}.nl"
            if filepath.exists():
                module = Module(name=name, filepath=str(filepath))
                module.load()
                self.loaded_modules[name] = module
                return module

        pkg_module = self._try_load_from_package(name)
        if pkg_module:
            return pkg_module

        try:
            import importlib
            mod = importlib.import_module(name)
            module = Module(name=name)
            module._py_module = mod  # 保存原始 Python 模块引用
            module.exports = {k: v for k, v in vars(mod).items() if not k.startswith("_")}
            module.loaded = True
            self.loaded_modules[name] = module
            return module
        except ImportError:
            pass

        raise ImportError(f"找不到模块: {name}")

    def _try_load_from_package(self, name: str) -> Module | None:
        packages_dir = self.project_dir / PACKAGES_DIR
        if not packages_dir.exists():
            return None

        pkg_dir = packages_dir / name
        if not pkg_dir.is_dir():
            return None

        manifest_path = pkg_dir / MANIFEST_FILE
        main_file = f"{name}.nl"
        if manifest_path.exists():
            try:
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                main_file = manifest_data.get("main", main_file)
            except (json.JSONDecodeError, OSError):
                pass

        main_path = pkg_dir / main_file
        if main_path.exists():
            module = Module(name=name, filepath=str(main_path))
            module.load()
            self.loaded_modules[name] = module
            return module

        nl_files = list(pkg_dir.glob("*.nl"))
        if nl_files:
            module = Module(name=name, filepath=str(nl_files[0]))
            module.load()
            for nl_file in nl_files[1:]:
                parser = NLFileParser()
                extra_nodes = parser.parse_file(str(nl_file))
                module.ir_nodes.extend(extra_nodes)
                for node in extra_nodes:
                    if isinstance(node, FuncDef):
                        module.exports[node.name] = node
            self.loaded_modules[name] = module
            return module

        return None

    def add_search_path(self, path: str) -> None:
        if path not in self.search_paths:
            self.search_paths.append(path)

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from llvmlite import binding as llvm

from .aot_codegen import AOTCodeGen
from .file_parser import NLFileParser


for _init_fn in [
    getattr(llvm, "initialize", None),
    getattr(llvm, "initialize_native_target", None),
    getattr(llvm, "initialize_native_asmprinter", None),
]:
    if _init_fn is not None:
        try:
            _init_fn()
        except RuntimeError:
            pass


def _find_linker() -> tuple[str | None, str]:
    """查找可用的C编译器/链接器 / Find available C compiler/linker.

    返回: (编译器路径, 编译器类型)
    类型: "gcc-like" | "msvc" | "lld"
    """
    for name in ["cc", "gcc", "clang"]:
        path = shutil.which(name)
        if path is not None:
            return path, "gcc-like"

    msys2_mingw_bin = os.path.join(os.environ.get("MSYS2_ROOT", r"C:\msys64"), "mingw64", "bin")
    gcc_in_msys2 = os.path.join(msys2_mingw_bin, "gcc.exe")
    if os.path.isfile(gcc_in_msys2):
        return gcc_in_msys2, "gcc-like"

    for name in ["cl.exe", "cl"]:
        path = shutil.which(name)
        if path is not None:
            return path, "msvc"

    for name in ["lld-link", "lld-link.exe"]:
        path = shutil.which(name)
        if path is not None:
            return path, "lld"

    llvm_dir = os.path.dirname(llvm.__file__)
    for name in ["lld-link.exe", "lld-link"]:
        candidate = os.path.join(llvm_dir, name)
        if os.path.isfile(candidate):
            return candidate, "lld"

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        for name in ["lld-link.exe", "lld-link"]:
            candidate = os.path.join(conda_prefix, "bin", name)
            if os.path.isfile(candidate):
                return candidate, "lld"

    return None, ""


def _get_obj_ext() -> str:
    return ".obj" if platform.system() == "Windows" else ".o"


def _get_exe_ext() -> str:
    return ".exe" if platform.system() == "Windows" else ""


class AOTCompiler:
    """AOT编译器 — 将.nl源文件编译为原生可执行文件.

    编译流程:
    1. 解析.nl文件 → IR节点
    2. IR节点 → LLVM IR模块 (AOTCodeGen)
    3. LLVM IR → 目标文件 (llvmlite TargetMachine)
    4. 目标文件 → 可执行文件 (系统链接器: gcc/clang/MSVC/lld)
    """

    def __init__(self, opt_level: int = 2) -> None:
        self.opt_level = opt_level
        self.codegen = AOTCodeGen()
        self._target_machine = None

    def _get_target_machine(self):
        if self._target_machine is not None:
            return self._target_machine
        target = llvm.Target.from_default_triple()
        cpu_name = "generic"
        try:
            cpu_name = llvm.get_host_cpu_name()
        except Exception:
            pass
        try:
            self._target_machine = target.create_target_machine(
                cpu=cpu_name,
                features="",
                opt=self.opt_level,
                reloc="pic",
                codemodel="default",
            )
        except Exception:
            self._target_machine = target.create_target_machine(
                cpu="generic",
                features="",
                opt=2,
                reloc="pic",
                codemodel="default",
            )
        return self._target_machine

    def compile_to_object(self, filepath: str, output_path: str | None = None) -> str:
        """编译.nl文件为目标文件(.o/.obj) / Compile .nl file to object file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        parser = NLFileParser()
        stmts = parser.parse_file(filepath)

        llvm_module = self.codegen.generate(stmts, module_name=path.stem)

        unsupported = self.codegen.get_unsupported()
        if unsupported:
            print(f"[警告] 不支持的节点类型: {', '.join(unsupported)}")

        llvm_ir = str(llvm_module)
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        target_machine = self._get_target_machine()

        if output_path is None:
            output_path = str(path.with_suffix(_get_obj_ext()))

        obj_data = target_machine.emit_object(mod)
        Path(output_path).write_bytes(obj_data)
        return output_path

    def compile_to_assembly(self, filepath: str, output_path: str | None = None) -> str:
        """编译.nl文件为汇编文件(.s) / Compile .nl file to assembly"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        parser = NLFileParser()
        stmts = parser.parse_file(filepath)

        llvm_module = self.codegen.generate(stmts, module_name=path.stem)
        llvm_ir = str(llvm_module)
        mod = llvm.parse_assembly(llvm_ir)
        mod.verify()

        target_machine = self._get_target_machine()

        if output_path is None:
            output_path = str(path.with_suffix(".s"))

        asm_text = target_machine.emit_assembly(mod)
        Path(output_path).write_text(asm_text, encoding="utf-8")
        return output_path

    def compile_to_llvm_ir(self, filepath: str, output_path: str | None = None) -> str:
        """编译.nl文件为LLVM IR文本(.ll) / Compile .nl file to LLVM IR text"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        parser = NLFileParser()
        stmts = parser.parse_file(filepath)

        llvm_module = self.codegen.generate(stmts, module_name=path.stem)

        if output_path is None:
            output_path = str(path.with_suffix(".ll"))

        Path(output_path).write_text(str(llvm_module), encoding="utf-8")
        return output_path

    def compile_to_executable(self, filepath: str, output_path: str | None = None) -> str:
        """编译.nl文件为原生可执行文件 / Compile .nl file to native executable"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")

        if output_path is None:
            output_path = str(path.with_suffix(_get_exe_ext()))

        obj_path = self.compile_to_object(filepath)

        linker, linker_type = _find_linker()
        if linker is None:
            obj_file = Path(obj_path)
            print(f"[AOT] 目标文件已生成: {obj_path}")
            print(f"[AOT] 未找到链接器，请手动链接:")
            if platform.system() == "Windows":
                print(f"  方式1 (MSVC): link {obj_path} /OUT:{output_path}")
                print(f"  方式2 (lld):  lld-link {obj_path} /OUT:{output_path}")
                print(f"  方式3 (gcc):  gcc {obj_path} -o {output_path} -municode")
            else:
                print(f"  gcc {obj_path} -o {output_path}")
                print(f"  clang {obj_path} -o {output_path}")
            print(f"\n提示: 安装 MinGW-w64 或 Visual Studio Build Tools 可启用自动链接。")
            return obj_path

        cmd = self._build_link_command(linker, linker_type, obj_path, output_path)
        link_env = os.environ.copy()
        linker_dir = os.path.dirname(os.path.abspath(linker))
        if linker_dir not in link_env.get("PATH", ""):
            link_env["PATH"] = linker_dir + os.pathsep + link_env.get("PATH", "")
        result = subprocess.run(cmd, capture_output=True, text=True, env=link_env)

        if result.returncode != 0:
            raise RuntimeError(
                f"链接失败 (exit code {result.returncode}):\n"
                f"命令: {' '.join(cmd)}\n"
                f"stderr: {result.stderr}"
            )

        exe_path = Path(output_path)
        if not exe_path.exists():
            raise RuntimeError(f"链接后未找到可执行文件: {output_path}")

        try:
            Path(obj_path).unlink(missing_ok=True)
        except Exception:
            pass

        print(f"[AOT] 编译完成: {output_path}")
        return output_path

    def _build_link_command(self, linker: str, linker_type: str, obj_path: str, output_path: str) -> list[str]:
        """构建链接命令 / Build linker command"""
        if linker_type == "msvc":
            return [linker, obj_path, f"/Fe{output_path}", "/link"]

        if linker_type == "lld":
            return [linker, obj_path, f"/OUT:{output_path}", "/DEFAULTLIB:libcmt", "/DEFAULTLIB:oldnames"]

        cmd = [linker, obj_path, "-o", output_path]
        system = platform.system()
        if system == "Darwin":
            pass
        elif system == "Windows":
            cmd.append("-lmingw32")
        else:
            cmd.append("-lm")
            cmd.append("-no-pie")

        return cmd

    def compile_and_run(self, filepath: str, args: list[str] | None = None) -> int:
        """编译并执行 / Compile and run"""
        path = Path(filepath)
        exe_path = str(path.with_suffix(_get_exe_ext()))

        self.compile_to_executable(filepath, output_path=exe_path)

        if not Path(exe_path).exists():
            raise RuntimeError(f"可执行文件未生成: {exe_path}")

        run_cmd = [exe_path]
        if args:
            run_cmd.extend(args)

        result = subprocess.run(run_cmd)
        return result.returncode

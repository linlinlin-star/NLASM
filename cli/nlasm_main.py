from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_file(filepath: str) -> None:
    """运行.nl文件 - 解析并解释执行 / Run .nl file - parse and interpret"""
    from core.file_parser import NLFileParser
    from core.ir_interpreter import IRInterpreter

    parser = NLFileParser()
    stmts = parser.parse_file(filepath)
    file_path = Path(filepath).resolve()
    project_dir = file_path.parent
    while project_dir != project_dir.parent:
        if (project_dir / "nlasm.json").exists():
            break
        project_dir = project_dir.parent
    interp = IRInterpreter(project_dir=str(project_dir))
    interp.run(stmts)
    for output in interp.outputs:
        if isinstance(output, tuple):
            print(" ".join(str(v) for v in output))
        else:
            print(output)


def compile_file(filepath: str, output_path: str | None = None) -> None:
    """编译.nl文件 - 转译为Python或编译为LLVM IR / Compile .nl file - transpile to Python or compile to LLVM IR"""
    from core.file_parser import NLFileParser
    from core.cfg_builder import CFGBuilder
    from core.dominator import DominatorTree
    from core.ssa_builder import SSABuilder
    from core.optimizer import Optimizer
    from core.llvm_codegen import LLVMCodeGen

    path = Path(filepath)
    if not path.exists():
        print(f"错误: 文件不存在: {filepath}")
        sys.exit(1)

    content = path.read_text(encoding="utf-8")

    # 检测是否为.nl代码文件（含定义关键字）/ Detect if .nl code file (contains 定义 keyword)
    if content.strip().startswith("定义") or "\n定义" in content:
        print("[信息] 检测到 .nl 代码文件，转译为 Python...")
        from core.python_transpiler import PythonTranspiler

        parser = NLFileParser()
        stmts = parser.parse_file(filepath)
        transpiler = PythonTranspiler()
        python_code = transpiler.transpile(stmts)

        if output_path is None:
            output_path = filepath.replace('.nl', '.py')

        Path(output_path).write_text(python_code, encoding="utf-8")
        print(f"转译完成: {output_path}")
        return

    # 自然语言输入，走完整编译链 / Natural language input, full compilation chain
    print("[信息] 检测到自然语言输入，编译为 LLVM IR...")

    try:
        from core.frontend import Frontend, RuleEntityExtractor
        from core.pattern_matcher import PatternMatcher
        from core.pattern_database import PATTERN_DB
        from core.slot_filler import SlotFiller
        from core.pattern_instantiator import PatternInstantiator
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"错误: 缺少依赖: {e}")
        print("请安装: pip install sentence-transformers")
        sys.exit(1)

    # 加载语义向量模型 / Load semantic vector model
    model_path = ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
    if model_path.exists():
        embedder = SentenceTransformer(str(model_path))
    else:
        embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    class _Embedder:
        def __init__(self, st_model):
            self._model = st_model

        def encode(self, text, normalize_embeddings=True):
            return self._model.encode(text, normalize_embeddings=normalize_embeddings)

    emb = _Embedder(embedder)

    # 构建编译流水线组件 / Build compilation pipeline components
    frontend = Frontend(embedder=emb, entity_extractor=RuleEntityExtractor())
    matcher = PatternMatcher(embedder=emb, patterns=PATTERN_DB)
    filler = SlotFiller(embedder=emb)
    instantiator = PatternInstantiator()

    # 前端处理 -> Pattern匹配 -> 槽位填充 -> IR实例化 / Frontend -> Pattern match -> Slot fill -> IR instantiate
    packet = frontend.process(content.strip())
    matches = matcher.match(packet.normalized, top_k=3)

    if not matches or matches[0][1] < 0.70:
        print(f"错误: 无法识别意图（最高分数: {matches[0][1] if matches else 0:.2f}）")
        sys.exit(1)

    pattern, score = matches[0]
    print(f"[信息] 匹配到 Pattern: {pattern.name} (分数: {score:.2f})")

    slots = filler.fill(pattern, packet)
    print(f"[信息] 槽位: {slots}")

    ir_nodes = instantiator.instantiate(pattern, slots)
    print(f"[信息] 生成 {len(ir_nodes)} 个 IR 节点")

    # 编译链: IR -> CFG -> SSA -> 优化 -> LLVM IR / Chain: IR -> CFG -> SSA -> Optimize -> LLVM IR
    print("[信息] 构建 CFG...")
    cfg_builder = CFGBuilder()
    cfg = cfg_builder.build(ir_nodes)

    print("[信息] 构建 SSA...")
    dom_tree = DominatorTree()
    dom_tree.build(cfg)
    ssa_builder = SSABuilder()
    ssa_cfg = ssa_builder.build(cfg)

    print("[信息] 优化...")
    optimizer = Optimizer()
    if hasattr(pattern, 'constraints') and pattern.constraints:
        optimizer.set_pattern_hints(pattern.constraints)
    optimized_cfg = optimizer.run(ssa_cfg)

    print("[信息] 生成 LLVM IR...")
    codegen = LLVMCodeGen()
    llvm_module = codegen.lower(optimized_cfg)

    if output_path is None:
        output_path = filepath.replace('.nl', '.ll')

    llvm_ir_text = str(llvm_module)
    Path(output_path).write_text(llvm_ir_text, encoding="utf-8")

    print(f"\n编译完成: {output_path}")
    print(f"\n执行方法:")
    print(f"  1. 使用 lli 解释执行: lli {output_path}")
    base = output_path.replace('.ll', '')
    print(f"  2. 编译为可执行文件:")
    print(f"     llc {output_path} -o {base}.s")
    print(f"     gcc {base}.s -o {base}")


def check_file(filepath: str) -> None:
    """语法和类型检查 / Syntax and type checking"""
    from core.file_parser import NLFileParser, ParseError
    from core.type_system import TypeChecker, infer_type
    from core.ir import (
        Add,
        Assign,
        VarDecl,
        FuncDef,
        If,
        While,
        For,
        ForRange,
        Loop,
        Return,
        Print,
        ImportStmt,
        ClassDef,
        TryExcept,
        Var,
        Literal,
        Mul,
        Sub,
        Div,
        Cmp,
        And,
        Or,
        Not,
        FuncCall,
        ListExpr,
        DictExpr,
        Neg,
        Mod,
    )

    try:
        parser = NLFileParser()
        stmts = parser.parse_file(filepath)

        checker = TypeChecker()

        def infer_expr_type(expr, env_types):
            """推断表达式类型（迭代式）/ Infer expression type (iterative)"""
            results: list = []
            work_stack: list[tuple] = [(expr, False)]

            while work_stack:
                item = work_stack.pop()
                current = item[0]
                processed = item[1]

                if processed:
                    marker = item[2] if len(item) > 2 else None
                    if marker == "add_sub":
                        rt = results.pop()
                        lt = results.pop()
                        if lt == infer_type(0.0) or rt == infer_type(0.0):
                            results.append(infer_type(0.0))
                        elif lt == infer_type("") or rt == infer_type(""):
                            results.append(infer_type(""))
                        else:
                            results.append(infer_type(0))
                    elif marker == "mul":
                        rt = results.pop()
                        lt = results.pop()
                        if lt == infer_type(0.0) or rt == infer_type(0.0):
                            results.append(infer_type(0.0))
                        else:
                            results.append(infer_type(0))
                    elif marker == "neg":
                        results.append(results.pop())
                    elif marker == "list_expr":
                        if results:
                            results.pop()
                        results.append(infer_type([0]))
                    continue

                if isinstance(current, Literal):
                    results.append(infer_type(current.value))
                elif isinstance(current, Var):
                    results.append(env_types.get(current.name, infer_type(0)))
                elif isinstance(current, (Add, Sub)):
                    work_stack.append((current, True, "add_sub"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Mul):
                    work_stack.append((current, True, "mul"))
                    work_stack.append((current.right, False))
                    work_stack.append((current.left, False))
                elif isinstance(current, Div):
                    results.append(infer_type(0.0))
                elif isinstance(current, Mod):
                    results.append(infer_type(0))
                elif isinstance(current, Neg):
                    work_stack.append((current, True, "neg"))
                    work_stack.append((current.operand, False))
                elif isinstance(current, (Cmp, And, Or, Not)):
                    results.append(infer_type(True))
                elif isinstance(current, FuncCall):
                    results.append(infer_type(None))
                elif isinstance(current, ListExpr):
                    if current.elements:
                        work_stack.append((current, True, "list_expr"))
                        work_stack.append((current.elements[0], False))
                    else:
                        results.append(infer_type([]))
                elif isinstance(current, DictExpr):
                    results.append(infer_type({}))
                else:
                    results.append(infer_type(0))

            return results[0] if results else infer_type(0)

        def check_stmt(stmt, env_types=None):
            """检查语句类型（迭代式）/ Check statement types (iterative)"""
            if env_types is None:
                env_types = {}
            work_stack: list[tuple] = [(stmt, env_types)]

            while work_stack:
                current, current_env = work_stack.pop()

                if isinstance(current, VarDecl):
                    val_type = infer_expr_type(current.value, current_env)
                    if current.type_hint:
                        from core.type_system import parse_type
                        expected = parse_type(current.type_hint)
                        checker.check(expected, val_type, context=f"变量 {current.name}")
                    current_env[current.name] = val_type
                elif isinstance(current, Assign):
                    val_type = infer_expr_type(current.value, current_env)
                    if current.target in current_env:
                        checker.check(current_env[current.target], val_type, context=f"赋值 {current.target}")
                    current_env[current.target] = val_type
                elif isinstance(current, FuncDef):
                    func_env = dict(current_env)
                    for pname, ptype in current.params:
                        if ptype:
                            from core.type_system import parse_type
                            func_env[pname] = parse_type(ptype)
                        else:
                            func_env[pname] = infer_type(None)
                    for s in reversed(current.body):
                        work_stack.append((s, func_env))
                elif isinstance(current, ClassDef):
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))
                elif isinstance(current, If):
                    for s in reversed(current.orelse):
                        work_stack.append((s, current_env))
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))
                elif isinstance(current, While):
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))
                elif isinstance(current, (For, ForRange)):
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))
                elif isinstance(current, Loop):
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))
                elif isinstance(current, TryExcept):
                    if current.finally_body:
                        for s in reversed(current.finally_body):
                            work_stack.append((s, current_env))
                    for _, _, handler_body in reversed(current.handlers):
                        for s in reversed(handler_body):
                            work_stack.append((s, current_env))
                    for s in reversed(current.body):
                        work_stack.append((s, current_env))

        for stmt in stmts:
            check_stmt(stmt)

        if checker.errors:
            for error in checker.errors:
                print(f"类型错误: {error}")
            sys.exit(1)

        print(f"语法和类型检查通过，共 {len(stmts)} 个语句")
    except ParseError as e:
        print(f"语法错误: {e}")
        sys.exit(1)


def format_file(filepath: str) -> None:
    """格式化.nl代码 / Format .nl code"""
    path = Path(filepath)
    source = path.read_text(encoding="utf-8")
    lines = source.split("\n")
    formatted: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            formatted.append("    " * (len(line) - len(line.lstrip())) // 4 + stripped)
        else:
            formatted.append(stripped)
    print("\n".join(formatted))


def repl() -> None:
    """简易交互式环境 / Simple interactive REPL"""
    from core.ir_interpreter import IRInterpreter
    from core.file_parser import NLFileParser

    print("NLASM v0.8.0 交互式环境")
    print("输入 NLASM 代码，输入 quit 退出")
    print()

    interp = IRInterpreter()
    parser = NLFileParser()

    while True:
        try:
            text = input("nlasm> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text:
            continue
        if text.lower() in ("quit", "exit", "q"):
            break

        try:
            stmts = parser.parse(text)
            interp.run(stmts)
            for output in interp.outputs:
                if isinstance(output, tuple):
                    print(" ".join(str(v) for v in output))
                else:
                    print(output)
            interp.outputs.clear()
        except Exception as exc:
            print(f"[错误] {type(exc).__name__}: {exc}")


def build_file(filepath: str, output_path: str | None = None, fmt: str = "exe") -> None:
    """AOT编译.nl文件为原生可执行文件 / AOT compile .nl file to native executable"""
    from core.aot_compiler import AOTCompiler

    path = Path(filepath)
    if not path.exists():
        print(f"错误: 文件不存在: {filepath}")
        sys.exit(1)

    try:
        compiler = AOTCompiler(opt_level=2)

        if fmt == "exe":
            result = compiler.compile_to_executable(filepath, output_path)
            print(f"AOT编译完成: {result}")
        elif fmt == "obj":
            result = compiler.compile_to_object(filepath, output_path)
            print(f"目标文件生成完成: {result}")
        elif fmt == "asm":
            result = compiler.compile_to_assembly(filepath, output_path)
            print(f"汇编文件生成完成: {result}")
        elif fmt == "ll":
            result = compiler.compile_to_llvm_ir(filepath, output_path)
            print(f"LLVM IR生成完成: {result}")
        else:
            print(f"错误: 不支持的输出格式: {fmt}")
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"编译错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"编译失败: {type(e).__name__}: {e}")
        sys.exit(1)


def _get_pkg_manager(args) -> "PackageManager":
    from core.package_manager import PackageManager
    project_dir = getattr(args, "project_dir", None)
    registry = getattr(args, "registry", None)
    kwargs: dict = {}
    if project_dir:
        kwargs["project_dir"] = Path(project_dir)
    if registry:
        kwargs["registry_url"] = registry
    return PackageManager(**kwargs)


def pkg_init(args) -> None:
    pm = _get_pkg_manager(args)
    pm.init_project(
        name=args.name,
        author=getattr(args, "author", "") or "",
        description=getattr(args, "description", "") or "",
    )


def pkg_install(args) -> None:
    pm = _get_pkg_manager(args)
    pm.install(
        package_names=args.packages or None,
        dev=getattr(args, "dev", False),
        local=getattr(args, "local", False),
    )


def pkg_uninstall(args) -> None:
    pm = _get_pkg_manager(args)
    pm.uninstall(args.packages)


def pkg_publish(args) -> None:
    pm = _get_pkg_manager(args)
    pm.publish(local=getattr(args, "local", False))


def pkg_list(args) -> None:
    pm = _get_pkg_manager(args)
    packages = pm.list_packages()
    if not packages:
        print("没有已安装的包")
        return
    print(f"{'包名':<20} {'版本':<12} {'描述'}")
    print("-" * 60)
    for pkg in packages:
        print(f"{pkg['name']:<20} {pkg['version']:<12} {pkg['description']}")


def pkg_outdated(args) -> None:
    pm = _get_pkg_manager(args)
    outdated = pm.outdated()
    if not outdated:
        print("所有包都是最新版本")
        return
    print(f"{'包名':<20} {'当前版本':<12} {'最新版本'}")
    print("-" * 50)
    for pkg in outdated:
        print(f"{pkg['name']:<20} {pkg['current']:<12} {pkg['latest']}")


def pkg_update(args) -> None:
    pm = _get_pkg_manager(args)
    pm.update(package_names=args.packages or None)


def pkg_info(args) -> None:
    pm = _get_pkg_manager(args)
    info = pm.info(args.package)
    if not info:
        print(f"找不到包: {args.package}")
        return
    print(f"包名: {info.get('name', '')}")
    print(f"描述: {info.get('description', '')}")
    versions = info.get("versions", {})
    if versions:
        print(f"版本:")
        for v in sorted(versions.keys(), reverse=True):
            print(f"  {v}")


def pkg_dep_tree(args) -> None:
    pm = _get_pkg_manager(args)
    tree = pm.dependency_tree()
    print(tree)


def pkg_registry_add(args) -> None:
    pm = _get_pkg_manager(args)
    pm.add_registry(
        name=args.name,
        url=args.url,
        mirror_of=getattr(args, "mirror_of", "") or "",
        priority=int(getattr(args, "priority", 0) or 0),
    )


def pkg_registry_remove(args) -> None:
    pm = _get_pkg_manager(args)
    pm.remove_registry(args.name)


def pkg_registry_list(args) -> None:
    pm = _get_pkg_manager(args)
    registries = pm.list_registries()
    if not registries:
        print("没有配置注册表")
        return
    print(f"{'名称':<15} {'URL':<40} {'镜像':<15} {'优先级'}")
    print("-" * 85)
    for r in registries:
        print(f"{r.get('name', ''):<15} {r.get('url', ''):<40} {r.get('mirrorOf', ''):<15} {r.get('priority', 0)}")


def main() -> None:
    arg_parser = argparse.ArgumentParser(prog="nlasm", description="NLASM 自然语言编程语言")
    arg_parser.add_argument("--version", action="version", version="NLASM v0.8.0")

    subparsers = arg_parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="运行 .nl 文件")
    run_parser.add_argument("file", help=".nl 文件路径")

    compile_parser = subparsers.add_parser("compile", help="编译 .nl 文件")
    compile_parser.add_argument("file", help=".nl 文件路径")
    compile_parser.add_argument("-o", "--output", help="输出路径")

    build_parser = subparsers.add_parser("build", help="AOT编译为原生可执行文件")
    build_parser.add_argument("file", help=".nl 文件路径")
    build_parser.add_argument("-o", "--output", help="输出路径")
    build_parser.add_argument(
        "--fmt", choices=["exe", "obj", "asm", "ll"], default="exe",
        help="输出格式: exe(可执行文件), obj(目标文件), asm(汇编), ll(LLVM IR)",
    )

    check_parser = subparsers.add_parser("check", help="语法和类型检查")
    check_parser.add_argument("file", help=".nl 文件路径")

    fmt_parser = subparsers.add_parser("format", help="格式化代码")
    fmt_parser.add_argument("file", help=".nl 文件路径")

    subparsers.add_parser("repl", help="交互式环境")

    init_parser = subparsers.add_parser("init", help="初始化新项目")
    init_parser.add_argument("name", help="项目名称")
    init_parser.add_argument("--author", default="", help="作者")
    init_parser.add_argument("--description", default="", help="项目描述")
    init_parser.add_argument("--project-dir", help="项目目录 (默认: 当前目录)")
    init_parser.add_argument("--registry", help="包仓库地址")

    install_parser = subparsers.add_parser("install", help="安装包依赖")
    install_parser.add_argument("packages", nargs="*", help="包名 (格式: name 或 name@version)")
    install_parser.add_argument("--dev", action="store_true", help="安装为开发依赖")
    install_parser.add_argument("--local", action="store_true", help="从本地仓库安装")
    install_parser.add_argument("--project-dir", help="项目目录")
    install_parser.add_argument("--registry", help="包仓库地址")

    uninstall_parser = subparsers.add_parser("uninstall", help="卸载包")
    uninstall_parser.add_argument("packages", nargs="+", help="要卸载的包名")
    uninstall_parser.add_argument("--project-dir", help="项目目录")
    uninstall_parser.add_argument("--registry", help="包仓库地址")

    publish_parser = subparsers.add_parser("publish", help="发布包到仓库")
    publish_parser.add_argument("--local", action="store_true", help="发布到本地仓库")
    publish_parser.add_argument("--project-dir", help="项目目录")
    publish_parser.add_argument("--registry", help="包仓库地址")

    list_parser = subparsers.add_parser("list", help="列出已安装的包")
    list_parser.add_argument("--project-dir", help="项目目录")
    list_parser.add_argument("--registry", help="包仓库地址")

    outdated_parser = subparsers.add_parser("outdated", help="检查过期的包")
    outdated_parser.add_argument("--project-dir", help="项目目录")
    outdated_parser.add_argument("--registry", help="包仓库地址")

    update_parser = subparsers.add_parser("update", help="更新包到最新版本")
    update_parser.add_argument("packages", nargs="*", help="要更新的包名 (留空则更新全部)")
    update_parser.add_argument("--project-dir", help="项目目录")
    update_parser.add_argument("--registry", help="包仓库地址")

    info_parser = subparsers.add_parser("info", help="查看包信息")
    info_parser.add_argument("package", help="包名")
    info_parser.add_argument("--project-dir", help="项目目录")
    info_parser.add_argument("--registry", help="包仓库地址")

    dep_tree_parser = subparsers.add_parser("dep-tree", help="显示依赖树")
    dep_tree_parser.add_argument("--project-dir", help="项目目录")
    dep_tree_parser.add_argument("--registry", help="包仓库地址")

    registry_parser = subparsers.add_parser("registry", help="管理注册表")
    registry_sub = registry_parser.add_subparsers(dest="registry_command")

    reg_add_parser = registry_sub.add_parser("add", help="添加注册表")
    reg_add_parser.add_argument("name", help="注册表名称")
    reg_add_parser.add_argument("url", help="注册表URL")
    reg_add_parser.add_argument("--mirror-of", default="", help="镜像的源注册表")
    reg_add_parser.add_argument("--priority", type=int, default=0, help="优先级 (数字越小越优先)")
    reg_add_parser.add_argument("--project-dir", help="项目目录")

    reg_remove_parser = registry_sub.add_parser("remove", help="移除注册表")
    reg_remove_parser.add_argument("name", help="注册表名称")
    reg_remove_parser.add_argument("--project-dir", help="项目目录")

    registry_sub.add_parser("list", help="列出所有注册表")

    args = arg_parser.parse_args()

    if args.command == "run":
        run_file(args.file)
    elif args.command == "compile":
        compile_file(args.file, args.output)
    elif args.command == "build":
        build_file(args.file, args.output, args.fmt)
    elif args.command == "check":
        check_file(args.file)
    elif args.command == "format":
        format_file(args.file)
    elif args.command == "repl":
        repl()
    elif args.command == "init":
        pkg_init(args)
    elif args.command == "install":
        pkg_install(args)
    elif args.command == "uninstall":
        pkg_uninstall(args)
    elif args.command == "publish":
        pkg_publish(args)
    elif args.command == "list":
        pkg_list(args)
    elif args.command == "outdated":
        pkg_outdated(args)
    elif args.command == "update":
        pkg_update(args)
    elif args.command == "info":
        pkg_info(args)
    elif args.command == "dep-tree":
        pkg_dep_tree(args)
    elif args.command == "registry":
        if getattr(args, "registry_command", None) == "add":
            pkg_registry_add(args)
        elif getattr(args, "registry_command", None) == "remove":
            pkg_registry_remove(args)
        elif getattr(args, "registry_command", None) == "list":
            pkg_registry_list(args)
        else:
            print("用法: nlasm registry <add|remove|list>")
    else:
        arg_parser.print_help()


if __name__ == "__main__":
    main()

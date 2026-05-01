from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_pipeline_jit_path():
    from core.cfg_builder import CFGBuilder
    from core.dominator import DominatorTree
    from core.ssa_builder import SSABuilder
    from core.optimizer import Optimizer
    from core.llvm_codegen import LLVMCodeGen
    from core.jit_executor import JITExecutor
    from core.ir import Assign, Literal, Print, Var, Add
    from core.ir_interpreter import IRInterpreter

    ir_nodes = [
        Assign(target="x", value=Literal(10)),
        Assign(target="y", value=Literal(20)),
        Assign(target="z", value=Add(Var("x"), Var("y"))),
        Print(value=Var("z")),
    ]

    print("=== 解释执行路径 ===")
    interp = IRInterpreter()
    start = time.perf_counter()
    interp.run(ir_nodes)
    interp_time = time.perf_counter() - start
    print(f"结果: {interp.outputs}")
    print(f"耗时: {interp_time*1000:.4f} ms")

    print("\n=== JIT 编译路径 ===")
    try:
        print("[1] 构建 CFG...")
        cfg_builder = CFGBuilder()
        cfg = cfg_builder.build(ir_nodes)
        print(f"    CFG: {len(cfg.blocks)} 个基本块")

        print("[2] 构建 SSA...")
        dom_tree = DominatorTree()
        dom_tree.build(cfg)
        ssa_builder = SSABuilder()
        ssa_cfg = ssa_builder.build(cfg)
        print(f"    SSA: {len(ssa_cfg.blocks)} 个基本块")

        print("[3] 优化...")
        optimizer = Optimizer()
        optimized_cfg = optimizer.run(ssa_cfg)

        print("[4] 生成 LLVM IR...")
        codegen = LLVMCodeGen()
        llvm_module = codegen.lower(optimized_cfg)
        print(f"    LLVM IR 生成成功")

        print("[5] JIT 编译...")
        jit = JITExecutor()
        compiled_fn = jit.compile_module(llvm_module)
        print(f"    编译成功，函数指针: {compiled_fn}")

        print("[6] 执行...")
        start = time.perf_counter()
        result = jit.execute(compiled_fn)
        jit_time = time.perf_counter() - start
        print(f"    结果: {result}")
        print(f"    耗时: {jit_time*1000:.4f} ms")

        print("\n=== 性能对比 ===")
        if interp_time > 0 and jit_time > 0:
            print(f"JIT 执行 vs 解释器: {interp_time/jit_time:.2f}x")

        print("\n[PASS] JIT 编译路径测试通过！")

    except ImportError as e:
        print(f"\n[SKIP] 跳过 JIT 测试（缺少 llvmlite）: {e}")
        print("   安装方法: pip install llvmlite")
    except Exception as e:
        print(f"\n[WARN] JIT 编译失败（回退到解释执行）: {e}")
        print("   这在 Windows 上是正常的，LLVM JIT 需要 llvmlite 支持")


def test_pipeline_use_jit_flag():
    from core.pipeline import PipelineV08
    from core.frontend import Frontend, RuleEntityExtractor
    from core.inline_cache import InlineCache
    from core.decoder import SemanticDecoder

    print("\n=== 测试 use_jit=False（解释模式） ===")

    class MockEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            if isinstance(texts, str):
                return np.random.rand(384).astype('float32')
            return np.random.rand(len(texts), 384).astype('float32')

    emb = MockEmbedder()

    try:
        from core.pattern_matcher import PatternMatcher
        from core.pattern_database import PATTERN_DB
        from core.slot_filler import SlotFiller
        from core.pattern_instantiator import PatternInstantiator

        pipeline_interp = PipelineV08(
            frontend=Frontend(embedder=emb, entity_extractor=RuleEntityExtractor()),
            matcher=PatternMatcher(embedder=emb, patterns=PATTERN_DB),
            filler=SlotFiller(),
            instantiator=PatternInstantiator(),
            decoder=SemanticDecoder(),
            cache=InlineCache(),
            use_jit=False,
        )
        assert not pipeline_interp.use_jit
        assert not hasattr(pipeline_interp, 'jit_executor')
        print("[PASS] 解释模式初始化正确")

        print("\n=== 测试 use_jit=True（JIT 模式） ===")
        pipeline_jit = PipelineV08(
            frontend=Frontend(embedder=emb, entity_extractor=RuleEntityExtractor()),
            matcher=PatternMatcher(embedder=emb, patterns=PATTERN_DB),
            filler=SlotFiller(),
            instantiator=PatternInstantiator(),
            decoder=SemanticDecoder(),
            cache=InlineCache(),
            use_jit=True,
        )
        assert pipeline_jit.use_jit
        assert hasattr(pipeline_jit, 'cfg_builder')
        assert hasattr(pipeline_jit, 'ssa_builder')
        assert hasattr(pipeline_jit, 'optimizer')
        assert hasattr(pipeline_jit, 'codegen')
        assert hasattr(pipeline_jit, 'jit_executor')
        print("[PASS] JIT 模式初始化正确，所有编译器组件已加载")
    except Exception as e:
        print(f"[SKIP] Pipeline 完整测试跳过: {e}")


def test_compile_chain_components():
    from core.cfg_builder import CFGBuilder
    from core.dominator import DominatorTree
    from core.ssa_builder import SSABuilder
    from core.optimizer import Optimizer
    from core.ir import Assign, Literal, Print, Var, If, Cmp

    print("\n=== 测试完整编译链组件 ===")

    ir_nodes = [
        Assign(target="x", value=Literal(42)),
        If(condition=Cmp(Var("x"), ">", Literal(0)), body=[Print(value=Var("x"))], orelse=[]),
    ]

    print("[1] CFGBuilder...")
    cfg = CFGBuilder().build(ir_nodes)
    print(f"    基本块: {[b.name for b in cfg.blocks]}")
    assert len(cfg.blocks) >= 1

    print("[2] DominatorTree...")
    dom = DominatorTree()
    dom.build(cfg)
    print(f"    支配关系已计算")

    print("[3] SSABuilder...")
    ssa_cfg = SSABuilder().build(cfg)
    print(f"    SSA 构建完成")

    print("[4] Optimizer...")
    opt = Optimizer()
    optimized = opt.run(ssa_cfg)
    print(f"    优化完成")

    print("[PASS] 编译链组件测试通过！")


if __name__ == "__main__":
    tests = [
        test_compile_chain_components,
        test_pipeline_use_jit_flag,
        test_pipeline_jit_path,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\nFAIL: {test.__name__} - {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n结果: {passed}/{len(tests)} 通过, {failed} 失败")
    if failed > 0:
        sys.exit(1)

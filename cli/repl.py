from __future__ import annotations

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_MODEL_PATH = _PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
_HF_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


def _load_embedder():
    """加载语义向量编码器 - 优先本地模型，回退到HuggingFace下载 / Load semantic vector encoder - prefer local model, fallback to HuggingFace download"""
    from sentence_transformers import SentenceTransformer

    if _LOCAL_MODEL_PATH.exists():
        model = SentenceTransformer(str(_LOCAL_MODEL_PATH))
    else:
        model = SentenceTransformer(_HF_MODEL_NAME)

    class _Embedder:
        def __init__(self, st_model):
            self._model = st_model

        def encode(self, text, normalize_embeddings=True):
            return self._model.encode(text, normalize_embeddings=normalize_embeddings)

    return _Embedder(model)


def build_runtime(embedder=None, use_jit: bool = True):
    """构建NLASM运行时 - 组装所有编译流水线组件 / Build NLASM runtime - assemble all compilation pipeline components.

    组件链: Frontend -> PatternMatcher -> SlotFiller -> PatternInstantiator -> Pipeline -> Runtime
    """
    from core.frontend import Frontend, RuleEntityExtractor
    from core.inline_cache import InlineCache
    from core.pattern_database import PATTERN_DB
    from core.pattern_instantiator import PatternInstantiator
    from core.pattern_matcher import PatternMatcher
    from core.pipeline import PipelineV08
    from core.profiler import Profiler
    from core.runtime import Runtime
    from core.slot_filler import SlotFiller
    from core.decoder import SemanticDecoder

    if embedder is None:
        embedder = _load_embedder()

    # 初始化各组件 / Initialize components
    frontend = Frontend(embedder=embedder, entity_extractor=RuleEntityExtractor())
    matcher = PatternMatcher(embedder=embedder, patterns=PATTERN_DB)
    filler = SlotFiller(embedder=embedder)
    instantiator = PatternInstantiator()
    decoder = SemanticDecoder()
    cache = InlineCache()

    # 组装流水线 / Assemble pipeline
    pipeline = PipelineV08(
        frontend=frontend,
        matcher=matcher,
        filler=filler,
        instantiator=instantiator,
        decoder=decoder,
        cache=cache,
        use_jit=use_jit,
    )

    profiler = Profiler()
    return Runtime(pipeline=pipeline, profiler=profiler)


def format_trace(trace: dict) -> str:
    """格式化执行追踪信息 / Format execution trace info"""
    lines = [
        f"pattern = {trace.get('pattern', 'N/A')}",
        f"score   = {trace.get('score', 0.0):.2f}",
        f"slots   = {trace.get('slots', {})}",
        f"cache   = {'hit' if trace.get('cache_hit') else 'miss'}",
        f"result  = {trace.get('result', 'N/A')}",
    ]
    profiling = trace.get("profiling", {})
    if profiling:
        lines.append("profiling:")
        for stage, elapsed in profiling.items():
            lines.append(f"  {stage}: {elapsed:.4f}s")
    return "\n".join(lines)


def main() -> None:
    """REPL主入口 - 自然语言交互式环境 / REPL main entry - natural language interactive environment"""
    use_jit = os.getenv('NLASM_USE_JIT', '1') == '1'  # 环境变量控制JIT开关 / Env var controls JIT toggle

    print("NLASM Engine v0.8.0")
    if use_jit:
        print("[模式] JIT 编译执行 - 语义驱动的即时编译系统")
    else:
        print("[模式] 解释执行（调试模式）")
    print("输入自然语言指令，输入 quit 退出")
    print()

    runtime = build_runtime(use_jit=use_jit)

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
            trace = runtime.run_with_trace(text)
            print(format_trace(trace))
        except Exception as exc:
            print(f"[错误] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

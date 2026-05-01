from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cfg import CFG, BasicBlock
from .ir import (
    Add,
    And,
    Assign,
    Break,
    Cmp,
    Continue,
    Div,
    For,
    ForRange,
    FuncCall,
    FuncDef,
    If,
    IndexAccess,
    Literal,
    Loop,
    Mod,
    Mul,
    Neg,
    Not,
    Or,
    Print,
    Return,
    Sub,
    Var,
    VarDecl,
    While,
)
from .ssa_builder import PhiInstruction


# ============================================================
# 内联小函数 Pass / Inline Small Functions Pass
# ============================================================

INLINE_MAX_BODY_SIZE = 5  # 函数体不超过5条语句则内联 / Inline if body <= 5 statements


class FunctionInliner:
    """函数内联优化 — 将小函数体直接嵌入调用点 / Function inlining — embed small function bodies directly at call sites.

    内联策略:
    1. 函数体 <= INLINE_MAX_BODY_SIZE 条语句 → 内联
    2. 递归函数 → 不内联
    3. 含控制流(If/While/For) → 不内联（避免复杂化）

    Inlining strategy:
    1. Body <= INLINE_MAX_BODY_SIZE statements → inline
    2. Recursive functions → no inline
    3. Contains control flow (If/While/For) → no inline (avoid complexity)
    """

    def __init__(self, functions: dict[str, FuncDef] | None = None) -> None:
        self.functions = functions or {}

    def run(self, cfg: CFG) -> None:
        """在CFG中内联小函数调用 / Inline small function calls in CFG"""
        for block in cfg.blocks:
            new_instrs = []
            for instr in block.instructions:
                inlined = self._try_inline(instr)
                if inlined is not None:
                    new_instrs.extend(inlined)
                else:
                    new_instrs.append(instr)
            block.instructions = new_instrs

    def _try_inline(self, instr) -> list | None:
        """尝试内联一条指令 / Try to inline a single instruction"""
        if not isinstance(instr, Assign):
            return None

        value = instr.value
        if not isinstance(value, FuncCall):
            return None

        func_def = self.functions.get(value.name)
        if func_def is None:
            return None

        # 检查是否可内联 / Check if inlineable
        if not self._can_inline(func_def):
            return None

        # 执行内联替换 / Perform inline substitution
        return self._inline_func(instr.target, func_def, value.args)

    def _can_inline(self, func_def: FuncDef) -> bool:
        """检查函数是否可内联 / Check if function is inlineable"""
        if len(func_def.body) > INLINE_MAX_BODY_SIZE:
            return False

        # 检查递归 / Check for recursion
        for stmt in func_def.body:
            if self._contains_call_to(stmt, func_def.name):
                return False

        # 检查控制流 / Check for control flow
        for stmt in func_def.body:
            if isinstance(stmt, (If, While, For, ForRange, Loop)):
                return False

        return True

    def _contains_call_to(self, stmt, func_name: str) -> bool:
        """检查语句中是否调用了指定函数 / Check if statement calls specified function"""
        if isinstance(stmt, (Assign, VarDecl, Print, Return)):
            return self._expr_contains_call(stmt.value if hasattr(stmt, 'value') and stmt.value else None, func_name)
        return False

    def _expr_contains_call(self, expr, func_name: str) -> bool:
        """检查表达式中是否调用了指定函数 / Check if expression calls specified function"""
        if expr is None:
            return False
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, FuncCall) and node.name == func_name:
                return True
            if isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                stack.append(node.left)
                stack.append(node.right)
        return False

    def _inline_func(self, target: str, func_def: FuncDef, call_args: list) -> list:
        """执行内联替换 / Perform inline substitution"""
        result = []

        # 参数绑定 — 将参数赋值给局部变量 / Argument binding — assign params to local vars
        param_map: dict[str, str] = {}
        for i, (pname, _ptype) in enumerate(func_def.params):
            local_name = f"_inline_{func_def.name}_{pname}"
            param_map[pname] = local_name
            if i < len(call_args):
                result.append(Assign(target=local_name, value=call_args[i]))

        # 复制函数体，替换参数名 / Copy function body, replace param names
        for stmt in func_def.body:
            inlined_stmt = self._rename_vars(stmt, param_map)
            if isinstance(inlined_stmt, Return) and inlined_stmt.value is not None:
                # 返回语句 → 赋值给目标变量 / Return → assign to target variable
                result.append(Assign(target=target, value=inlined_stmt.value))
            elif not isinstance(inlined_stmt, Return):
                result.append(inlined_stmt)

        return result if result else [Assign(target=target, value=Literal(0))]

    def _rename_vars(self, stmt, param_map: dict[str, str]):
        """重命名语句中的变量 / Rename variables in statement"""
        if isinstance(stmt, Assign):
            return Assign(
                target=param_map.get(stmt.target, stmt.target),
                value=self._rename_expr(stmt.value, param_map),
            )
        if isinstance(stmt, Return):
            return Return(
                value=self._rename_expr(stmt.value, param_map) if stmt.value else None,
            )
        if isinstance(stmt, Print):
            return Print(value=self._rename_expr(stmt.value, param_map))
        return stmt

    def _rename_expr(self, expr, param_map: dict[str, str]):
        """重命名表达式中的变量 / Rename variables in expression"""
        stack: list[tuple[object, bool]] = [(expr, False)]
        results: list[object] = []
        while stack:
            node, processed = stack.pop()
            if processed:
                if isinstance(node, Var):
                    results.append(Var(param_map.get(node.name, node.name)))
                elif isinstance(node, (Add, Sub, Mul, Div, Mod)):
                    right = results.pop()
                    left = results.pop()
                    results.append(type(node)(left=left, right=right))
                elif isinstance(node, (And, Or)):
                    right = results.pop()
                    left = results.pop()
                    results.append(type(node)(left=left, right=right))
                elif isinstance(node, Cmp):
                    right = results.pop()
                    left = results.pop()
                    results.append(Cmp(left=left, op=node.op, right=right))
                elif isinstance(node, Neg):
                    operand = results.pop()
                    results.append(Neg(operand=operand))
                elif isinstance(node, Not):
                    operand = results.pop()
                    results.append(Not(operand=operand))
                else:
                    results.append(node)
            else:
                if isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                elif isinstance(node, (Neg, Not)):
                    stack.append((node, True))
                    stack.append((node.operand, False))
                elif isinstance(node, Var):
                    results.append(Var(param_map.get(node.name, node.name)))
                else:
                    results.append(node)
        return results[0] if results else expr


# ============================================================
# 常量折叠优化 / Constant Folding Pass
# ============================================================

class ConstFold:
    """常量折叠 — 编译期计算常量表达式 + 代数简化 / Constant folding — compute constant expressions at compile time + algebraic simplification"""

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            new_instrs = []
            for instr in block.instructions:
                if isinstance(instr, Assign):
                    instr.value = self._fold_expr(instr.value)
                new_instrs.append(instr)
            block.instructions = new_instrs

    def _fold_expr(self, expr):
        stack: list[tuple[object, bool]] = [(expr, False)]
        results: list[object] = []
        while stack:
            node, processed = stack.pop()
            if processed:
                if isinstance(node, Add):
                    right, left = results.pop(), results.pop()
                    if isinstance(left, Literal) and isinstance(right, Literal):
                        if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                            results.append(Literal(left.value + right.value))
                            continue
                        if isinstance(left.value, str) or isinstance(right.value, str):
                            results.append(Literal(str(left.value) + str(right.value)))
                            continue
                    if isinstance(right, Literal) and right.value == 0:
                        results.append(left)
                        continue
                    if isinstance(left, Literal) and left.value == 0:
                        results.append(right)
                        continue
                    results.append(Add(left, right))
                elif isinstance(node, Sub):
                    right, left = results.pop(), results.pop()
                    if isinstance(left, Literal) and isinstance(right, Literal):
                        if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                            results.append(Literal(left.value - right.value))
                            continue
                    if isinstance(right, Literal) and right.value == 0:
                        results.append(left)
                        continue
                    if isinstance(left, Literal) and isinstance(right, Literal) and left.value == right.value:
                        results.append(Literal(0))
                        continue
                    results.append(Sub(left, right))
                elif isinstance(node, Mul):
                    right, left = results.pop(), results.pop()
                    if isinstance(left, Literal) and isinstance(right, Literal):
                        if isinstance(left.value, (int, float)) and isinstance(right.value, (int, float)):
                            results.append(Literal(left.value * right.value))
                            continue
                    if isinstance(right, Literal) and right.value == 1:
                        results.append(left)
                        continue
                    if isinstance(left, Literal) and left.value == 1:
                        results.append(right)
                        continue
                    if isinstance(right, Literal) and right.value == 0:
                        results.append(Literal(0))
                        continue
                    if isinstance(left, Literal) and left.value == 0:
                        results.append(Literal(0))
                        continue
                    if isinstance(right, Literal) and right.value == 2:
                        results.append(Add(left, left))
                        continue
                    results.append(Mul(left, right))
                elif isinstance(node, Div):
                    right, left = results.pop(), results.pop()
                    if isinstance(left, Literal) and isinstance(right, Literal):
                        if isinstance(right.value, (int, float)) and right.value != 0:
                            if isinstance(left.value, (int, float)):
                                if isinstance(left.value, int) and isinstance(right.value, int):
                                    results.append(Literal(left.value // right.value))
                                    continue
                                results.append(Literal(left.value / right.value))
                                continue
                    if isinstance(right, Literal) and right.value == 1:
                        results.append(left)
                        continue
                    results.append(Div(left, right))
                elif isinstance(node, Mod):
                    right, left = results.pop(), results.pop()
                    if isinstance(left, Literal) and isinstance(right, Literal):
                        if isinstance(left.value, int) and isinstance(right.value, int) and right.value != 0:
                            results.append(Literal(left.value % right.value))
                            continue
                    if isinstance(right, Literal) and right.value == 1:
                        results.append(Literal(0))
                        continue
                    results.append(Mod(left, right))
                else:
                    results.append(node)
            else:
                if isinstance(node, (Add, Sub, Mul, Div, Mod)):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                else:
                    results.append(node)
        return results[0] if results else expr


# ============================================================
# 全局值编号 / Global Value Numbering
# ============================================================

class GVN:
    """全局值编号 — 消除冗余计算 / Global Value Numbering — eliminate redundant computations"""

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            seen: dict[str, Assign] = {}
            new_instrs = []
            for instr in block.instructions:
                if isinstance(instr, Assign):
                    key = self._expr_key(instr.value)
                    if key and key in seen:
                        orig = seen[key]
                        instr.value = Var(orig.target)
                    elif key:
                        seen[key] = instr
                new_instrs.append(instr)
            block.instructions = new_instrs

    def _expr_key(self, expr) -> str | None:
        stack: list[tuple[object, bool]] = [(expr, False)]
        results: list[str | None] = []
        while stack:
            node, processed = stack.pop()
            if processed:
                if isinstance(node, Add):
                    rk, lk = results.pop(), results.pop()
                    if lk and rk:
                        parts = sorted([lk, rk])
                        results.append(f"Add({parts[0]},{parts[1]})")
                    else:
                        results.append(None)
                elif isinstance(node, Mul):
                    rk, lk = results.pop(), results.pop()
                    if lk and rk:
                        parts = sorted([lk, rk])
                        results.append(f"Mul({parts[0]},{parts[1]})")
                    else:
                        results.append(None)
                elif isinstance(node, Sub):
                    rk, lk = results.pop(), results.pop()
                    if lk and rk:
                        results.append(f"Sub({lk},{rk})")
                    else:
                        results.append(None)
                else:
                    results.append(None)
            else:
                if isinstance(node, Add):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                elif isinstance(node, Mul):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                elif isinstance(node, Sub):
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                elif isinstance(node, Var):
                    results.append(f"Var({node.name})")
                elif isinstance(node, Literal):
                    results.append(f"Lit({node.value})")
                else:
                    results.append(None)
        return results[0] if results else None


# ============================================================
# 死代码消除 / Dead Code Elimination
# ============================================================

class DCE:
    """死代码消除 — 移除未使用的赋值 / Dead Code Elimination — remove unused assignments"""

    def run(self, cfg: CFG) -> None:
        used: set[str] = set()
        for block in cfg.blocks:
            for instr in block.instructions:
                self._collect_uses(instr, used)

        for block in cfg.blocks:
            new_instrs = []
            for instr in block.instructions:
                if isinstance(instr, Assign) and instr.target not in used:
                    if self._has_side_effects(instr.value):
                        new_instrs.append(instr)
                    continue
                new_instrs.append(instr)
            block.instructions = new_instrs

    def _collect_uses(self, instr, used: set[str]) -> None:
        if isinstance(instr, Assign):
            self._collect_expr_uses(instr.value, used)
        if isinstance(instr, PhiInstruction):
            for ver in instr.sources.values():
                used.add(ver)
        if isinstance(instr, Print):
            self._collect_expr_uses(instr.value, used)
        if isinstance(instr, Return) and instr.value:
            self._collect_expr_uses(instr.value, used)

    def _collect_expr_uses(self, expr, used: set[str]) -> None:
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Var):
                used.add(node.name)
            elif isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                stack.append(node.left)
                stack.append(node.right)
            elif isinstance(node, (Neg, Not)):
                stack.append(node.operand)
            elif isinstance(node, FuncCall):
                for arg in node.args:
                    stack.append(arg)

    def _has_side_effects(self, expr) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, FuncCall):
                return True
            if isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                stack.append(node.left)
                stack.append(node.right)
            elif isinstance(node, (Neg, Not)):
                stack.append(node.operand)
        return False


# ============================================================
# 循环不变量外提 / Loop Invariant Code Motion
# ============================================================

class LICM:
    """循环不变量外提 — 将循环内不变的计算移到循环外 / Loop Invariant Code Motion — move invariant computations out of loops"""

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            if "loop" not in block.name:
                continue

            # 查找循环前驱块（循环入口）/ Find loop predecessor (loop header)
            preheader = None
            for pred in block.preds:
                if "loop" not in pred.name:
                    preheader = pred
                    break

            if preheader is None:
                continue

            # 识别循环不变量 / Identify loop invariants
            loop_vars = self._collect_loop_vars(block)
            invariant_instrs = []

            new_instrs = []
            for instr in block.instructions:
                if isinstance(instr, Assign) and self._is_loop_invariant(instr, loop_vars):
                    invariant_instrs.append(instr)
                else:
                    new_instrs.append(instr)

            # 将不变量移到前驱块末尾 / Move invariants to end of preheader
            if invariant_instrs:
                preheader.instructions.extend(invariant_instrs)
                block.instructions = new_instrs

    def _collect_loop_vars(self, loop_block: BasicBlock) -> set[str]:
        """收集循环内被赋值的变量 / Collect variables assigned inside loop"""
        vars_set: set[str] = set()
        for instr in loop_block.instructions:
            if isinstance(instr, Assign):
                vars_set.add(instr.target)
        return vars_set

    def _is_loop_invariant(self, instr: Assign, loop_vars: set[str]) -> bool:
        """检查赋值是否为循环不变量 / Check if assignment is loop invariant"""
        if instr.target in loop_vars:
            return not self._expr_uses_vars(instr.value, loop_vars)
        return False

    def _expr_uses_vars(self, expr, vars_set: set[str]) -> bool:
        """检查表达式是否使用了指定变量集 / Check if expression uses specified variable set"""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Var):
                if node.name in vars_set:
                    return True
            elif isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                stack.append(node.left)
                stack.append(node.right)
            elif isinstance(node, (Neg, Not)):
                stack.append(node.operand)
        return False


# ============================================================
# 循环强度消减 / Loop Strength Reduction
# ============================================================

class LoopStrengthReduction:
    """循环强度消减 — 将乘法替换为加法 / Loop Strength Reduction — replace multiplication with addition.

    示例: for i in range(n): arr[i] = i * 4 → arr[i] = t; t += 4
    Example: for i in range(n): arr[i] = i * 4 → arr[i] = t; t += 4
    """

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            if "loop" not in block.name:
                continue
            new_instrs = []
            for instr in block.instructions:
                reduced = self._try_reduce(instr)
                if reduced:
                    new_instrs.extend(reduced)
                else:
                    new_instrs.append(instr)
            block.instructions = new_instrs

    def _try_reduce(self, instr) -> list | None:
        """尝试强度消减 / Try strength reduction"""
        if not isinstance(instr, Assign):
            return None
        value = instr.value
        if not isinstance(value, Mul):
            return None
        # 检测 i * constant 模式 / Detect i * constant pattern
        if isinstance(value.right, Literal) and isinstance(value.left, Var):
            const_val = value.right.value
            if isinstance(const_val, int) and const_val > 1:
                acc_name = f"_lsr_{instr.target}"
                return [
                    Assign(target=instr.target, value=Var(acc_name)),
                    Assign(target=acc_name, value=Add(Var(acc_name), Literal(const_val))),
                ]
        return None


# ============================================================
# 逃逸分析 / Escape Analysis
# ============================================================

class EscapeAnalysis:
    """逃逸分析 — 识别不逃逸的对象，允许栈分配 / Escape Analysis — identify non-escaping objects for stack allocation.

    当前为占位实现，标记可优化的分配点。
    Currently a placeholder implementation, marks optimizable allocation points.
    """

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            for instr in block.instructions:
                if isinstance(instr, Assign):
                    self._check_escape(instr)

    def _check_escape(self, instr: Assign) -> None:
        """检查赋值是否逃逸 / Check if assignment escapes"""
        # 标记: 如果值是ListExpr/DictExpr且未被返回/传递，则不逃逸
        # Mark: if value is ListExpr/DictExpr and not returned/passed, it doesn't escape
        pass


# ============================================================
# 向量化标记 / Vectorization Marking
# ============================================================

class VectorizationMarker:
    """向量化标记 — 识别可向量化的循环并标记 / Vectorization marking — identify and mark vectorizable loops.

    可向量化条件:
    1. 循环体内无控制流依赖
    2. 数组访问是连续的
    3. 无循环携带依赖

    Vectorizable conditions:
    1. No control flow dependency in loop body
    2. Array access is contiguous
    3. No loop-carried dependency
    """

    def run(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            if "loop" not in block.name:
                continue
            if self._is_vectorizable(block):
                for instr in block.instructions:
                    if hasattr(instr, '_metadata'):
                        instr._metadata["vectorizable"] = True

    def _is_vectorizable(self, block: BasicBlock) -> bool:
        """检查循环是否可向量化 / Check if loop is vectorizable"""
        has_array_access = False
        has_control_flow = False
        has_loop_carried_dep = False

        assigned_vars: set[str] = set()
        for instr in block.instructions:
            if isinstance(instr, Assign):
                value = instr.value
                # 检测数组访问 / Detect array access
                if isinstance(value, (Add, Sub, Mul)):
                    if self._contains_index_access(value):
                        has_array_access = True
                # 检测控制流 / Detect control flow
                if isinstance(value, (And, Or, Cmp)):
                    has_control_flow = True
                # 检测循环携带依赖 / Detect loop-carried dependency
                if self._expr_uses_vars(value, assigned_vars):
                    has_loop_carried_dep = True
                assigned_vars.add(instr.target)

        return has_array_access and not has_control_flow and not has_loop_carried_dep

    def _contains_index_access(self, expr) -> bool:
        """检查表达式是否包含索引访问 / Check if expression contains index access"""
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, IndexAccess):
                return True
            if isinstance(node, (Add, Sub, Mul, Div)):
                stack.append(node.left)
                stack.append(node.right)
        return False

    def _expr_uses_vars(self, expr, vars_set: set[str]) -> bool:
        stack = [expr]
        while stack:
            node = stack.pop()
            if isinstance(node, Var):
                if node.name in vars_set:
                    return True
            elif isinstance(node, (Add, Sub, Mul, Div, Mod, And, Or, Cmp)):
                stack.append(node.left)
                stack.append(node.right)
        return False


# ============================================================
# 优化器 — 串联所有Pass / Optimizer — chain all passes
# ============================================================

class Optimizer:
    """IR优化器 — 串联执行多个优化Pass / IR optimizer — runs multiple optimization passes in sequence.

    Pass顺序 / Pass order:
    1. FunctionInliner — 函数内联 / Function inlining
    2. ConstFold — 常量折叠 / Constant folding
    3. GVN — 全局值编号 / Global value numbering
    4. DCE — 死代码消除 / Dead code elimination
    5. LICM — 循环不变量外提 / Loop invariant code motion
    6. LoopStrengthReduction — 循环强度消减 / Loop strength reduction
    7. VectorizationMarker — 向量化标记 / Vectorization marking
    8. EscapeAnalysis — 逃逸分析 / Escape analysis
    """

    def __init__(self) -> None:
        self._pattern_hints: dict = {}
        self._functions: dict[str, FuncDef] = {}

    def set_pattern_hints(self, constraints: dict) -> None:
        self._pattern_hints = dict(constraints)

    def set_functions(self, functions: dict[str, FuncDef]) -> None:
        """设置可用函数表（用于内联）/ Set available function table (for inlining)"""
        self._functions = functions

    def run(self, ssa_cfg: CFG) -> CFG:
        """执行所有优化Pass / Run all optimization passes"""
        # 1. 函数内联 / Function inlining
        if self._functions:
            FunctionInliner(self._functions).run(ssa_cfg)

        # 2. 常量折叠 / Constant folding
        ConstFold().run(ssa_cfg)

        # 3. 全局值编号 / Global value numbering
        GVN().run(ssa_cfg)

        # 4. 死代码消除 / Dead code elimination
        DCE().run(ssa_cfg)

        # 5. 循环不变量外提 / Loop invariant code motion
        LICM().run(ssa_cfg)

        # 6. 循环强度消减 / Loop strength reduction
        LoopStrengthReduction().run(ssa_cfg)

        # 7. 向量化标记 / Vectorization marking
        if self._pattern_hints.get("vectorizable") or self._pattern_hints.get("parallelizable"):
            VectorizationMarker().run(ssa_cfg)

        # 8. 逃逸分析 / Escape analysis
        EscapeAnalysis().run(ssa_cfg)

        # 根据Pattern提示标记 / Mark based on pattern hints
        if self._pattern_hints.get("vectorizable"):
            self._mark_vectorizable(ssa_cfg)
        if self._pattern_hints.get("parallelizable"):
            self._mark_parallelizable(ssa_cfg)

        return ssa_cfg

    def _mark_vectorizable(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            if "loop" in block.name:
                for instr in block.instructions:
                    if hasattr(instr, '_metadata'):
                        instr._metadata["vectorizable"] = True

    def _mark_parallelizable(self, cfg: CFG) -> None:
        for block in cfg.blocks:
            if "loop" in block.name:
                for instr in block.instructions:
                    if hasattr(instr, '_metadata'):
                        instr._metadata["parallelizable"] = True

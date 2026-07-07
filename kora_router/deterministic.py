"""Deterministic front-door rules for the KORA routing agent.

These rules resolve a task with a provably-correct computation and no model
call at all. They are answer-blind: the decision is made from the request text,
never from a peeked ground truth. They are also conservative by design. Under
the challenge scoring an accuracy gate must be cleared before token count
matters, so a rule fires only when its output is certain to be correct.
Anything the rules cannot resolve with confidence returns None, and the caller
escalates it to the remote model. This keeps dangerous over-routing at zero:
the rules never guess.

The only rule wired today is a safe arithmetic evaluator. It handles prompts
that reduce to a pure numeric expression (optionally behind a thin natural
wrapper like "what is ... ?"). Everything else is left for escalation. More
rules can plug in behind the same try/return-None contract as real task
samples are observed.
"""

from __future__ import annotations

import ast
import operator
import re
from typing import Any

# Binary and unary operators the safe evaluator permits. Anything outside this
# set (names, calls, attribute access, comprehensions, subscripts) is rejected,
# so the evaluator can only ever compute over literal numbers.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Thin natural-language wrappers stripped before the arithmetic check. Kept
# short and unambiguous so nothing that changes the math slips through.
_WRAPPERS = (
    r"^\s*what\s+is\s+",
    r"^\s*what'?s\s+",
    r"^\s*compute\s+",
    r"^\s*calculate\s+",
    r"^\s*evaluate\s+",
    r"^\s*how\s+much\s+is\s+",
)

# After wrapper stripping, the remainder must contain only these characters to
# be considered a pure arithmetic expression.
_ARITH_ONLY = re.compile(r"^[\d\s.+\-*/()%]+$")

# Guard against pathological exponents that would hang or blow memory. A rule
# that cannot answer quickly should escalate rather than stall the container.
_MAX_POW_EXPONENT = 1000


class _Unsafe(Exception):
    """Raised when an expression contains anything outside the numeric grammar."""


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        # bool is a subclass of int; reject it and any non-numeric literal.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _Unsafe("non-numeric constant")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise _Unsafe("operator not allowed")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise _Unsafe("exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise _Unsafe("unary operator not allowed")
        return op(_eval_node(node.operand))
    raise _Unsafe("node not allowed")


def safe_eval_arithmetic(expr: str) -> float | int:
    """Evaluate a pure arithmetic expression or raise.

    Only literal numbers and + - * / // % ** with parentheses are permitted.
    No names, calls, or attribute access can appear, so this can never execute
    arbitrary code.
    """
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree)


def _extract_arithmetic(prompt: str) -> str | None:
    """Return a pure arithmetic expression from the prompt, or None.

    Conservative: if the prompt is anything other than a clean numeric
    expression (possibly behind one thin wrapper), this returns None so the
    task escalates instead of being guessed at.
    """
    s = prompt.strip().lower()
    s = s.replace("\u00d7", "*").replace("\u00f7", "/")  # unicode times/divide
    s = re.sub(r"[?=]\s*$", "", s).strip()
    for pat in _WRAPPERS:
        stripped = re.sub(pat, "", s)
        if stripped != s:
            s = stripped
            break
    s = s.strip().rstrip(".").strip()
    if not s or not _ARITH_ONLY.match(s):
        return None
    return s


def format_number(value: float | int) -> str:
    """Render a numeric result as a clean string.

    Whole-valued results render without a decimal point; other floats are
    trimmed to a reasonable precision with no trailing zeros. Answer-format
    matching against the real accuracy gate is still to be validated on sample
    tasks, so this is kept simple and centralized for easy adjustment.
    """
    if isinstance(value, int):
        return str(value)
    if value == int(value):
        return str(int(value))
    return f"{value:.10g}"


def try_deterministic(task: dict[str, Any]) -> tuple[str, str] | None:
    """Resolve a task deterministically, or return None to escalate.

    On success returns (answer, reason). Answer-blind: only the request text is
    inspected.
    """
    prompt = str(task.get("prompt") or task.get("text") or "")
    expr = _extract_arithmetic(prompt)
    if expr is None:
        return None
    try:
        value = safe_eval_arithmetic(expr)
    except (_Unsafe, SyntaxError, ValueError, TypeError,
            ZeroDivisionError, OverflowError):
        return None
    return format_number(value), f"deterministic: arithmetic '{expr}'"

"""KORA routing core for the token-efficient agent.

Given a task, the router decides, before any remote inference, how each unit of
work should be handled:

  DETERMINISTIC : answered by a rule / lookup / computation, no model at all.
  LOCAL         : handled by the small local model (free under scoring).
  REMOTE        : escalated to the remote Fireworks model (counts toward score).

This mirrors KORA's front-door philosophy: the cheapest correct path wins, and
the remote model is used only when nothing cheaper can produce a confident,
accurate answer. Local answers are not trusted blindly: every local output
passes a validation gate (non-empty, well-formed, no runaway generation), and
anything that fails the gate escalates to the remote model. Escalation is the
exception path, not a category assignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .fireworks_client import FireworksClient
from .local_model import LocalModel

# System prompt for the local small model. Small instruct models follow short
# prompts more faithfully than long rule lists (long category lists get
# echoed back into the answer), so this stays minimal and generic.
_LOCAL_SYSTEM = (
    "Answer directly. Output only the answer itself. "
    "No explanation, no labels, no extra sections. "
    "Never use markdown or code fences. "
    "Answer in English."
)

# System prompt for numeric word problems on the local path. Direct-answer
# prompting collapses small-model arithmetic (measured 5/12 on a 12-problem
# probe), while brief step-by-step reasoning with a fixed final-answer line
# measured 12/12 on the same probe. Only the extracted final number is ever
# shipped as the answer; the reasoning text stays local.
_LOCAL_NUMERIC_SYSTEM = (
    "You answer math word problems. Think step by step briefly, then end "
    "with a final line in exactly this form: ANSWER: <number>"
)

# System prompt for the remote model. Compact on purpose: it is re-sent on
# every remote call and therefore billed on every remote call. General output
# hygiene only, keyed to the published task categories, never per-dataset.
_REMOTE_SYSTEM = (
    "Give only the answer: no preamble, no explanation, no markdown, and "
    "never use code fences or backticks. For code tasks output only the raw "
    "code. For classification output only the label. Obey any stated length "
    "or format constraint. Answer in English."
)


class Route(str, Enum):
    DETERMINISTIC = "deterministic"
    LOCAL = "local"
    REMOTE = "remote"


@dataclass
class RouteDecision:
    route: Route
    reason: str
    # For DETERMINISTIC, the answer is produced directly by the rule.
    answer: str | None = None
    # For LOCAL, request the numeric step-by-step prompt and final-answer
    # extraction instead of the generic direct-answer prompt.
    numeric_local: bool = False


@dataclass
class TaskResult:
    task_id: str
    answer: str
    route: Route
    reason: str
    remote_prompt_tokens: int = 0
    remote_completion_tokens: int = 0

    @property
    def remote_tokens(self) -> int:
        return self.remote_prompt_tokens + self.remote_completion_tokens


# A decision function inspects a task and returns how to route it. It must be
# answer-blind in the same spirit as the benchmark dispatcher: it decides on the
# request, not on a peeked ground truth.
DecisionFn = Callable[[dict[str, Any]], RouteDecision]


# Signals that a task is a coding task (debug or generation), based on the
# published task categories. Pattern matching on the request only: answer-blind.
_CODE_SIGNALS = (
    "def ", "return ", "function", "code", "bug", "debug", "python",
    "program", "script", "class ", "import ", "print(",
)


def _is_code_task(task: dict[str, Any]) -> bool:
    text = str(task.get("prompt") or task.get("text") or "").lower()
    return any(sig in text for sig in _CODE_SIGNALS)


# --- output hygiene, shared by local and remote paths ----------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+#.-]*\n?|```")


def strip_fences(text: str) -> str:
    """Remove markdown code fences while keeping the fenced content."""
    return _FENCE_RE.sub("", text or "").strip()


def _collapse_repeats(text: str) -> str:
    """Collapse a verbatim-repeated answer into a single copy.

    Small models sometimes emit the same answer block twice. If every
    paragraph-separated block is identical, keep one; otherwise leave the
    text untouched.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    if len(blocks) >= 2 and len(set(blocks)) == 1:
        return blocks[0]
    return (text or "").strip()


def clean_answer(text: str) -> str:
    return _collapse_repeats(strip_fences(text))


# --- numeric final-answer extraction ----------------------------------------

_NUMERIC_ANSWER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_numeric_answer(text: str) -> str:
    """Pull the final number after the ANSWER: marker, or '' to escalate.

    Conservative: without the marker we do not guess which number in the
    reasoning is the final one; the empty result escalates to remote.
    """
    if not text or "ANSWER:" not in text:
        return ""
    tail = text.rsplit("ANSWER:", 1)[-1]
    nums = _NUMERIC_ANSWER_RE.findall(tail.replace(",", ""))
    return nums[-1] if nums else ""


# Validation gate for local answers. Conservative and content-blind: it checks
# form, not correctness. Anything that fails is escalated to remote instead of
# being submitted, so a misbehaving local generation can lower token savings
# but never silently ships a malformed answer.
_RUNAWAY_MARKERS = ("question:", "answer directly", "elaborated", "###")
_MAX_ANSWER_CHARS = 1500
_MAX_ANSWER_LINES = 40


def local_answer_ok(answer: str) -> tuple[bool, str]:
    if not answer:
        return False, "empty answer"
    low = answer.lower()
    for marker in _RUNAWAY_MARKERS:
        if marker in low:
            return False, f"runaway generation ({marker!r})"
    if len(answer) > _MAX_ANSWER_CHARS:
        return False, "answer too long"
    if answer.count("\n") > _MAX_ANSWER_LINES:
        return False, "too many lines"
    return True, "ok"


class Router:
    def __init__(self, decide: DecisionFn, local: LocalModel,
                 remote: FireworksClient,
                 remote_code: FireworksClient | None = None) -> None:
        self._decide = decide
        self._local = local
        self._remote = remote
        # Optional code-specialized remote backend. When present, coding tasks
        # that reach the remote path go to it instead of the general model.
        self._remote_code = remote_code

    def run_task(self, task: dict[str, Any]) -> TaskResult:
        task_id = str(task.get("task_id", task.get("id", "")))
        decision = self._decide(task)

        if decision.route is Route.DETERMINISTIC:
            return TaskResult(
                task_id=task_id,
                answer=decision.answer or "",
                route=decision.route,
                reason=decision.reason,
            )

        if decision.route is Route.LOCAL:
            if decision.numeric_local:
                out = self._local.chat(
                    _messages(task, _LOCAL_NUMERIC_SYSTEM), max_tokens=256)
                answer = extract_numeric_answer(out.text)
                if answer:
                    return TaskResult(
                        task_id=task_id,
                        answer=answer,
                        route=Route.LOCAL,
                        reason=decision.reason,
                    )
                # No trustworthy final answer: escalate, never guess.
                return self._run_remote(
                    task, task_id,
                    reason="local escalation: numeric extraction failed")
            out = self._local.chat(_messages(task, _LOCAL_SYSTEM),
                                   max_tokens=256)
            answer = clean_answer(out.text)
            ok, why = local_answer_ok(answer)
            if ok:
                return TaskResult(
                    task_id=task_id,
                    answer=answer,
                    route=Route.LOCAL,
                    reason=decision.reason,
                )
            # Validation failed: escalate this task to the remote model.
            return self._run_remote(
                task, task_id, reason=f"local escalation: {why}")

        return self._run_remote(task, task_id, reason=decision.reason)

    def _run_remote(self, task: dict[str, Any], task_id: str,
                    reason: str) -> TaskResult:
        client = self._remote
        if self._remote_code is not None and _is_code_task(task):
            client = self._remote_code
        out = client.chat(_messages(task, _REMOTE_SYSTEM))
        return TaskResult(
            task_id=task_id,
            answer=clean_answer(out.text),
            route=Route.REMOTE,
            reason=reason,
            remote_prompt_tokens=out.prompt_tokens,
            remote_completion_tokens=out.completion_tokens,
        )


def _messages(task: dict[str, Any], default_system: str) -> list[dict]:
    """Adapt a task dict into chat messages. Minimal and task-agnostic."""
    system = task.get("system") or default_system
    user = task.get("prompt") or task.get("text") or ""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(user)})
    return messages

"""KORA routing core for the token-efficient agent.

Given a task, the router decides, before any remote inference, how each unit of
work should be handled:

  DETERMINISTIC : answered by a rule / lookup / computation, no model at all.
  LOCAL         : handled by the small local model (free under scoring).
  REMOTE        : escalated to the remote Fireworks model (counts toward score).

This mirrors KORA's front-door philosophy: the cheapest correct path wins, and
the remote model is used only when nothing cheaper can produce a confident,
accurate answer. The per-task decision logic is attached in main; this module
fixes the decision types, the routing contract, and the accounting so that
logic plugs in without reshaping the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .fireworks_client import FireworksClient
from .local_model import LocalModel

# Default system prompt used when a task does not carry its own. The remote
# judge scores answers against expected intent, so the model must respond with
# a clean, direct answer and no decorative formatting. These are general output
# hygiene rules keyed to the published task categories, not per-dataset tuning:
# the router stays answer-blind and applies the same guidance to every task.
_DEFAULT_SYSTEM = (
    "You are a precise task-solving assistant. Follow these rules for every "
    "answer:\n"
    "- Give only the answer. No preamble, no explanation, and do not restate "
    "the question, unless the task explicitly asks you to show reasoning.\n"
    "- Do not use markdown, LaTeX, bold, headers, or tables. Do not wrap output "
    "in code fences, except when the task is to write or fix code, in which "
    "case output only the code.\n"
    "- For classification, output only the label.\n"
    "- For summaries, obey any length or format constraint stated in the task.\n"
    "- For entity extraction, list each entity with its type, one per line.\n"
    "- Keep the answer short and directly responsive to what is asked.\n"
    "- Answer in English."
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


class Router:
    def __init__(self, decide: DecisionFn, local: LocalModel,
                 remote: FireworksClient) -> None:
        self._decide = decide
        self._local = local
        self._remote = remote

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

        messages = _to_messages(task)

        if decision.route is Route.LOCAL:
            out = self._local.chat(messages)
            return TaskResult(
                task_id=task_id,
                answer=out.text,
                route=decision.route,
                reason=decision.reason,
            )

        # REMOTE
        out = self._remote.chat(messages)
        return TaskResult(
            task_id=task_id,
            answer=out.text,
            route=decision.route,
            reason=decision.reason,
            remote_prompt_tokens=out.prompt_tokens,
            remote_completion_tokens=out.completion_tokens,
        )


def _to_messages(task: dict[str, Any]) -> list[dict]:
    """Adapt a task dict into chat messages. Minimal and task-agnostic."""
    system = task.get("system") or _DEFAULT_SYSTEM
    user = task.get("prompt") or task.get("text") or ""
    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    messages.append({"role": "user", "content": str(user)})
    return messages

"""Entry point for running the KORA routing agent over a task set.

Scoring-harness contract:

    input  : /input/tasks.json   -> [{"task_id": ..., "prompt": ...}, ...]
    output : /output/results.json -> [{"task_id": ..., "answer": ...}, ...]

The container runs with no arguments and uses those default paths. Both can be
overridden for local testing. Routing decisions and token accounting are
written to stdout and, optionally, to a debug sidecar; the scored results file
stays a clean array of {task_id, answer}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .deterministic import try_deterministic
from .fireworks_client import FireworksClient
from .local_model import LocalModel
from .router import Route, RouteDecision, Router

DEFAULT_TASKS = "/input/tasks.json"
DEFAULT_OUT = "/output/results.json"


# Numeric word problems (quantities embedded in prose, asked as "how many /
# how much") are the one category where small local models reliably
# miscalculate, and the deterministic evaluator only accepts pure expressions,
# so these go straight to the remote model. Answer-blind: the signal is the
# shape of the request, never a peeked answer.
_MATH_QUERY = re.compile(
    r"how\s+(many|much)|what\s+(number|fraction|percent(age)?)", re.IGNORECASE)


def _is_numeric_word_problem(task: dict[str, Any]) -> bool:
    text = str(task.get("prompt") or task.get("text") or "")
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    return len(numbers) >= 2 and bool(_MATH_QUERY.search(text))


def default_decision(task: dict[str, Any]) -> RouteDecision:
    """KORA front-door policy: the cheapest correct path wins.

    Deterministic rules first: if a rule can resolve the task with a
    provably-correct answer and no model, take it (zero tokens). Numeric word
    problems escalate straight to remote (known local-model weak spot).
    Everything else is answered by the local model at zero remote tokens; the
    router's validation gate escalates any malformed local output to remote.
    """
    resolved = try_deterministic(task)
    if resolved is not None:
        answer, reason = resolved
        return RouteDecision(route=Route.DETERMINISTIC, reason=reason,
                             answer=answer)
    if _is_numeric_word_problem(task):
        return RouteDecision(
            route=Route.REMOTE,
            reason="escalate: numeric word problem (local unreliable)")
    return RouteDecision(route=Route.LOCAL,
                         reason="local-first: zero remote tokens")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        return list(data["tasks"])
    if isinstance(data, list):
        return data
    raise ValueError("unrecognized task file shape")


def _model_basename(model_id: str) -> str:
    """Last path segment: accounts/fireworks/models/minimax-m3 -> minimax-m3."""
    return model_id.rsplit("/", 1)[-1].strip()


def select_model(explicit: str) -> str:
    """Pick the remote model from ALLOWED_MODELS.

    Prefers the explicit override when its basename matches an allowed entry,
    returning the allow-list's own spelling so the harness format is
    preserved. Falls back to the first allowed model otherwise. The allow-list
    is controlled by the harness, so no model id outside it can be selected.
    """
    allowed = [m.strip() for m in os.getenv("ALLOWED_MODELS", "").split(",")
               if m.strip()]
    if explicit and not allowed:
        return explicit
    if explicit and allowed:
        want = _model_basename(explicit)
        for a in allowed:
            if _model_basename(a) == want:
                return a
    if allowed:
        return allowed[0]
    if explicit:
        return explicit
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=DEFAULT_TASKS, help="path to task JSON")
    ap.add_argument("--out", default=DEFAULT_OUT, help="path to results JSON")
    ap.add_argument("--remote-model", default=os.getenv("REMOTE_MODEL", ""),
                    help="Fireworks model id; must be in ALLOWED_MODELS")
    ap.add_argument("--debug-out", default=os.getenv("KORA_DEBUG_OUT", ""),
                    help="optional path for per-task routing diagnostics")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    model = select_model(args.remote_model)
    local = LocalModel()
    remote = FireworksClient(model=model)
    # Code-specialized backend: used only when the allow-list offers a code
    # model distinct from the general one. Selection goes through the same
    # allow-list matching, so nothing outside ALLOWED_MODELS can be chosen.
    code_model = select_model("kimi-k2p7-code")
    remote_code = None
    if code_model and code_model != model:
        remote_code = FireworksClient(model=code_model)
    # Time-budget watchdog: past the soft budget, tasks that would run on the
    # slower local path are escalated to remote instead, so the container
    # always finishes inside the scoring window. Deterministic answers stay
    # deterministic (they are instant).
    t0 = time.time()
    budget = float(os.getenv("KORA_TIME_BUDGET", "520"))

    def decide_with_budget(task):
        d = default_decision(task)
        if d.route is Route.LOCAL and time.time() - t0 > budget:
            return RouteDecision(
                route=Route.REMOTE,
                reason="time budget exceeded: local skipped")
        return d

    router = Router(decide=decide_with_budget, local=local, remote=remote,
                    remote_code=remote_code)

    results: list[dict[str, Any]] = []
    debug: list[dict[str, Any]] = []
    for task in tasks:
        # Recover the id defensively so a failure still reports under the right
        # task_id. A single task must never crash the whole run: on any error we
        # record an empty answer and keep going, so the container still writes a
        # complete results file and exits 0.
        task_id = str(task.get("task_id", task.get("id", "")))
        try:
            r = router.run_task(task)
            answer, route, reason, rtok = (
                r.answer, r.route.value, r.reason, r.remote_tokens)
            ptok, ctok = r.remote_prompt_tokens, r.remote_completion_tokens
        except Exception as exc:
            answer, route = "", "error"
            reason = f"error: {type(exc).__name__}: {exc}"
            rtok = 0
            ptok = ctok = 0
        results.append({"task_id": task_id, "answer": answer})
        debug.append({
            "task_id": task_id,
            "route": route,
            "reason": reason,
            "remote_tokens": rtok,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    total_remote = sum(d["remote_tokens"] for d in debug)
    route_counts = _route_counts(debug)
    if args.debug_out:
        dbg_path = Path(args.debug_out)
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        dbg_path.write_text(
            json.dumps({
                "summary": {
                    "n_tasks": len(results),
                    "total_remote_tokens": total_remote,
                    "remote_calls": remote.usage.calls + (remote_code.usage.calls if remote_code else 0),
                    "route_counts": route_counts,
                },
                "tasks": debug,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"wrote {out_path}: {len(results)} tasks, "
          f"{total_remote} remote tokens, "
          f"{remote.usage.calls + (remote_code.usage.calls if remote_code else 0)} remote calls, "
          f"routes={route_counts}")


def _route_counts(debug: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in debug:
        counts[d["route"]] = counts.get(d["route"], 0) + 1
    return counts


if __name__ == "__main__":
    main()

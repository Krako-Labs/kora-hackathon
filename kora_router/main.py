"""Entry point for running the KORA routing agent over a task set.

Usage (finalized on launch day once the task I/O format is published):

    python -m kora_router.main --tasks tasks.json --out results.json

For now this wires the pipeline end-to-end with a placeholder decision function
so the container is runnable and testable before the tasks are released.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .fireworks_client import FireworksClient
from .local_model import LocalModel
from .router import Route, RouteDecision, Router


def default_decision(task: dict[str, Any]) -> RouteDecision:
    """Placeholder routing policy.

    Replaced on launch day with the real KORA decision logic (deterministic
    rules first, local model for cheap-but-non-trivial work, remote only when
    nothing cheaper is confident). Until then, everything escalates to remote so
    the pipeline produces answers end-to-end.
    """
    return RouteDecision(route=Route.REMOTE, reason="placeholder: escalate all")


def load_tasks(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tasks" in data:
        return list(data["tasks"])
    if isinstance(data, list):
        return data
    raise ValueError("unrecognized task file shape")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="path to task JSON")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--remote-model",
                    default=os.getenv("REMOTE_MODEL", ""),
                    help="Fireworks model id (accounts/fireworks/models/...)")
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    local = LocalModel()
    remote = FireworksClient(model=args.remote_model)
    router = Router(decide=default_decision, local=local, remote=remote)

    results = []
    for task in tasks:
        r = router.run_task(task)
        results.append({
            "id": r.task_id,
            "answer": r.answer,
            "route": r.route.value,
            "reason": r.reason,
            "remote_tokens": r.remote_tokens,
        })

    total_remote = sum(r["remote_tokens"] for r in results)
    payload = {
        "results": results,
        "summary": {
            "n_tasks": len(results),
            "total_remote_tokens": total_remote,
            "remote_calls": remote.usage.calls,
            "route_counts": _route_counts(results),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}: {len(results)} tasks, "
          f"{total_remote} remote tokens, {remote.usage.calls} remote calls")


def _route_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r["route"]] = counts.get(r["route"], 0) + 1
    return counts


if __name__ == "__main__":
    main()

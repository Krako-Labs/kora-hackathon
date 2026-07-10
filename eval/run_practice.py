#!/usr/bin/env python3
"""End-to-end practice run through the full routing pipeline.

Exercises the same code path the scored container runs (deterministic ->
local with validation gate -> remote escalation) against a practice task set,
and prints per-task routes, answers, and remote token spend. Run this before
any image push: the acceptance bar is every task routed as expected and the
remote token total within budget.

Usage:
    python3 eval/run_practice.py /path/to/tasks.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from kora_router.fireworks_client import FireworksClient
from kora_router.local_model import LocalModel
from kora_router.main import default_decision, select_model
from kora_router.router import Router


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_practice.py /path/to/tasks.json")
        return 2
    tasks = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if isinstance(tasks, dict) and "tasks" in tasks:
        tasks = tasks["tasks"]

    import os
    model = select_model(os.getenv("REMOTE_MODEL", "minimax-m3"))
    remote = FireworksClient(model=model)
    code_model = select_model("kimi-k2p7-code")
    remote_code = FireworksClient(model=code_model) if (
        code_model and code_model != model) else None
    router = Router(decide=default_decision, local=LocalModel(),
                    remote=remote, remote_code=remote_code)

    total_remote = 0
    t_all = time.time()
    for task in tasks:
        t0 = time.time()
        r = router.run_task(task)
        dt = time.time() - t0
        total_remote += r.remote_tokens
        print(f"\n=== {r.task_id} [{r.route.value}] ({dt:.1f}s, "
              f"remote={r.remote_tokens}) ===")
        print(f"    reason: {r.reason}")
        print(r.answer)

    calls = remote.usage.calls + (remote_code.usage.calls if remote_code else 0)
    print(f"\nTOTAL: {total_remote} remote tokens, {calls} remote calls, "
          f"{time.time()-t_all:.1f}s wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

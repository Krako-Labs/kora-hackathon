#!/usr/bin/env python3
"""Local eval for the KORA front-door (deterministic path).

Runs each case through default_decision and scores routing against the labeled
expectations. The deterministic path is fully checkable offline (no remote):

  - deterministic precision: of the cases the router resolved deterministically,
    how many produced the correct answer. This is the accuracy-gate safety
    number: a wrong deterministic answer would fail the gate.
  - over-routing (false deflect): cases whose expected_route is remote but that
    the router resolved deterministically. Target is zero.
  - deterministic recall on math: of the cases meant to be deterministic, how
    many the router actually caught (coverage / token savings signal).

Remote answer quality is intentionally out of scope here: remote tasks are only
checked for being routed to remote, not for the content of the model reply
(that depends on the model and the official gate).

Usage:
    python3 eval/run_eval.py            # uses eval/eval_set.json
    python3 eval/run_eval.py path.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kora_router.main import default_decision
from kora_router.router import Route


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("eval/eval_set.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]

    det_total = 0            # cases router resolved deterministically
    det_correct = 0          # of those, answer matched expected_answer
    det_unverifiable = 0     # deterministic but no expected_answer to check
    over_routing = []        # expected remote, router deflected (bad)
    expected_det = 0         # cases labeled deterministic
    expected_det_caught = 0  # of those, router deflected (good coverage)
    route_mismatch = []      # any expected_route vs actual mismatch

    for c in cases:
        decision = default_decision({"task_id": c["task_id"], "prompt": c["prompt"]})
        actual = decision.route.value
        expected = c["expected_route"]

        if decision.route is Route.DETERMINISTIC:
            det_total += 1
            if "expected_answer" in c:
                if (decision.answer or "") == c["expected_answer"]:
                    det_correct += 1
                else:
                    route_mismatch.append(
                        (c["task_id"], "wrong-answer",
                         f"got {decision.answer!r} want {c['expected_answer']!r}"))
            else:
                det_unverifiable += 1
            if expected != "deterministic":
                over_routing.append((c["task_id"], c["prompt"]))

        if expected == "deterministic":
            expected_det += 1
            if decision.route is Route.DETERMINISTIC:
                expected_det_caught += 1

        if actual != expected:
            route_mismatch.append((c["task_id"], f"route {actual} != {expected}",
                                   c["prompt"]))

    n = len(cases)
    det_verified = det_correct + det_unverifiable
    precision = (det_correct / (det_correct + _wrong_answers(route_mismatch))
                 if det_total else 1.0)

    print(f"cases: {n}")
    print(f"deterministic resolved: {det_total}")
    print(f"  correct answers: {det_correct}")
    print(f"  unverifiable (no expected_answer): {det_unverifiable}")
    print(f"deterministic precision (answer correct | deflected, verifiable): "
          f"{_ratio(det_correct, det_correct + _wrong_answers(route_mismatch))}")
    print(f"math coverage (deflected | expected deterministic): "
          f"{_ratio(expected_det_caught, expected_det)}")
    print(f"over-routing (deflected but should escalate): {len(over_routing)}  "
          f"<-- target 0")
    for tid, prompt in over_routing:
        print(f"    OVER-ROUTED {tid}: {prompt}")

    hard_fail = _wrong_answers(route_mismatch) > 0 or len(over_routing) > 0
    print("\n--- route mismatches ---")
    if not route_mismatch:
        print("none: every case routed as expected")
    else:
        for row in route_mismatch:
            print("   ", row)

    print("\nRESULT:", "FAIL" if hard_fail else "PASS")
    return 1 if hard_fail else 0


def _wrong_answers(mismatches) -> int:
    return sum(1 for m in mismatches if len(m) > 1 and m[1] == "wrong-answer")


def _ratio(num: int, den: int) -> str:
    if den == 0:
        return "n/a (0 cases)"
    return f"{num}/{den} = {num / den:.3f}"


if __name__ == "__main__":
    raise SystemExit(main())

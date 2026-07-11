"""Local small-model layer for the KORA routing agent.

Runs a quantized instruct model on CPU via llama.cpp. Local inference is free
under the challenge scoring, so every task the local model answers correctly
costs zero remote tokens. The model loads lazily on first use and is reused
across all tasks in a run, so a task set that never routes locally pays no
load cost at all.

The backend stays pluggable behind the `LocalBackend` protocol: everything
above this module depends only on the `LocalResult` shape and the `generate`
method, so swapping the runtime or the weights does not touch the router.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

# Weights are baked into the image at build time; the path is overridable for
# local testing outside the container.
DEFAULT_MODEL_PATH = os.getenv(
    "KORA_LOCAL_MODEL", "/app/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")

def _available_cpus() -> int:
    """CPUs actually available to this process, cgroup-aware.

    os.cpu_count() reports the host's cores even inside a CPU-limited
    container, which oversubscribes threads and thrashes. Prefer the cgroup
    quota, then the scheduler affinity mask, then cpu_count.
    """
    try:  # cgroup v2
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            n = int(int(quota) / int(period))
            if n >= 1:
                return n
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            n = int(q / p)
            if n >= 1:
                return n
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 4



@dataclass
class LocalResult:
    text: str
    # Local tokens are free under scoring, but we still track them for analysis.
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LocalBackend(Protocol):
    def generate(self, messages: list[dict], *, temperature: float,
                 max_tokens: int) -> LocalResult:
        ...


class LlamaCppBackend:
    """CPU inference over a GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str | None = None, n_ctx: int = 2048,
                 n_threads: int | None = None) -> None:
        self._model_path = model_path or DEFAULT_MODEL_PATH
        self._n_ctx = n_ctx
        env_threads = os.getenv("KORA_LOCAL_THREADS", "")
        self._n_threads = (int(env_threads) if env_threads.isdigit()
                           else n_threads or _available_cpus())
        self._llm = None

    def _ensure(self):
        if self._llm is None:
            from llama_cpp import Llama  # deferred: only needed on local route
            self._llm = Llama(model_path=self._model_path, n_ctx=self._n_ctx,
                              n_threads=self._n_threads, verbose=False)
        return self._llm

    def generate(self, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 512) -> LocalResult:
        llm = self._ensure()
        # Hard per-task wall-clock limit, checked per streamed token. On
        # expiry we return an empty result: the router's validation gate
        # treats empty output as a failure and escalates to remote, so no
        # new code path is needed. This bounds the cost of a slow or
        # runaway local generation on constrained scoring hardware.
        limit = float(os.getenv("KORA_LOCAL_TASK_TIMEOUT", "40"))
        start = time.monotonic()
        pieces: list[str] = []
        n_out = 0
        stream = llm.create_chat_completion(
            messages=messages, temperature=temperature,
            max_tokens=max_tokens, stream=True)
        for chunk in stream:
            if time.monotonic() - start > limit:
                return LocalResult(text="", prompt_tokens=0,
                                   completion_tokens=n_out)
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                pieces.append(piece)
                n_out += 1
        return LocalResult(
            text="".join(pieces).strip(),
            prompt_tokens=0,
            completion_tokens=n_out,
        )


class LocalModel:
    def __init__(self, backend: LocalBackend | None = None) -> None:
        # Lazy default so importing this module never requires llama_cpp;
        # the backend is only constructed when a task actually routes local.
        self._backend = backend

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> LocalResult:
        if self._backend is None:
            self._backend = LlamaCppBackend()
        return self._backend.generate(
            messages, temperature=temperature, max_tokens=max_tokens)

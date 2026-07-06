"""Local small-model wrapper.

The routing layer sends "easy" work here instead of the remote model. Under
the challenge scoring, tokens spent locally count as zero, so the local model
should be sized to run inside the standardized scoring environment (a small
open model such as Gemma is the intended fit).

The concrete backend is deliberately pluggable: the exact model and runtime
(transformers, llama.cpp, a served endpoint, etc.) are fixed on launch day once
the allowed models and environment constraints are published. Everything above
this module only depends on the `LocalResult` shape and the `generate` method,
so swapping the backend does not touch the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


class EchoBackend:
    """Placeholder backend used until the real local model is wired in.

    Returns an empty completion so the pipeline is runnable end-to-end before
    launch day. Replaced by the actual small-model backend once the allowed
    models are known.
    """

    def generate(self, messages: list[dict], *, temperature: float = 0.0,
                 max_tokens: int = 512) -> LocalResult:
        return LocalResult(text="", prompt_tokens=0, completion_tokens=0)


class LocalModel:
    def __init__(self, backend: LocalBackend | None = None) -> None:
        self._backend = backend or EchoBackend()

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> LocalResult:
        return self._backend.generate(
            messages, temperature=temperature, max_tokens=max_tokens
        )

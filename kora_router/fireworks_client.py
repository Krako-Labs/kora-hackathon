"""Remote model client for the Fireworks AI inference API.

Fireworks exposes an OpenAI-compatible endpoint, so the standard `openai`
client works by pointing `base_url` at the Fireworks inference host. Every call
records prompt/completion token usage, which the routing layer uses to account
for remote spend (local tokens are free under the challenge scoring).

The base URL and API key are read from the environment the scoring harness
injects (FIREWORKS_BASE_URL, FIREWORKS_API_KEY), so every remote call goes
through the provided endpoint rather than a hardcoded host. The client is
created lazily on first use, so a task set that is fully resolved without any
remote call runs even when no credentials are present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

try:  # openai is only needed when a remote call actually happens
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without the dep
    OpenAI = None  # type: ignore[assignment]

# Fallback only. The harness injects FIREWORKS_BASE_URL; that value wins.
DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"


@dataclass
class RemoteUsage:
    """Running tally of remote token spend."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1


@dataclass
class RemoteResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class FireworksClient:
    """Thin wrapper over the OpenAI-compatible Fireworks chat endpoint."""

    def __init__(self, model: str, api_key: str | None = None,
                 base_url: str | None = None,
                 timeout: float | None = None) -> None:
        self._model = model
        self._api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        self._base_url = (base_url or os.getenv("FIREWORKS_BASE_URL")
                          or DEFAULT_BASE_URL)
        # Per-call ceiling so one hung request cannot consume the total time
        # budget. Kept just under the per-request limit; overridable via env.
        self._timeout = (timeout if timeout is not None
                         else float(os.getenv("KORA_REMOTE_TIMEOUT", "25")))
        self._client = None
        self.usage = RemoteUsage()

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):
        if self._client is None:
            if OpenAI is None:
                raise RuntimeError("openai package is not installed")
            if not self._api_key:
                raise RuntimeError("FIREWORKS_API_KEY is not set")
            self._client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        return self._client

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> RemoteResult:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self._timeout,
        )
        usage = resp.usage
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.usage.add(prompt, completion)
        return RemoteResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=self._model,
        )

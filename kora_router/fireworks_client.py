"""Remote model client for the Fireworks AI inference API.

Fireworks exposes an OpenAI-compatible endpoint, so the standard `openai`
client works by pointing `base_url` at the Fireworks inference host. Every
call records prompt/completion token usage, which the routing layer uses to
account for remote spend (local tokens are free under the challenge scoring).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


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
                 base_url: str = FIREWORKS_BASE_URL) -> None:
        key = api_key or os.getenv("FIREWORKS_API_KEY")
        if not key:
            raise RuntimeError("FIREWORKS_API_KEY is not set")
        self._client = OpenAI(base_url=base_url, api_key=key)
        self.model = model
        self.usage = RemoteUsage()

    def chat(self, messages: list[dict], *, temperature: float = 0.0,
             max_tokens: int = 512) -> RemoteResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = resp.usage
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        self.usage.add(prompt, completion)
        return RemoteResult(
            text=resp.choices[0].message.content or "",
            prompt_tokens=prompt,
            completion_tokens=completion,
            model=self.model,
        )

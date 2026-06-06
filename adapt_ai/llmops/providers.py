"""Provider abstraction so the pipeline/baseline/judge are model-agnostic.

One synchronous `complete()` returning text + token counts. Two backends:
  - AnthropicProvider         — Anthropic Messages API (system is a top-level kwarg)
  - OpenAICompatibleProvider  — OpenAI Chat Completions (also Ollama at /v1, vLLM, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CompletionResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


class LLMProvider:
    """Interface. `complete()` is synchronous (callers run it off the async loop)."""
    model: str

    def complete(self, *, system: str, user: str, max_tokens: int,
                 temperature: float) -> CompletionResult:  # pragma: no cover
        raise NotImplementedError


class AnthropicProvider(LLMProvider):
    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    def complete(self, *, system: str, user: str, max_tokens: int,
                 temperature: float) -> CompletionResult:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return CompletionResult(
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=self.model,
        )


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI Chat Completions shape — works for Ollama, vLLM, OpenAI, etc.
    System prompt is folded into the messages array (no top-level system kwarg)."""

    def __init__(self, client: Any, model: str) -> None:
        self._client = client
        self.model = model

    def complete(self, *, system: str, user: str, max_tokens: int,
                 temperature: float) -> CompletionResult:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        usage = resp.usage
        return CompletionResult(
            text=resp.choices[0].message.content or "",
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            model=self.model,
        )


def get_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMProvider:
    """Build a provider from explicit args or settings."""
    from adapt_ai.config import settings

    provider = provider or settings.llm_provider
    model = model or settings.model_name

    if provider == "anthropic":
        from anthropic import Anthropic
        key = api_key or settings.anthropic_api_key.get_secret_value()
        return AnthropicProvider(Anthropic(api_key=key), model)

    if provider == "openai_compatible":
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key or settings.llm_api_key or "ollama",
            base_url=base_url or settings.llm_base_url,
        )
        return OpenAICompatibleProvider(client, model)

    raise ValueError(f"Unknown llm_provider: {provider!r}")

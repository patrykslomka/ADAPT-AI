"""Provider-agnostic LLM-as-judge for response correctness scoring.

The judge model MUST differ from the system-under-test model to avoid
self-preference bias. A ValueError is raised at construction time if they match.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from adapt_ai.llmops.providers import LLMProvider

logger = logging.getLogger(__name__)

_RUBRIC_PROMPT = """\
Reference answer:
{reference}

Candidate response:
{prediction}

Score the candidate response 0.0–1.0 for correctness relative to the reference:
- 1.0 = Fully correct; matches the reference conclusion; no important gaps
- 0.7–0.9 = Mostly correct with minor gaps or extra caveats
- 0.4–0.6 = Partially correct; right direction but misses key points
- 0.0–0.3 = Incorrect or contradicts the correct conclusion

Respond with ONLY a single number, e.g. "0.8"."""


class Judge:
    """Wraps an LLMProvider and scores a (prediction, reference) pair 0.0–1.0.

    Args:
        provider:   a pre-built LLMProvider instance
        sut_model:  the model name of the system-under-test (used only for the guard)

    Raises:
        ValueError: if the judge's model matches sut_model (circularity guard)
    """

    def __init__(self, provider: "LLMProvider", *, sut_model: str) -> None:
        if provider.model == sut_model:
            raise ValueError(
                f"Judge model ({provider.model!r}) is the same model as the "
                f"system-under-test ({sut_model!r}). Use a different model to avoid "
                "self-preference bias."
            )
        self._provider = provider
        self.sut_model = sut_model

    @classmethod
    def from_provider(cls, provider: "LLMProvider", *, sut_model: str) -> "Judge":
        """Construct a Judge from an already-built provider."""
        return cls(provider, sut_model=sut_model)

    @classmethod
    def from_settings(cls, *, sut_model: str) -> "Judge":
        """Build a Judge from settings.judge_provider / settings.judge_model."""
        from adapt_ai.config import settings
        from adapt_ai.llmops.providers import get_provider

        provider = get_provider(
            provider=settings.judge_provider,
            model=settings.judge_model,
            api_key=(
                settings.judge_api_key
                or settings.anthropic_api_key.get_secret_value()
            ),
            base_url=settings.judge_base_url or None,
        )
        return cls(provider, sut_model=sut_model)

    def score(self, *, prediction: str, reference: str, query: str = "") -> float:
        """Return a correctness score in [0.0, 1.0]. Returns 0.5 on parse failure."""
        prompt = _RUBRIC_PROMPT.format(
            reference=reference,
            prediction=prediction[:2000],
        )
        try:
            result = self._provider.complete(
                system="You are an objective evaluator. Follow the scoring rubric exactly.",
                user=prompt,
                max_tokens=16,
                temperature=0.0,
            )
            text = result.text.strip()
            m = re.search(r"[0-9]+(?:\.[0-9]+)?", text)
            if m:
                return min(1.0, max(0.0, float(m.group())))
        except Exception as exc:
            logger.warning("Judge scoring failed: %s", exc)
        return 0.5  # neutral fallback


def build_judge(
    provider: str,
    model: str,
    *,
    sut_model: str,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Judge:
    """Convenience constructor matching the two-string (provider, model) signature."""
    from adapt_ai.llmops.providers import get_provider
    prov = get_provider(provider=provider, model=model, api_key=api_key, base_url=base_url)
    return Judge(prov, sut_model=sut_model)

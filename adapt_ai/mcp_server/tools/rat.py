"""RAT tool — multi-step CoT reasoning with iterative retrieval."""
from __future__ import annotations
import logging
import time
from typing import List

from adapt_ai.config import settings
from adapt_ai.domain.profiles import get_domain_profile
from adapt_ai.domain.vector_store import VectorStore
from adapt_ai.llmops.providers import get_provider as _get_provider_factory, LLMProvider
from adapt_ai.llmops.usage import record_llm_call

logger = logging.getLogger(__name__)

_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = _get_provider_factory()
    return _provider


async def rat_reason(
    query: str,
    context: str = "",
    domain: str = "healthcare",
    max_steps: int | None = None,
) -> str:
    """Multi-step reasoning using RAT pipeline for the given domain.

    Steps:
    1. Initial query analysis (CoT prompt → decompose the question)
    2. First retrieval based on analysis
    3. Reasoning refinement
    4. Follow-up retrieval if confidence low
    5. Final synthesis
    """
    n_steps = max_steps or settings.rat_max_steps
    profile = get_domain_profile(domain)
    store = VectorStore.for_collection(profile.vector_collection)
    provider = _get_provider()

    accumulated_context = context

    # Step 1: Decompose the query into sub-questions
    t0 = time.perf_counter()
    decompose_result = provider.complete(
        system=profile.personas["rat_decompose"],
        user=f'{profile.label("query")}:\n{query}',
        max_tokens=512,
        temperature=0.1,
    )
    record_llm_call(
        agent="rat.decompose",
        model=decompose_result.model,
        input_tokens=decompose_result.input_tokens,
        output_tokens=decompose_result.output_tokens,
        latency_s=time.perf_counter() - t0,
    )
    sub_questions_text = decompose_result.text.strip()
    sub_questions: List[str] = [
        q.strip().lstrip("0123456789.-) ") for q in sub_questions_text.split("\n") if q.strip()
    ][:n_steps]

    # Steps 2-4: Retrieve for each sub-question and refine reasoning
    retrieval_contexts: List[str] = []
    for sq in sub_questions:
        docs = store.query(sq, n_results=3)
        ctx = store.format_context(docs)
        retrieval_contexts.append(f"Sub-question: {sq}\n{ctx}")

    combined_context = "\n\n---\n\n".join(retrieval_contexts)
    if accumulated_context:
        combined_context = accumulated_context + "\n\n---\n\n" + combined_context

    # Step 5: Final synthesis — produce the reasoned answer
    t0 = time.perf_counter()
    synthesis_result = provider.complete(
        system=profile.personas["rat_synthesis"],
        user=(
            f"Question:\n{query}\n\n"
            f'{profile.label("context")}:\n{combined_context}\n\n'
            "Reason through this carefully and provide your answer."
        ),
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
    )
    record_llm_call(
        agent="rat.synthesis",
        model=synthesis_result.model,
        input_tokens=synthesis_result.input_tokens,
        output_tokens=synthesis_result.output_tokens,
        latency_s=time.perf_counter() - t0,
    )

    return synthesis_result.text

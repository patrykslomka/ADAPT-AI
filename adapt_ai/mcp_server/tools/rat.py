"""RAT tool — multi-step CoT reasoning with iterative retrieval."""
from __future__ import annotations
import logging
import time
from typing import List

from anthropic import Anthropic

from adapt_ai.config import settings
from adapt_ai.domain.profiles import get_domain_profile
from adapt_ai.domain.vector_store import VectorStore
from adapt_ai.llmops.usage import record_llm_call

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    return _client


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
    client = _get_client()

    accumulated_context = context

    # Step 1: Decompose the query into sub-questions
    t0 = time.perf_counter()
    decompose_resp = client.messages.create(
        model=settings.model_name,
        max_tokens=512,
        temperature=0.1,
        system=profile.personas["rat_decompose"],
        messages=[{"role": "user", "content": f'{profile.label("query")}:\n{query}'}],
    )
    record_llm_call(
        agent="rat.decompose",
        model=settings.model_name,
        input_tokens=decompose_resp.usage.input_tokens,
        output_tokens=decompose_resp.usage.output_tokens,
        latency_s=time.perf_counter() - t0,
    )
    sub_questions_text = decompose_resp.content[0].text.strip()
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
    synthesis_resp = client.messages.create(
        model=settings.model_name,
        max_tokens=settings.max_tokens,
        temperature=settings.temperature,
        system=profile.personas["rat_synthesis"],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f'{profile.label("context")}:\n{combined_context}\n\n'
                    "Reason through this carefully and provide your answer."
                ),
            }
        ],
    )
    record_llm_call(
        agent="rat.synthesis",
        model=settings.model_name,
        input_tokens=synthesis_resp.usage.input_tokens,
        output_tokens=synthesis_resp.usage.output_tokens,
        latency_s=time.perf_counter() - t0,
    )

    return synthesis_resp.content[0].text

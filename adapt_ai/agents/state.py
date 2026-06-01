"""Typed AgentState — shared across all LangGraph nodes."""
from typing import Annotated, Optional, TypedDict


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer that merges parallel agents' partial dict updates (e.g. agent_statuses)."""
    return {**a, **b}


class AgentState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    subject_id: Optional[str]     # domain entity id (person/case/account)
    session_id: str
    domain: str             # regulated domain key: "healthcare" | "legal" | "finance"

    # ── Routing ───────────────────────────────────────────────────────────────
    use_rat: bool           # True → RAT tool; False → RAG tool
    retrieved_context: str  # output of RAG/RAT tool call

    # ── Agent outputs ─────────────────────────────────────────────────────────
    primary_response: str
    compliance_result: dict     # {"passed": bool, "status": str, "issues": [...]}
    quality_result: dict        # {"passed": bool, "score": float, "issues": [...]}
    final_response: str

    # ── Feedback loop ─────────────────────────────────────────────────────────
    revision_count: int         # max 1 retry
    revision_feedback: str      # feedback injected into primary agent on retry

    # ── Metadata ──────────────────────────────────────────────────────────────
    # Annotated with _merge_dicts so parallel compliance + quality writes are merged.
    agent_statuses: Annotated[dict, _merge_dicts]
    error: Optional[str]
    llm_usage: Optional[dict]   # populated by aggregate_response; see llmops/usage.py

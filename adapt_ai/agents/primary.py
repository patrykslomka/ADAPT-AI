"""Primary Domain Agent — clinical reasoning via MCP tool calls."""
from __future__ import annotations
import logging
import time
from typing import TYPE_CHECKING

from anthropic import Anthropic

from adapt_ai.agents.state import AgentState
from adapt_ai.config import settings
from adapt_ai.llmops.usage import record_llm_call

if TYPE_CHECKING:
    from adapt_ai.orchestrator.client import MCPClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an expert clinical diagnostic assistant supporting healthcare providers.

Your role:
1. Analyse patient presentations and medical history.
2. Generate evidence-based differential diagnoses.
3. Recommend appropriate diagnostic work-ups.
4. Suggest treatment considerations based on established guidelines.

When answering a multiple-choice question (A / B / C / D / E):
- Reason step-by-step through the clinical scenario.
- Eliminate incorrect options explicitly.
- End your response with exactly: ANSWER: X
  (where X is the single letter of the best choice).

You are providing decision support for qualified healthcare providers — not making diagnoses.\
"""


def make_primary_node(mcp_client: "MCPClient", anthropic_client: Anthropic):
    """Return a LangGraph node function for the Primary Domain Agent."""

    async def primary_agent(state: AgentState) -> dict:
        query = state["query"]
        context = state.get("retrieved_context", "")
        feedback = state.get("revision_feedback", "")
        revision = state.get("revision_count", 0)

        user_content = f"Clinical question:\n{query}"
        if context:
            user_content += f"\n\nRetrieved clinical context:\n{context}"
        if feedback:
            user_content += (
                f"\n\n[Quality feedback from previous attempt — please address these issues:]\n{feedback}"
            )
        if revision > 0:
            user_content += "\n\nPlease provide an improved, more accurate response."

        try:
            t0 = time.perf_counter()
            resp = anthropic_client.messages.create(
                model=settings.model_name,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            latency = time.perf_counter() - t0
            record_llm_call(
                agent="primary" if not feedback else "primary_retry",
                model=settings.model_name,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
                latency_s=latency,
                run_id=state["session_id"],
            )
            response_text = resp.content[0].text
            statuses = dict(state.get("agent_statuses", {}))
            statuses["primary"] = "approved"
            # Increment revision_count here (the only place state is safely updated).
            # On a retry call, feedback is non-empty; incrementing signals to
            # route_after_quality that one retry has already occurred.
            return {
                "primary_response": response_text,
                "revision_count": revision + (1 if feedback else 0),
                "agent_statuses": statuses,
                "error": None,
            }
        except Exception as e:
            logger.error("Primary agent error: %s", e)
            return {
                "primary_response": "",
                "revision_count": revision,
                "error": str(e),
                "agent_statuses": {**state.get("agent_statuses", {}), "primary": "error"},
            }

    return primary_agent

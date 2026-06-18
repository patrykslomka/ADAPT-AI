"""Quality Agent - hallucination detection and confidence scoring via Claude."""
from __future__ import annotations
import json
import logging
import re
import time
from typing import TYPE_CHECKING

from adapt_ai.agents.state import AgentState
from adapt_ai.config import settings
from adapt_ai.domain.lexicon import check_lexicon
from adapt_ai.domain.profiles import get_domain_profile
from adapt_ai.llmops.providers import LLMProvider
from adapt_ai.llmops.usage import record_llm_call

if TYPE_CHECKING:
    from adapt_ai.orchestrator.client import MCPClient

logger = logging.getLogger(__name__)


def make_quality_node(mcp_client: "MCPClient", provider: LLMProvider):

    async def quality_agent(state: AgentState) -> dict:
        query = state["query"]
        primary_response = state.get("primary_response", "")
        context = state.get("retrieved_context", "")

        if not primary_response:
            statuses = {**state.get("agent_statuses", {}), "quality": "skipped"}
            return {
                "quality_result": {"passed": True, "score": 0.5, "issues": [], "feedback": ""},
                "agent_statuses": statuses,
            }

        profile = get_domain_profile(state.get("domain"))

        # Pre-LLM lexicon check (cheap regex, catches hallucinated terms).
        warnings = check_lexicon(primary_response, profile.lexicon)

        evaluation_prompt = (
            f"Original question:\n{query}\n\n"
            f'{profile.label("quality_context")}:\n{context[:800] if context else "None"}\n\n'
        )
        if warnings:
            evaluation_prompt += (
                "[Pre-check flags - verify these in the response below:]\n"
                + "\n".join(f"  • {w}" for w in warnings)
                + "\n\n"
            )
        evaluation_prompt += f"AI response to evaluate:\n{primary_response}"

        try:
            t0 = time.perf_counter()
            result = provider.complete(
                system=profile.personas["quality"],
                user=evaluation_prompt,
                max_tokens=512,
                temperature=0.1,
            )
            record_llm_call(
                agent="quality",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_s=time.perf_counter() - t0,
                run_id=state["session_id"],
            )
            raw = result.text.strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                result = json.loads(m.group(0))
            else:
                logger.warning("Quality agent returned non-JSON: %s", raw[:200])
                result = {"passed": False, "score": 0.0, "issues": ["Quality agent returned unparseable output"], "feedback": "Re-evaluate and provide a well-structured response."}
        except Exception as e:
            logger.warning("Quality agent error: %s", e)
            result = {"passed": False, "score": 0.0, "issues": [f"Quality evaluation failed: {e}"], "feedback": "Re-evaluate and provide a well-structured response."}

        status = "approved" if result.get("passed") else "rejected"
        statuses = {**state.get("agent_statuses", {}), "quality": status}
        return {
            "quality_result": result,
            "revision_feedback": result.get("feedback", ""),
            "agent_statuses": statuses,
        }

    return quality_agent

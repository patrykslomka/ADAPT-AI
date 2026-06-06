"""LangGraph StateGraph — ADAPT-AI multi-agent pipeline.

Flow:
    intent_and_retrieve → primary_agent → [compliance_agent ─┐
                                           quality_agent    ─┤] → review_results → aggregate_response
                                                 ↑ (retry if quality rejects, max 1 time)
"""
from __future__ import annotations
import logging
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from adapt_ai.agents.state import AgentState
from adapt_ai.agents.primary import make_primary_node
from adapt_ai.agents.compliance import make_compliance_node
from adapt_ai.agents.quality import make_quality_node
from adapt_ai.domain.profiles import get_domain_profile
from adapt_ai.llmops.providers import LLMProvider
from adapt_ai.llmops.usage import new_accumulator, get_accumulator
from adapt_ai.orchestrator.client import MCPClient
from adapt_ai.orchestrator.router import should_use_rat

logger = logging.getLogger(__name__)


# ── Node: intent detection + retrieval ────────────────────────────────────────

def make_retrieval_node(mcp_client: MCPClient):
    async def intent_and_retrieve(state: AgentState) -> dict:
        new_accumulator(state["session_id"])
        query = state["query"]
        domain = state.get("domain", "healthcare")
        use_rat = should_use_rat(query, domain)
        try:
            if use_rat:
                logger.debug("Routing to RAT for complex query")
                context = await mcp_client.call_tool(
                    "rat_reason_tool", {"query": query, "context": "", "domain": domain}
                )
            else:
                logger.debug("Routing to RAG for simple query")
                context = await mcp_client.call_tool(
                    "rag_retrieve_tool", {"query": query, "n_results": 5, "domain": domain}
                )
        except Exception as e:
            logger.error("Retrieval error: %s", e)
            context = ""
            use_rat = False

        return {
            "use_rat": use_rat,
            "retrieved_context": context,
            "revision_count": 0,
            "revision_feedback": "",
            "agent_statuses": {},
            "llm_usage": None,
            "error": None,
        }
    return intent_and_retrieve


# ── Node: aggregate final response ────────────────────────────────────────────

async def aggregate_response(state: AgentState) -> dict:
    primary = state.get("primary_response", "")
    compliance = state.get("compliance_result", {})
    quality = state.get("quality_result", {})

    acc = get_accumulator(state["session_id"])
    llm_usage = acc.to_dict() if acc is not None else None

    parts = [primary]

    if compliance.get("status") == "warning" and compliance.get("issues"):
        warnings = "\n".join(
            f"- {i.get('description', i)}" for i in compliance["issues"]
        )
        parts.append(f"\n**Compliance Considerations:**\n{warnings}")

    if quality.get("score", 1.0) < 0.7:
        parts.append(
            f"\n**Note:** Response flagged for quality review "
            f"(confidence: {quality.get('score', 0):.0%})"
        )

    profile = get_domain_profile(state.get("domain"))
    if profile.disclaimer:
        parts.append(f"\n---\n{profile.disclaimer}")

    return {"final_response": "\n".join(parts), "llm_usage": llm_usage}


# ── Node: fan-in join after parallel compliance + quality ──────────────────────

async def review_results(state: AgentState) -> dict:
    return {}


# ── Routing function (replaces route_after_compliance + route_after_quality) ──

def route_after_review(
    state: AgentState,
) -> Literal["primary_agent", "aggregate_response", "__end__"]:
    compliance = state.get("compliance_result", {})
    quality = state.get("quality_result", {})
    revision_count = state.get("revision_count", 0)

    if not compliance.get("passed", True):
        logger.warning("Compliance rejected — ending pipeline")
        return END
    if not quality.get("passed", True) and revision_count < 1:
        logger.info("Quality check failed — routing to retry (revision_count=%d)", revision_count)
        return "primary_agent"
    return "aggregate_response"


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(
    mcp_client: MCPClient,
    include_quality: bool = True,
    provider: LLMProvider | None = None,
) -> "CompiledGraph":
    if provider is None:
        from adapt_ai.llmops.providers import get_provider
        provider = get_provider()

    graph = StateGraph(AgentState)

    graph.add_node("intent_and_retrieve", make_retrieval_node(mcp_client))
    graph.add_node("primary_agent", make_primary_node(mcp_client, provider))
    graph.add_node("compliance_agent", make_compliance_node(mcp_client))
    if include_quality:
        graph.add_node("quality_agent", make_quality_node(mcp_client, provider))
    graph.add_node("review_results", review_results)
    graph.add_node("aggregate_response", aggregate_response)

    graph.set_entry_point("intent_and_retrieve")
    graph.add_edge("intent_and_retrieve", "primary_agent")

    graph.add_edge("primary_agent", "compliance_agent")
    if include_quality:
        graph.add_edge("primary_agent", "quality_agent")

    graph.add_edge("compliance_agent", "review_results")
    if include_quality:
        graph.add_edge("quality_agent", "review_results")

    graph.add_conditional_edges(
        "review_results",
        route_after_review,
        {
            "primary_agent": "primary_agent",
            "aggregate_response": "aggregate_response",
            END: END,
        },
    )
    graph.set_finish_point("aggregate_response")

    return graph.compile(checkpointer=MemorySaver())

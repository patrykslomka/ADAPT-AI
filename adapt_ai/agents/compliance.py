"""Compliance Agent - validates content against domain regulations via MCP."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from adapt_ai.agents.state import AgentState

if TYPE_CHECKING:
    from adapt_ai.orchestrator.client import MCPClient

logger = logging.getLogger(__name__)


def make_compliance_node(mcp_client: "MCPClient"):

    async def compliance_agent(state: AgentState) -> dict:
        content = state.get("primary_response", "")
        if not content:
            statuses = {**state.get("agent_statuses", {}), "compliance": "skipped"}
            return {
                "compliance_result": {"passed": True, "status": "skipped", "issues": [], "suggestions": []},
                "agent_statuses": statuses,
            }

        try:
            result = await mcp_client.call_tool_dict(
                "validate_output_tool",
                {"content": content, "domain": state.get("domain", "healthcare")},
            )
        except Exception as e:
            logger.error("Compliance agent MCP error: %s", e)
            result = {"passed": True, "status": "warning", "issues": [str(e)], "suggestions": []}

        status = result.get("status", "approved")
        statuses = {**state.get("agent_statuses", {}), "compliance": status}
        return {
            "compliance_result": result,
            "agent_statuses": statuses,
        }

    return compliance_agent

"""The compliance agent must validate against the domain carried in state,
not a hardcoded value — this is the prerequisite for config-only domain ports."""
import pytest

from adapt_ai.agents.compliance import make_compliance_node
from tests.test_adapt_ai.conftest import FakeMCPClient, make_state


@pytest.mark.asyncio
async def test_compliance_passes_state_domain_to_validation_tool():
    mcp = FakeMCPClient()
    node = make_compliance_node(mcp)

    state = make_state(domain="legal", primary_response="Some legal answer text.")
    await node(state)

    assert len(mcp.dict_calls) == 1
    tool_name, args = mcp.dict_calls[0]
    assert tool_name == "validate_output_tool"
    assert args["domain"] == "legal"


@pytest.mark.asyncio
async def test_compliance_defaults_to_healthcare_when_domain_absent():
    mcp = FakeMCPClient()
    node = make_compliance_node(mcp)

    state = make_state(primary_response="Some clinical answer.")
    del state["domain"]  # legacy callers may omit it
    await node(state)

    _, args = mcp.dict_calls[0]
    assert args["domain"] == "healthcare"

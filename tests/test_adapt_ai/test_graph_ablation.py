"""Ablation switch: build_graph(include_quality=False) must run the pipeline
without the quality agent (no quality status, no quality-driven retry)."""
import pytest

from adapt_ai.agents.graph import build_graph
from tests.test_adapt_ai.conftest import FakeProvider, FakeMCPClient, make_state


async def _run(include_quality: bool) -> dict:
    mcp = FakeMCPClient(context="ctx")
    pipeline = build_graph(mcp, include_quality=include_quality, provider=FakeProvider())
    return await pipeline.ainvoke(
        make_state(session_id="ablation-test"),
        config={"configurable": {"thread_id": "ablation-test"}},
    )


@pytest.mark.asyncio
async def test_pipeline_without_quality_agent_has_no_quality_status():
    result = await _run(include_quality=False)
    assert "quality" not in result.get("agent_statuses", {})
    assert result.get("quality_result", {}) == {}
    # Primary and compliance still run.
    assert "primary" in result["agent_statuses"]
    assert "compliance" in result["agent_statuses"]
    assert result.get("final_response")


@pytest.mark.asyncio
async def test_pipeline_with_quality_agent_records_quality_status():
    result = await _run(include_quality=True)
    assert "quality" in result["agent_statuses"]


def test_disclaimer_can_be_disabled(fake_mcp):
    """aggregate_response must omit the profile disclaimer when append_disclaimer=False."""
    from adapt_ai.agents.graph import build_graph
    from adapt_ai.domain.profiles import get_domain_profile
    import asyncio

    g = build_graph(fake_mcp, include_quality=False, append_disclaimer=False,
                    provider=FakeProvider())
    state = make_state(query="q", domain="healthcare")
    out = asyncio.run(
        g.ainvoke(state, config={"configurable": {"thread_id": "test-ablation-1"}})
    )
    profile = get_domain_profile("healthcare")
    assert profile.disclaimer not in out.get("final_response", "")


def test_compliance_can_be_disabled(fake_mcp):
    """build_graph with include_compliance=False must not include compliance_agent node."""
    from adapt_ai.agents.graph import build_graph

    g = build_graph(fake_mcp, include_compliance=False, provider=FakeProvider())
    node_names = set(g.get_graph().nodes.keys())
    assert "compliance_agent" not in node_names

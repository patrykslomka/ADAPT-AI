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

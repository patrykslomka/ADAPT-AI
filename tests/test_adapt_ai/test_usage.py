"""Regression tests for per-run LLM usage/cost tracking (adapt_ai/llmops/usage.py)."""
from adapt_ai.llmops.usage import (
    new_accumulator,
    get_accumulator,
    record_llm_call,
)


def test_accumulator_aggregates_cost_across_agents():
    """Cost and tokens sum across all recorded calls for a run_id."""
    new_accumulator("run-a")
    record_llm_call("primary", "claude-haiku-4-5-20251001", 1000, 500, 1.2, run_id="run-a")
    record_llm_call("rat.synthesis", "claude-haiku-4-5-20251001", 800, 300, 0.9, run_id="run-a")

    acc = get_accumulator("run-a")
    assert acc is not None
    assert len(acc.calls) == 2
    assert acc.total_input_tokens == 1800
    assert acc.total_output_tokens == 800
    assert acc.total_cost_usd > 0.0

    d = acc.to_dict()
    assert d["total_input_tokens"] == 1800
    assert d["total_cost_usd"] == acc.total_cost_usd
    assert {c["agent"] for c in d["calls"]} == {"primary", "rat.synthesis"}


def test_accumulators_isolated_by_run_id():
    """Two runs do not bleed into each other."""
    new_accumulator("run-x")
    new_accumulator("run-y")
    record_llm_call("primary", "claude-haiku-4-5-20251001", 100, 100, run_id="run-x")

    assert len(get_accumulator("run-x").calls) == 1
    assert len(get_accumulator("run-y").calls) == 0


def test_record_without_active_accumulator_is_noop():
    """An unknown run_id must not raise."""
    record_llm_call("primary", "claude-haiku-4-5-20251001", 10, 10, run_id="does-not-exist")
    assert get_accumulator("does-not-exist") is None

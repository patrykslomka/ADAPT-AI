# ADAPT-AI Measurement Harness & Integrity Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the first automated test suite for `adapt_ai/`, add the quality-agent ablation switch, thread `domain` through agent state (prerequisite for config-only multi-domain ports), and fix the resume-skips-errored-entries bug in both benchmark scripts.

**Architecture:** All work is pure code + unit/behavioural tests with a **mocked Anthropic client and a fake in-process MCP client** — no live API calls, so this plan runs even while the Anthropic usage cap is active (resets 2026-06-01). It is the dependency root for the legal/finance domain plans (Plans 2–3) and the cross-domain analysis (Plan 4).

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio (1.3.0, strict mode — async tests need `@pytest.mark.asyncio`), LangGraph, Anthropic SDK, FastMCP.

**Verified preconditions (2026-05-30, against current code):**
- Cost tracking **already works** (`adapt_ai/llmops/usage.py`; 29/30 clinical results carry non-null `total_cost_usd`). Task 2 locks it with a regression test — it does **not** implement it.
- MedQA Q27–44 nulls were an **API usage-cap abort**, not a parse bug. The real defect is resume logic treating errored entries as done (Task 5).
- `compliance.py:29` hardcodes `domain="healthcare"` → blocks config-only ports (Task 3).
- `adapt_ai/` currently has **zero tests**. Task 1 bootstraps the suite.

---

## File Structure

**Create:**
- `tests/test_adapt_ai/__init__.py` — package marker.
- `tests/test_adapt_ai/conftest.py` — shared fakes: `FakeAnthropic`, `FakeMCPClient`, state helper.
- `tests/test_adapt_ai/test_usage.py` — cost-accumulator regression tests (pure).
- `tests/test_adapt_ai/test_compliance_domain.py` — domain-threading test.
- `tests/test_adapt_ai/test_graph_ablation.py` — quality-agent ablation behavioural test.
- `tests/test_adapt_ai/test_benchmark_resume.py` — resume helper unit test.

**Modify:**
- `adapt_ai/agents/state.py` — add `domain` field to `AgentState`.
- `adapt_ai/agents/compliance.py` — read `state["domain"]` instead of hardcoded `"healthcare"`.
- `adapt_ai/agents/graph.py` — add `include_quality: bool = True` to `build_graph()`; conditional wiring.
- `scripts/run_clinical_benchmark.py` — `_completed_ids()` helper + `--no-quality` flag + set `domain` in initial state.
- `scripts/run_medqa_benchmark.py` — `_completed_ids()` helper + `--no-quality` flag + set `domain` in initial state.

Each file has one responsibility; tests mirror the module they cover.

---

## Task 1: Bootstrap the `adapt_ai/` test package and shared fakes

**Files:**
- Create: `tests/test_adapt_ai/__init__.py`
- Create: `tests/test_adapt_ai/conftest.py`

- [ ] **Step 1: Create the package marker**

Create `tests/test_adapt_ai/__init__.py` with a single line:

```python
"""Tests for the adapt_ai/ rebuilt architecture (LangGraph + FastMCP)."""
```

- [ ] **Step 2: Write the shared fakes in conftest**

Create `tests/test_adapt_ai/conftest.py`:

```python
"""Shared fakes for adapt_ai tests — no live Anthropic or MCP calls."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeMessages:
    """Records calls; returns a canned Anthropic-shaped response."""

    def __init__(self, text: str, in_tok: int = 100, out_tok: int = 50) -> None:
        self._text = text
        self._in = in_tok
        self._out = out_tok
        self.calls: list[dict] = []

    def create(self, **kwargs: Any):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(text=self._text)],
            usage=SimpleNamespace(input_tokens=self._in, output_tokens=self._out),
        )


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic — never touches the network."""

    def __init__(self, *args: Any, text: str = "ANSWER: A", **kwargs: Any) -> None:
        self.messages = _FakeMessages(text)


class FakeMCPClient:
    """Stand-in for orchestrator.client.MCPClient with recorded calls."""

    def __init__(self, context: str = "", validation: dict | None = None) -> None:
        self._context = context
        self._validation = validation or {
            "passed": True, "status": "approved", "issues": [], "suggestions": [],
        }
        self.tool_calls: list[tuple[str, dict]] = []
        self.dict_calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.tool_calls.append((name, arguments))
        return self._context

    async def read_resource(self, uri: str) -> str:
        return ""

    async def call_tool_dict(self, name: str, arguments: dict) -> dict:
        self.dict_calls.append((name, arguments))
        return self._validation


@pytest.fixture
def fake_mcp() -> FakeMCPClient:
    return FakeMCPClient()


def make_state(query: str = "What is the first-line treatment for hypertension?",
               session_id: str = "test-1", **overrides: Any) -> dict:
    """Build a complete initial AgentState dict for pipeline tests."""
    state = {
        "query": query,
        "patient_id": None,
        "session_id": session_id,
        "domain": "healthcare",
        "use_rat": False,
        "retrieved_context": "",
        "primary_response": "",
        "compliance_result": {},
        "quality_result": {},
        "final_response": "",
        "revision_count": 0,
        "revision_feedback": "",
        "agent_statuses": {},
        "llm_usage": None,
        "error": None,
    }
    state.update(overrides)
    return state
```

- [ ] **Step 3: Verify the fakes import cleanly**

Run: `pytest tests/test_adapt_ai/ -v`
Expected: `no tests ran` (collection succeeds, no import errors).

- [ ] **Step 4: Commit**

```bash
git add tests/test_adapt_ai/__init__.py tests/test_adapt_ai/conftest.py
git commit -m "test: bootstrap adapt_ai test package with shared fakes"
```

---

## Task 2: Cost-tracking regression tests (lock the working feature)

**Files:**
- Create: `tests/test_adapt_ai/test_usage.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_adapt_ai/test_usage.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass (this is a regression lock, not new code)**

Run: `pytest tests/test_adapt_ai/test_usage.py -v`
Expected: PASS (3 passed). The feature already exists; these tests pin it.

> If any test FAILS, the usage module regressed since 2026-05-30 — stop and investigate before continuing; do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_adapt_ai/test_usage.py
git commit -m "test: lock per-run cost/usage aggregation behaviour"
```

---

## Task 3: Thread `domain` through agent state (unblock config-only ports)

**Files:**
- Modify: `adapt_ai/agents/state.py`
- Modify: `adapt_ai/agents/compliance.py:17-40`
- Test: `tests/test_adapt_ai/test_compliance_domain.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapt_ai/test_compliance_domain.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_ai/test_compliance_domain.py -v`
Expected: FAIL — `test_compliance_passes_state_domain_to_validation_tool` asserts `legal` but the agent sends hardcoded `healthcare`.

- [ ] **Step 3: Add the `domain` field to AgentState**

In `adapt_ai/agents/state.py`, inside the `AgentState` `TypedDict`, add `domain` to the Input block (after `session_id`):

```python
    query: str
    patient_id: Optional[str]
    session_id: str
    domain: str             # regulated domain key: "healthcare" | "legal" | "finance"
```

- [ ] **Step 4: Read the domain from state in the compliance agent**

In `adapt_ai/agents/compliance.py`, replace the `call_tool_dict` block (lines 27-30):

```python
            result = await mcp_client.call_tool_dict(
                "validate_output_tool",
                {"content": content, "domain": state.get("domain", "healthcare")},
            )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_adapt_ai/test_compliance_domain.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
git add adapt_ai/agents/state.py adapt_ai/agents/compliance.py tests/test_adapt_ai/test_compliance_domain.py
git commit -m "feat: thread domain through agent state for config-only domain ports"
```

---

## Task 4: Quality-agent ablation switch in `build_graph`

**Files:**
- Modify: `adapt_ai/agents/graph.py:125-161`
- Test: `tests/test_adapt_ai/test_graph_ablation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapt_ai/test_graph_ablation.py`:

```python
"""Ablation switch: build_graph(include_quality=False) must run the pipeline
without the quality agent (no quality status, no quality-driven retry)."""
from unittest.mock import patch

import pytest

from tests.test_adapt_ai.conftest import FakeAnthropic, FakeMCPClient, make_state


async def _run(include_quality: bool) -> dict:
    # Patch the Anthropic class constructed inside build_graph.
    with patch("adapt_ai.agents.graph.Anthropic", FakeAnthropic):
        from adapt_ai.agents.graph import build_graph
        mcp = FakeMCPClient(context="ctx")
        pipeline = build_graph(mcp, include_quality=include_quality)
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
```

> Note: `FakeAnthropic` returns valid JSON only for clinical text, so the quality node may mark `passed=False` and trigger one retry — that is fine; the assertion only checks that a quality status is recorded.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_ai/test_graph_ablation.py -v`
Expected: FAIL — `build_graph()` does not accept `include_quality` (TypeError).

- [ ] **Step 3: Add the ablation switch to build_graph**

In `adapt_ai/agents/graph.py`, change the signature and node/edge wiring. Replace `def build_graph(mcp_client: MCPClient) -> "CompiledGraph":` with:

```python
def build_graph(mcp_client: MCPClient, include_quality: bool = True) -> "CompiledGraph":
```

Then make node registration conditional — replace the `graph.add_node("quality_agent", ...)` line with:

```python
    if include_quality:
        graph.add_node("quality_agent", make_quality_node(mcp_client, anthropic_client))
```

And make the fan-out / fan-in edges conditional. Replace the two blocks:

```python
    # Fan-out: compliance and quality run in parallel after primary_agent
    graph.add_edge("primary_agent", "compliance_agent")
    graph.add_edge("primary_agent", "quality_agent")

    # Fan-in: both agents join at review_results
    graph.add_edge("compliance_agent", "review_results")
    graph.add_edge("quality_agent", "review_results")
```

with:

```python
    # Fan-out: compliance (and quality, if enabled) run after primary_agent
    graph.add_edge("primary_agent", "compliance_agent")
    if include_quality:
        graph.add_edge("primary_agent", "quality_agent")

    # Fan-in into review_results
    graph.add_edge("compliance_agent", "review_results")
    if include_quality:
        graph.add_edge("quality_agent", "review_results")
```

> `route_after_review` and `aggregate_response` already default quality to "pass" (`quality.get("passed", True)`, `quality.get("score", 1.0)`), so no further changes are needed when quality is absent.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_adapt_ai/test_graph_ablation.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Run the full adapt_ai suite to confirm no regressions**

Run: `pytest tests/test_adapt_ai/ -v`
Expected: PASS (all tests so far).

- [ ] **Step 6: Commit**

```bash
git add adapt_ai/agents/graph.py tests/test_adapt_ai/test_graph_ablation.py
git commit -m "feat: add include_quality ablation switch to build_graph"
```

---

## Task 5: Fix resume-skips-errored-entries + wire `--no-quality` and `domain`

**Files:**
- Modify: `scripts/run_clinical_benchmark.py` (resume block ~200-206, `build_graph` call ~197, initial state ~92-107, argparse ~298-303)
- Modify: `scripts/run_medqa_benchmark.py` (resume block ~200-205, `build_graph` call ~196, initial state ~98-113, argparse ~270-273)
- Test: `tests/test_adapt_ai/test_benchmark_resume.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapt_ai/test_benchmark_resume.py`:

```python
"""Resume must re-run entries that errored; only fully-successful entries count as done."""
from scripts.run_clinical_benchmark import _completed_ids as clinical_completed
from scripts.run_medqa_benchmark import _completed_ids as medqa_completed


def _results():
    return [
        {"id": 0, "adapt_ai": {"error": None}, "baseline": {"error": None}},
        {"id": 1, "adapt_ai": {"error": "usage limit"}, "baseline": {"error": "usage limit"}},
        {"id": 2, "adapt_ai": {"error": None}, "baseline": {"error": "boom"}},
    ]


def test_clinical_completed_ids_excludes_errored():
    assert clinical_completed(_results()) == {0}


def test_medqa_completed_ids_excludes_errored():
    assert medqa_completed(_results()) == {0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_adapt_ai/test_benchmark_resume.py -v`
Expected: FAIL — `_completed_ids` does not exist (ImportError).

- [ ] **Step 3: Add `_completed_ids` to the clinical script**

In `scripts/run_clinical_benchmark.py`, add this helper above `benchmark(...)` (after `score_with_evaluator`):

```python
def _completed_ids(results: list[dict]) -> set:
    """IDs that finished WITHOUT error on either pipeline. Errored entries are
    intentionally excluded so --resume re-runs them (e.g. after a usage-cap abort)."""
    done = set()
    for r in results:
        if r.get("adapt_ai", {}).get("error") or r.get("baseline", {}).get("error"):
            continue
        done.add(r["id"])
    return done
```

- [ ] **Step 4: Use the helper in the clinical resume block**

In `scripts/run_clinical_benchmark.py`, replace:

```python
    done_ids = {r["id"] for r in results}
    todo = [q for q in items if q["id"] not in done_ids]
```

with:

```python
    done_ids = _completed_ids(results)
    # Drop errored entries so they are re-run and re-appended cleanly.
    results = [r for r in results if r["id"] in done_ids]
    todo = [q for q in items if q["id"] not in done_ids]
```

- [ ] **Step 5: Add `_completed_ids` to the MedQA script**

In `scripts/run_medqa_benchmark.py`, add the identical helper above `benchmark(...)` (after `run_baseline`):

```python
def _completed_ids(results: list[dict]) -> set:
    """IDs that finished WITHOUT error on either pipeline. Errored entries are
    intentionally excluded so --resume re-runs them (e.g. after a usage-cap abort)."""
    done = set()
    for r in results:
        if r.get("adapt_ai", {}).get("error") or r.get("baseline", {}).get("error"):
            continue
        done.add(r["id"])
    return done
```

- [ ] **Step 6: Use the helper in the MedQA resume block**

In `scripts/run_medqa_benchmark.py`, replace:

```python
    done_ids = {r["id"] for r in results}
    todo = [q for q in questions if q["id"] not in done_ids]
```

with:

```python
    done_ids = _completed_ids(results)
    results = [r for r in results if r["id"] in done_ids]
    todo = [q for q in questions if q["id"] not in done_ids]
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_adapt_ai/test_benchmark_resume.py -v`
Expected: PASS (2 passed).

- [ ] **Step 8: Wire `--no-quality` and `domain` into the clinical script**

In `scripts/run_clinical_benchmark.py`:

(a) Pass `include_quality` when building the graph — replace `pipeline = build_graph(mcp_client)` with:

```python
    pipeline = build_graph(mcp_client, include_quality=include_quality)
```

(b) Thread the flag through `benchmark(...)` — change its signature to add `include_quality: bool` and the `--no-quality` argument in `main()`:

```python
    parser.add_argument("--no-quality", action="store_true",
                        help="Ablation: run without the quality agent")
```

and in the `asyncio.run(benchmark(...))` call add `include_quality=not args.no_quality,`. Update the `async def benchmark(...)` signature to accept `include_quality: bool = True`.

(c) Add `"domain": "healthcare"` to the initial-state dict inside `run_adapt_ai` (alongside `"patient_id": None`):

```python
                "patient_id": None,
                "domain": "healthcare",
```

- [ ] **Step 9: Wire `--no-quality` and `domain` into the MedQA script**

Repeat Step 8 for `scripts/run_medqa_benchmark.py`:
- `pipeline = build_graph(mcp_client, include_quality=include_quality)`
- add `--no-quality` arg in `main()`, thread `include_quality` through `benchmark(questions, resume, include_quality)`,
- add `"domain": "healthcare",` after `"patient_id": None,` in the `run_adapt_ai` initial state.

- [ ] **Step 10: Verify both scripts still import and parse args**

Run: `python scripts/run_clinical_benchmark.py --help && python scripts/run_medqa_benchmark.py --help`
Expected: both print usage including `--no-quality`; no import errors.

- [ ] **Step 11: Run the full adapt_ai suite**

Run: `pytest tests/test_adapt_ai/ -v`
Expected: PASS (all tests).

- [ ] **Step 12: Commit**

```bash
git add scripts/run_clinical_benchmark.py scripts/run_medqa_benchmark.py tests/test_adapt_ai/test_benchmark_resume.py
git commit -m "fix: resume re-runs errored entries; add --no-quality ablation + domain to benchmarks"
```

---

## Self-Review

**1. Spec coverage (against `2026-05-30-adapt-ai-domain-adaptive-reframe-design.md`):**
- §5.1 cost tracking → verified already working; locked by Task 2. ✅
- §5.2 MedQA null bug → root cause was usage-cap abort + resume skipping errors; fixed in Task 5. ✅ (re-running the 18 questions needs live API after 2026-06-01 — out of this plan's scope, it is a run, not code.)
- §4 quality-agent ablation → switch added (Task 4) + `--no-quality` flag (Task 5). Running the ablation across domains is Plan 4. ✅
- §3 config-only ports → `domain` threading (Task 3) removes the hardcoded `healthcare` blocker. Full legal/finance config stacks are Plans 2–3. ✅
- §5.3 statistical power / effect sizes → analyze script already has Wilcoxon + CIs; effect sizes + multi-domain generalisation deferred to Plan 4 (noted, not silently dropped).

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to". Every code step shows real code. ✅

**3. Type/name consistency:** `_completed_ids(results) -> set` identical in both scripts and both tests. `include_quality: bool` consistent across `build_graph`, both `benchmark()` signatures, and `--no-quality`. `domain` key consistent across `AgentState`, `make_state`, compliance agent, and both initial states. ✅

---

## Notes for downstream plans (not this plan)
- **Plan 2 (legal):** the `validate_output` tool loads regulations by domain; a `legal` regulations JSON + ontology + corpus + uniform-schema benchmark are required before `domain="legal"` produces meaningful compliance output.
- **Bar-3 watch:** after Task 3, audit primary/quality/RAT prompts and the RAT tool for any remaining healthcare-specific hardcoding before claiming "0 agent-code lines changed."
- **Re-run MedQA Q27–44** with `--resume` once the API cap resets (2026-06-01).

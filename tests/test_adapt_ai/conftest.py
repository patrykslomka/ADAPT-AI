"""Shared fakes for adapt_ai tests — no live Anthropic or MCP calls."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any

import pytest

from adapt_ai.llmops.providers import CompletionResult, LLMProvider


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


class FakeProvider(LLMProvider):
    """Stand-in for any LLMProvider — returns canned text, no network."""

    def __init__(self, text: str = "ANSWER: A", in_tok: int = 100, out_tok: int = 50) -> None:
        self.model = "fake-model"
        self._text = text
        self._in = in_tok
        self._out = out_tok
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, max_tokens: int,
                 temperature: float) -> CompletionResult:
        self.calls.append({"system": system, "user": user,
                           "max_tokens": max_tokens, "temperature": temperature})
        return CompletionResult(
            text=self._text,
            input_tokens=self._in,
            output_tokens=self._out,
            model=self.model,
        )


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()


def make_state(query: str = "What is the first-line treatment for hypertension?",
               session_id: str = "test-1", **overrides: Any) -> dict:
    """Build a complete initial AgentState dict for pipeline tests."""
    state = {
        "query": query,
        "subject_id": None,
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

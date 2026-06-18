"""Per-request LLM usage tracking via ContextVar.

Each pipeline run calls new_accumulator() in the first node (intent_and_retrieve).
All subsequent nodes and MCP tools call record_llm_call() to register their token
counts. aggregate_response() drains the accumulator into AgentState["llm_usage"].

ContextVar propagates through await chains and asyncio.create_task() copies,
so it works across all nodes and the FastMCP tool boundary.
"""
from __future__ import annotations
import contextvars
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# USD per token, keyed by model name prefix
_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5":  (0.80 / 1_000_000,  4.00 / 1_000_000),
    "claude-haiku-3-5":  (0.80 / 1_000_000,  4.00 / 1_000_000),
    "claude-sonnet-4-6": (3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-opus-4-7":   (15.00 / 1_000_000, 75.00 / 1_000_000),
}
_FALLBACK_PRICES = (1.00 / 1_000_000, 5.00 / 1_000_000)


def _price_for(model: str) -> tuple[float, float]:
    for prefix, prices in _PRICES.items():
        if model.startswith(prefix):
            return prices
    logger.warning("Unknown model %r - using conservative fallback pricing", model)
    return _FALLBACK_PRICES


@dataclass
class LLMCall:
    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float


@dataclass
class UsageAccumulator:
    calls: list[LLMCall] = field(default_factory=list)

    def record(
        self,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_s: float = 0.0,
    ) -> LLMCall:
        in_price, out_price = _price_for(model)
        cost = input_tokens * in_price + output_tokens * out_price
        call = LLMCall(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=round(latency_s, 3),
            cost_usd=round(cost, 8),
        )
        self.calls.append(call)
        logger.info(
            "llm_call agent=%-16s model=%s in_tok=%d out_tok=%d cost=$%.6f latency=%.2fs",
            agent, model, input_tokens, output_tokens, cost, latency_s,
        )
        return call

    @property
    def total_input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.calls)

    @property
    def total_output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 8)

    def to_dict(self) -> dict:
        return {
            "calls": [
                {
                    "agent": c.agent,
                    "model": c.model,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "latency_s": c.latency_s,
                    "cost_usd": c.cost_usd,
                }
                for c in self.calls
            ],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


# LangGraph runs each node in an isolated asyncio task context, so ContextVar
# changes in one node are not visible in subsequent nodes. A module-level dict
# keyed by session/run ID solves this without changing the graph topology.
_ACCUMULATORS: dict[str, "UsageAccumulator"] = {}

# Kept for single-context usage (unit tests, non-LangGraph callers).
_CURRENT: contextvars.ContextVar[Optional[UsageAccumulator]] = contextvars.ContextVar(
    "_adapt_ai_usage", default=None
)


def new_accumulator(run_id: str) -> UsageAccumulator:
    """Create a fresh accumulator for a pipeline run identified by run_id."""
    acc = UsageAccumulator()
    _ACCUMULATORS[run_id] = acc
    _CURRENT.set(acc)
    return acc


def get_accumulator(run_id: str = "") -> Optional[UsageAccumulator]:
    """Retrieve the accumulator for run_id, falling back to the ContextVar."""
    if run_id:
        return _ACCUMULATORS.get(run_id)
    return _CURRENT.get()


def record_llm_call(
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_s: float = 0.0,
    run_id: str = "",
) -> None:
    """Record one LLM call in the accumulator for run_id. No-op if none is set."""
    acc = get_accumulator(run_id)
    if acc is not None:
        acc.record(agent, model, input_tokens, output_tokens, latency_s)
    else:
        logger.debug("record_llm_call: no active accumulator (agent=%s)", agent)

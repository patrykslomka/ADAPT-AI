#!/usr/bin/env python3
"""MedQA benchmark: ADAPT-AI multi-agent vs monolithic baseline.

Runs 100 USMLE 5-option questions through two pipelines and records
accuracy, response time, and cost.

Usage:
    python scripts/run_medqa_benchmark.py
    python scripts/run_medqa_benchmark.py --questions 10   # quick smoke-test
    python scripts/run_medqa_benchmark.py --resume         # skip already-done questions
"""
import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from anthropic import Anthropic
from adapt_ai.config import settings
from adapt_ai.orchestrator.client import build_mcp_client
from adapt_ai.agents.graph import build_graph
from adapt_ai.llmops.tracing import setup_tracing

setup_tracing()

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
SAMPLE_PATH = DATA_DIR / "medqa_sample.json"
RESULTS_PATH = DATA_DIR / "medqa_results.json"

# ── pricing (claude-haiku-4-5-20251001, per token) ───────────────────────────
HAIKU_INPUT_PRICE = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

SLEEP_BETWEEN_QUESTIONS = 1  # seconds (reduced — new pipeline is faster)

# Substrings in error messages that mean "stop now, not a transient failure"
_RATE_LIMIT_SIGNALS = (
    "usage limits",
    "rate limit",
    "rate_limit_error",
    "overloaded",
    "529",
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RateLimitAbort(Exception):
    """Raised when a hard usage/rate-limit error is detected mid-benchmark."""


def format_question(q: dict) -> str:
    opts = "\n".join(f"  {k}. {v}" for k, v in q["options"].items())
    return (
        f"Clinical Question:\n{q['question']}\n\n"
        f"Answer choices:\n{opts}\n\n"
        "Which answer choice is correct? Think step by step, then end with: ANSWER: X"
    )


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * HAIKU_INPUT_PRICE + output_tokens * HAIKU_OUTPUT_PRICE


def extract_letter(text: str) -> str | None:
    """Extract answer letter — prefers explicit 'ANSWER: X' pattern."""
    m = re.search(r"ANSWER\s*:\s*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()
    # Fallback: last standalone letter A-E
    letters = re.findall(r"\b([A-Ea-e])\b", text)
    return letters[-1].upper() if letters else None


# ── pipeline 1: ADAPT-AI (new LangGraph architecture) ────────────────────────

def _is_rate_limit(err: str) -> bool:
    err_lower = err.lower()
    return any(signal in err_lower for signal in _RATE_LIMIT_SIGNALS)


async def run_adapt_ai(pipeline, formatted_q: str, options: dict, q_id: int) -> dict:
    t0 = time.perf_counter()
    try:
        result = await pipeline.ainvoke(
            {
                "query": formatted_q,
                "patient_id": None,
                "domain": "healthcare",
                "session_id": f"benchmark-{q_id}",
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
            },
            config={"configurable": {"thread_id": f"bench-{q_id}"}},
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        err_str = str(e)
        logger.error("ADAPT-AI pipeline error q%d: %s", q_id, err_str)
        if _is_rate_limit(err_str):
            raise RateLimitAbort(err_str) from e
        return {"letter": None, "time": round(elapsed, 3), "cost": None, "error": err_str}

    elapsed = time.perf_counter() - t0

    pipeline_err = result.get("error")
    if pipeline_err:
        if _is_rate_limit(str(pipeline_err)):
            raise RateLimitAbort(str(pipeline_err))
        return {"letter": None, "time": round(elapsed, 3), "cost": None, "error": pipeline_err}

    response_text = result.get("final_response") or result.get("primary_response", "")
    letter = extract_letter(response_text)

    usage = result.get("llm_usage") or {}
    return {
        "letter": letter,
        "correct": None,  # filled in by caller
        "time": round(elapsed, 3),
        "total_cost_usd": usage.get("total_cost_usd"),
        "total_input_tokens": usage.get("total_input_tokens"),
        "total_output_tokens": usage.get("total_output_tokens"),
        "pipeline_time": round(elapsed, 3),
        "orchestrator_status": "success",
        "use_rat": result.get("use_rat", False),
        "revision_count": result.get("revision_count", 0),
        "agent_statuses": result.get("agent_statuses", {}),
        "error": None,
    }


# ── pipeline 2: monolithic baseline ──────────────────────────────────────────

def run_baseline(client: Anthropic, formatted_q: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=settings.model_name,
            max_tokens=1024,
            temperature=0.3,
            system="You are a clinical expert. Reason through the question carefully, then output your final answer.",
            messages=[
                {
                    "role": "user",
                    "content": formatted_q,
                }
            ],
        )
    except Exception as e:
        err_str = str(e)
        if _is_rate_limit(err_str):
            raise RateLimitAbort(err_str) from e
        return {"letter": None, "time": round(time.perf_counter() - t0, 3), "cost": None, "error": err_str}

    elapsed = time.perf_counter() - t0
    text = resp.content[0].text
    letter = extract_letter(text)
    cost = compute_cost(resp.usage.input_tokens, resp.usage.output_tokens)

    return {
        "letter": letter,
        "time": round(elapsed, 3),
        "cost": round(cost, 6),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def _completed_ids(results: list[dict]) -> set:
    """IDs that finished WITHOUT error on either pipeline. Errored entries are
    intentionally excluded so --resume re-runs them (e.g. after a usage-cap abort)."""
    done = set()
    for r in results:
        if r.get("adapt_ai", {}).get("error") or r.get("baseline", {}).get("error"):
            continue
        done.add(r["id"])
    return done


# ── main loop ─────────────────────────────────────────────────────────────────

async def benchmark(questions: list[dict], resume: bool, include_quality: bool = True) -> None:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    print("Initialising ADAPT-AI pipeline (LangGraph + MCP)…")
    mcp_client = build_mcp_client()
    pipeline = build_graph(mcp_client, include_quality=include_quality)
    print("Pipeline ready.\n")

    results: list[dict] = []
    if resume and RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Resuming from {len(results)} completed questions.")

    done_ids = _completed_ids(results)
    results = [r for r in results if r["id"] in done_ids]
    todo = [q for q in questions if q["id"] not in done_ids]

    total = len(questions)

    try:
        for q in todo:
            q_id = q["id"]
            correct = q["answer"]
            formatted_q = format_question(q)

            print(f"Q{q_id + 1}/{total} ", end="", flush=True)

            adapt_result = await run_adapt_ai(pipeline, formatted_q, q["options"], q_id)
            adapt_result["correct"] = adapt_result["letter"] == correct if adapt_result["letter"] else False

            base_result = run_baseline(client, formatted_q)
            base_result["correct"] = base_result["letter"] == correct if base_result["letter"] else False

            adapt_letter = adapt_result.get("letter", "?") or "?"
            base_letter = base_result.get("letter", "?") or "?"
            adapt_ok = "✓" if adapt_result["correct"] else "✗"
            base_ok = "✓" if base_result["correct"] else "✗"
            cost_str = f"  ${adapt_result['total_cost_usd']:.4f}" if adapt_result.get("total_cost_usd") else ""
            print(
                f"ADAPT={adapt_letter}{adapt_ok} Base={base_letter}{base_ok} "
                f"({adapt_result['time']:.1f}s / {base_result['time']:.1f}s){cost_str}"
            )

            results.append({
                "id": q_id,
                "question": q["question"][:120],
                "correct": correct,
                "adapt_ai": adapt_result,
                "baseline": base_result,
            })

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

            if q_id < total - 1:
                time.sleep(SLEEP_BETWEEN_QUESTIONS)

    except RateLimitAbort as e:
        print(f"\n\n⚠  Rate / usage limit hit — stopping early. Progress saved to {RESULTS_PATH}")
        print(f"   Error: {e}")
        print("   Re-run with --resume once the limit resets.")
        _print_summary(results)
        return

    print(f"\nDone. Results saved to {RESULTS_PATH}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    adapt_correct = sum(1 for r in results if r["adapt_ai"].get("correct"))
    base_correct = sum(1 for r in results if r["baseline"].get("correct"))
    n = len(results)
    print(f"\n{'='*50}")
    print(f"Results over {n} questions:")
    print(f"  ADAPT-AI : {adapt_correct}/{n} = {adapt_correct/n*100:.1f}%")
    print(f"  Baseline : {base_correct}/{n} = {base_correct/n*100:.1f}%")
    print(f"  Delta    : {(adapt_correct - base_correct)/n*100:+.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-quality", action="store_true",
                        help="Ablation: run without the quality agent")
    args = parser.parse_args()

    if not SAMPLE_PATH.exists():
        print(f"Sample not found at {SAMPLE_PATH}. Run: python scripts/download_medqa.py")
        sys.exit(1)

    questions = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    if args.questions:
        questions = questions[: args.questions]

    asyncio.run(benchmark(questions, resume=args.resume, include_quality=not args.no_quality))


if __name__ == "__main__":
    main()

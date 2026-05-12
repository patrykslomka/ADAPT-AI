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
from config.settings import settings
from src.mcp.orchestrator import MCPOrchestrator

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
SAMPLE_PATH = DATA_DIR / "medqa_sample.json"
RESULTS_PATH = DATA_DIR / "medqa_results.json"

# ── pricing (claude-haiku-4-5-20251001, per token) ───────────────────────────
HAIKU_INPUT_PRICE = 0.80 / 1_000_000   # $ per input token
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000  # $ per output token

SLEEP_BETWEEN_QUESTIONS = 2  # seconds

logging.basicConfig(
    level=logging.WARNING,  # suppress agent-level chatter during benchmark
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def format_question(q: dict) -> str:
    """Format a MedQA question dict into a clinical vignette string."""
    opts = "\n".join(f"  {k}. {v}" for k, v in q["options"].items())
    return (
        f"Clinical Question:\n{q['question']}\n\n"
        f"Answer choices:\n{opts}\n\n"
        "Which answer choice is correct?"
    )


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * HAIKU_INPUT_PRICE + output_tokens * HAIKU_OUTPUT_PRICE


def extract_letter_regex(text: str) -> str | None:
    """Extract answer letter from 'ANSWER: X' pattern with regex fallback."""
    m = re.search(r"ANSWER\s*:\s*([A-Ea-e])", text)
    if m:
        return m.group(1).upper()
    # Fallback: last standalone letter A-E in the text
    letters = re.findall(r"\b([A-Ea-e])\b", text)
    return letters[-1].upper() if letters else None


# ── pipeline 1: ADAPT-AI ──────────────────────────────────────────────────────

async def run_adapt_ai(
    orchestrator: MCPOrchestrator,
    client: Anthropic,
    formatted_q: str,
    options: dict,
) -> dict:
    """Run question through ADAPT-AI, then extract a letter with a second LLM call."""
    t0 = time.perf_counter()
    try:
        result = await orchestrator.process_query(query=formatted_q, patient_id=None)
    except Exception as e:
        logger.error("ADAPT-AI orchestrator error: %s", e)
        return {"letter": None, "time": time.perf_counter() - t0, "cost": None, "error": str(e)}

    pipeline_time = time.perf_counter() - t0

    if result.get("status") != "success":
        return {
            "letter": None,
            "time": pipeline_time,
            "cost": None,
            "status": result.get("status"),
        }

    content = result["content"]

    # Letter extractor: give the model the actual answer choices so it can
    # match the clinical reasoning to the correct option rather than guessing.
    options_text = "\n".join(f"  {k}. {v}" for k, v in options.items())
    try:
        extract_resp = client.messages.create(
            model=settings.model_name,
            max_tokens=10,
            temperature=0.0,
            system="You are a medical exam answer extractor. Given clinical reasoning and a set of answer choices, output the single letter (A, B, C, D, or E) that best matches the reasoning. Output only the letter, nothing else.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Answer choices:\n{options_text}\n\n"
                        f"Clinical reasoning:\n{content}\n\n"
                        "Which answer choice letter (A, B, C, D, or E) does this reasoning support? "
                        "Output only the letter."
                    ),
                }
            ],
        )
        letter_text = extract_resp.content[0].text.strip()
        m = re.search(r"[A-Ea-e]", letter_text)
        letter = m.group(0).upper() if m else None
        extractor_cost = compute_cost(
            extract_resp.usage.input_tokens,
            extract_resp.usage.output_tokens,
        )
    except Exception as e:
        logger.error("Letter extractor error: %s", e)
        letter = None
        extractor_cost = None

    total_time = time.perf_counter() - t0
    return {
        "letter": letter,
        "time": round(total_time, 3),
        "cost": extractor_cost,  # only extractor cost; orchestrator internal costs not exposed
        "orchestrator_status": result.get("status"),
        "pipeline_time": round(pipeline_time, 3),
    }


# ── pipeline 2: monolithic baseline ──────────────────────────────────────────

def run_baseline(client: Anthropic, formatted_q: str) -> dict:
    """Call Claude Haiku directly with CoT reasoning."""
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=settings.model_name,
            max_tokens=1024,
            temperature=0.7,
            system="You are a clinical expert. Reason through the question carefully, then output your final answer.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        formatted_q
                        + "\n\nThink step by step, then on the last line write: "
                        "ANSWER: X (where X is A, B, C, D, or E)"
                    ),
                }
            ],
        )
    except Exception as e:
        logger.error("Baseline error: %s", e)
        return {"letter": None, "time": time.perf_counter() - t0, "cost": None, "error": str(e)}

    elapsed = time.perf_counter() - t0
    text = resp.content[0].text
    letter = extract_letter_regex(text)
    cost = compute_cost(resp.usage.input_tokens, resp.usage.output_tokens)

    return {
        "letter": letter,
        "time": round(elapsed, 3),
        "cost": round(cost, 6),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


# ── main loop ─────────────────────────────────────────────────────────────────

async def benchmark(questions: list[dict], resume: bool) -> None:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    orchestrator = MCPOrchestrator()

    # Load any existing partial results
    results: list[dict] = []
    if resume and RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Resuming from {len(results)} completed questions.")

    done_ids = {r["id"] for r in results}

    for q in questions:
        if q["id"] in done_ids:
            continue

        formatted_q = format_question(q)
        correct = q["answer"]

        # ── ADAPT-AI ──────────────────────────────────────────────────────────
        adapt = await run_adapt_ai(orchestrator, client, formatted_q, q["options"])
        adapt_letter = adapt.get("letter")

        # ── Baseline ──────────────────────────────────────────────────────────
        baseline = run_baseline(client, formatted_q)
        baseline_letter = baseline.get("letter")

        # ── Record ────────────────────────────────────────────────────────────
        n = len(results) + 1
        print(
            f"Q{n:3d}/100 | "
            f"ADAPT-AI: {adapt_letter or '?':1} | "
            f"Baseline: {baseline_letter or '?':1} | "
            f"Correct: {correct}"
        )

        results.append({
            "id": q["id"],
            "question": q["question"],
            "correct": correct,
            "adapt_ai": {
                "letter": adapt_letter,
                "correct": adapt_letter == correct if adapt_letter else False,
                "time": adapt.get("time"),
                "cost": adapt.get("cost"),
                "pipeline_time": adapt.get("pipeline_time"),
                "orchestrator_status": adapt.get("orchestrator_status"),
                "error": adapt.get("error"),
            },
            "baseline": {
                "letter": baseline_letter,
                "correct": baseline_letter == correct if baseline_letter else False,
                "time": baseline.get("time"),
                "cost": baseline.get("cost"),
                "input_tokens": baseline.get("input_tokens"),
                "output_tokens": baseline.get("output_tokens"),
                "error": baseline.get("error"),
            },
        })

        # Incremental save every 10 questions
        if len(results) % 10 == 0:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  [saved {len(results)} results → {RESULTS_PATH}]")

        await asyncio.sleep(SLEEP_BETWEEN_QUESTIONS)

    # Final save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nDone. {len(results)} results saved → {RESULTS_PATH}")

    # Quick summary
    adapt_correct = sum(1 for r in results if r["adapt_ai"]["correct"])
    base_correct = sum(1 for r in results if r["baseline"]["correct"])
    n = len(results)
    print(f"\nQuick summary ({n} questions):")
    print(f"  ADAPT-AI accuracy : {adapt_correct}/{n} = {adapt_correct/n:.1%}")
    print(f"  Baseline accuracy : {base_correct}/{n} = {base_correct/n:.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MedQA benchmark runner")
    parser.add_argument(
        "--questions", type=int, default=100,
        help="Number of questions to run (default: all 100)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip questions already recorded in medqa_results.json",
    )
    args = parser.parse_args()

    if not SAMPLE_PATH.exists():
        sys.exit(f"ERROR: {SAMPLE_PATH} not found. Run scripts/download_medqa.py first.")

    questions = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    if args.questions < len(questions):
        questions = questions[: args.questions]
        print(f"Running on first {args.questions} questions (smoke-test mode).")

    asyncio.run(benchmark(questions, resume=args.resume))


if __name__ == "__main__":
    main()

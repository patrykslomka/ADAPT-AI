#!/usr/bin/env python3
"""Clinical reasoning benchmark: ADAPT-AI multi-agent vs monolithic baseline.

Evaluates 30 open-ended clinical queries across 5 categories using ClinicalEvaluator
(ROUGE-L, concept recall, safety score, hallucination detection) instead of letter matching.
The router naturally decides RAT vs RAG for each query—no override.

Categories:
  complex_reasoning    — pathophysiology / mechanism queries (triggers RAT)
  differential_diagnosis — case vignettes requiring DDx lists (triggers RAT)
  treatment_planning   — management plans (triggers RAT)
  compliance_safety    — HIPAA / prescribing safety edge cases
  hallucination_trap   — false-premise queries that stress quality agent

Usage:
    python scripts/run_clinical_benchmark.py
    python scripts/run_clinical_benchmark.py --questions 5   # smoke test
    python scripts/run_clinical_benchmark.py --resume       # skip completed
    python scripts/run_clinical_benchmark.py --no-bertscore  # skip slow BERTScore
"""
import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from anthropic import Anthropic
from adapt_ai.config import settings
from adapt_ai.orchestrator.client import build_mcp_client
from adapt_ai.agents.graph import build_graph
from adapt_ai.llmops.tracing import setup_tracing
from adapt_ai.llmops.usage import get_accumulator
from evaluation.metrics import ClinicalEvaluator

setup_tracing()

DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
DATASET_PATH = DATA_DIR / "clinical_reasoning_benchmark.json"
RESULTS_PATH = DATA_DIR / "clinical_benchmark_results.json"

HAIKU_INPUT_PRICE = 0.80 / 1_000_000
HAIKU_OUTPUT_PRICE = 4.00 / 1_000_000

SLEEP_BETWEEN_QUESTIONS = 1

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * HAIKU_INPUT_PRICE + output_tokens * HAIKU_OUTPUT_PRICE


def score_with_evaluator(
    evaluator: ClinicalEvaluator,
    prediction: str,
    item: dict,
) -> dict:
    result = evaluator.evaluate_response(
        prediction=prediction,
        reference=item["reference_answer"],
        required_concepts=item.get("required_concepts"),
        critical_concepts=item.get("critical_concepts"),
        hallucination_patterns=item.get("hallucination_patterns"),
    )
    return {
        "rouge_l": result.rouge_l,
        "bleu_4": result.bleu_4,
        "concept_recall": result.concept_recall,
        "concept_f1": result.concept_f1,
        "safety_score": result.safety_score,
        "critical_omissions": result.critical_omission_count,
        "hallucinations": result.hallucination_count,
        "overall_score": result.overall_score,
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


# ── ADAPT-AI pipeline ─────────────────────────────────────────────────────────

async def run_adapt_ai(pipeline, item: dict, q_id: int) -> dict:
    query = item["query"]
    t0 = time.perf_counter()
    try:
        result = await pipeline.ainvoke(
            {
                "query": query,
                "subject_id": None,
                "domain": "healthcare",
                "session_id": f"clinical-bench-{q_id}",
                "use_rat": False,      # overwritten by intent_and_retrieve node
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
            config={"configurable": {"thread_id": f"clinical-{q_id}"}},
        )
    except Exception as e:
        elapsed = time.perf_counter() - t0
        logger.error("ADAPT-AI pipeline error q%d: %s", q_id, e)
        return {
            "response": "",
            "time": round(elapsed, 3),
            "error": str(e),
            "use_rat": False,
            "revision_count": 0,
        }

    elapsed = time.perf_counter() - t0
    response_text = result.get("final_response") or result.get("primary_response", "")

    usage = result.get("llm_usage") or {}
    if not usage:
        # Fallback: on the retry path aggregate_response may not receive the
        # accumulator via state (MemorySaver isolation edge case). Read directly.
        acc = get_accumulator(f"clinical-bench-{q_id}")
        if acc:
            usage = acc.to_dict()
    return {
        "response": response_text,
        "time": round(elapsed, 3),
        "error": result.get("error"),
        "use_rat": result.get("use_rat", False),
        "revision_count": result.get("revision_count", 0),
        "agent_statuses": result.get("agent_statuses", {}),
        "compliance_result": result.get("compliance_result", {}),
        "quality_result": result.get("quality_result", {}),
        "llm_usage": usage,
        "total_cost_usd": usage.get("total_cost_usd"),
        "total_input_tokens": usage.get("total_input_tokens"),
        "total_output_tokens": usage.get("total_output_tokens"),
    }


# ── Monolithic baseline ───────────────────────────────────────────────────────

def run_baseline(client: Anthropic, item: dict) -> dict:
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=settings.model_name,
            max_tokens=1024,
            temperature=0.3,
            system=(
                "You are a clinical expert. "
                "Answer the following medical question accurately and completely. "
                "If the question contains a false or dangerous premise, correct it clearly."
            ),
            messages=[{"role": "user", "content": item["query"]}],
        )
    except Exception as e:
        return {
            "response": "",
            "time": round(time.perf_counter() - t0, 3),
            "cost": None,
            "error": str(e),
        }

    elapsed = time.perf_counter() - t0
    text = resp.content[0].text
    cost = compute_cost(resp.usage.input_tokens, resp.usage.output_tokens)

    return {
        "response": text,
        "time": round(elapsed, 3),
        "cost": round(cost, 6),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "error": None,
    }


# ── Main benchmark loop ───────────────────────────────────────────────────────

async def benchmark(items: list[dict], resume: bool, use_bertscore: bool, use_llm_judge: bool, include_quality: bool = True) -> None:
    client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
    evaluator = ClinicalEvaluator(
        use_bertscore=use_bertscore,
        use_llm_judge=use_llm_judge,
        anthropic_api_key=settings.anthropic_api_key.get_secret_value() if use_llm_judge else None,
    )

    print("Initialising ADAPT-AI pipeline (LangGraph + MCP)…")
    mcp_client = build_mcp_client()
    pipeline = build_graph(mcp_client, include_quality=include_quality)
    print("Pipeline ready.\n")

    results: list[dict] = []
    if resume and RESULTS_PATH.exists():
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        print(f"Resuming from {len(results)} completed questions.")

    done_ids = _completed_ids(results)
    # Drop errored entries so they are re-run and re-appended cleanly.
    results = [r for r in results if r["id"] in done_ids]
    todo = [q for q in items if q["id"] not in done_ids]
    total = len(items)

    for item in todo:
        q_id = item["id"]
        category = item["category"]
        print(f"Q{q_id + 1:02d}/{total} [{category}] ", end="", flush=True)

        adapt_raw = await run_adapt_ai(pipeline, item, q_id)
        base_raw = run_baseline(client, item)

        adapt_scores: dict = {}
        base_scores: dict = {}

        if adapt_raw["response"] and not adapt_raw.get("error"):
            adapt_scores = score_with_evaluator(evaluator, adapt_raw["response"], item)
        if base_raw["response"] and not base_raw.get("error"):
            base_scores = score_with_evaluator(evaluator, base_raw["response"], item)

        adapt_overall = adapt_scores.get("overall_score")
        base_overall = base_scores.get("overall_score")

        a_str = f"{adapt_overall:.3f}" if adapt_overall is not None else "ERR"
        b_str = f"{base_overall:.3f}" if base_overall is not None else "ERR"
        rat_str = "RAT" if adapt_raw.get("use_rat") else "RAG"
        print(
            f"ADAPT={a_str} Base={b_str} "
            f"[{rat_str}] ({adapt_raw['time']:.1f}s / {base_raw['time']:.1f}s)"
        )

        results.append({
            "id": q_id,
            "category": category,
            "query": item["query"][:120],
            "adapt_ai": {**adapt_raw, **adapt_scores},
            "baseline": {**base_raw, **base_scores},
        })

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

        if q_id < total - 1:
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    print(f"\nDone. Results saved to {RESULTS_PATH}")
    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    from collections import defaultdict

    categories = defaultdict(lambda: {"adapt": [], "base": []})
    for r in results:
        cat = r["category"]
        a = r["adapt_ai"].get("overall_score")
        b = r["baseline"].get("overall_score")
        if a is not None:
            categories[cat]["adapt"].append(a)
        if b is not None:
            categories[cat]["base"].append(b)

    all_adapt = [r["adapt_ai"].get("overall_score") for r in results if r["adapt_ai"].get("overall_score") is not None]
    all_base = [r["baseline"].get("overall_score") for r in results if r["baseline"].get("overall_score") is not None]

    def avg(lst):
        return sum(lst) / len(lst) if lst else None

    n = len(results)
    print(f"\n{'='*60}")
    print(f"  Clinical Reasoning Benchmark  ({n} questions)")
    print(f"{'='*60}")
    print(f"\nOverall mean score (0-1):")
    adapt_avg = avg(all_adapt)
    base_avg = avg(all_base)
    if adapt_avg is not None:
        print(f"  ADAPT-AI : {adapt_avg:.3f}")
    if base_avg is not None:
        print(f"  Baseline : {base_avg:.3f}")
    if adapt_avg and base_avg:
        print(f"  Delta    : {adapt_avg - base_avg:+.3f}")

    print(f"\nPer-category mean overall score:")
    for cat in sorted(categories):
        a_avg = avg(categories[cat]["adapt"])
        b_avg = avg(categories[cat]["base"])
        a_str = f"{a_avg:.3f}" if a_avg is not None else "N/A"
        b_str = f"{b_avg:.3f}" if b_avg is not None else "N/A"
        delta = f"{a_avg - b_avg:+.3f}" if a_avg is not None and b_avg is not None else "N/A"
        print(f"  {cat:<25} ADAPT={a_str}  Base={b_str}  Δ={delta}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=None, help="Limit to first N questions")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed questions")
    parser.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (faster)")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge correctness scoring (+30%% weight)")
    parser.add_argument("--no-quality", action="store_true",
                        help="Ablation: run without the quality agent")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        print(f"Dataset not found at {DATASET_PATH}")
        sys.exit(1)

    items = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    if args.questions:
        items = items[: args.questions]

    asyncio.run(benchmark(
        items,
        resume=args.resume,
        use_bertscore=not args.no_bertscore,
        use_llm_judge=args.judge,
        include_quality=not args.no_quality,
    ))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze MedQA benchmark results.

Reads data/evaluation/medqa_results.json and produces:
  - Accuracy + 95% Wilson score CI for each pipeline
  - McNemar's test for statistical significance
  - Per-question comparison table
  - Where each pipeline diverges
  - Response time and cost comparison
  - data/evaluation/medqa_report.json
  - data/evaluation/medqa_summary.md
"""
import json
import math
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"
RESULTS_PATH = DATA_DIR / "medqa_results.json"
REPORT_PATH = DATA_DIR / "medqa_report.json"
SUMMARY_PATH = DATA_DIR / "medqa_summary.md"


# ── statistics ────────────────────────────────────────────────────────────────

def wilson_ci(n_correct: int, n_total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n_total == 0:
        return (0.0, 0.0)
    p = n_correct / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denom
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def mcnemar_test(b: int, c: int) -> tuple[float, str]:
    """McNemar's test with continuity correction.

    b = cases where adapt_ai correct, baseline wrong
    c = cases where adapt_ai wrong, baseline correct
    Returns (chi2, interpretation)
    """
    if b + c == 0:
        return (0.0, "No discordant pairs — test not applicable.")
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # chi2 distribution, df=1: critical values
    if chi2 >= 10.83:
        sig = "p < 0.001 (highly significant)"
    elif chi2 >= 6.63:
        sig = "p < 0.01 (significant)"
    elif chi2 >= 3.84:
        sig = "p < 0.05 (significant)"
    else:
        sig = "p ≥ 0.05 (not significant)"
    return (round(chi2, 4), sig)


def safe_mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt_pct(n: int, d: int) -> str:
    return f"{n}/{d} ({n/d:.1%})" if d else "N/A"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not RESULTS_PATH.exists():
        sys.exit(f"ERROR: {RESULTS_PATH} not found. Run run_medqa_benchmark.py first.")

    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    n = len(results)

    if n == 0:
        sys.exit("ERROR: results file is empty.")

    print(f"\n{'='*60}")
    print(f"  MedQA Benchmark Analysis  ({n} questions)")
    print(f"{'='*60}")

    # ── accuracy ──────────────────────────────────────────────────────────────
    adapt_correct_list = [r["adapt_ai"]["correct"] for r in results]
    base_correct_list  = [r["baseline"]["correct"]  for r in results]

    adapt_n = sum(adapt_correct_list)
    base_n  = sum(base_correct_list)

    adapt_acc = adapt_n / n
    base_acc  = base_n  / n

    adapt_ci = wilson_ci(adapt_n, n)
    base_ci  = wilson_ci(base_n,  n)

    print(f"\n{'─'*60}")
    print("Accuracy (95% Wilson CI)")
    print(f"{'─'*60}")
    print(f"  ADAPT-AI : {adapt_n}/{n} = {adapt_acc:.1%}  [{adapt_ci[0]:.1%} – {adapt_ci[1]:.1%}]")
    print(f"  Baseline : {base_n}/{n}  = {base_acc:.1%}  [{base_ci[0]:.1%} – {base_ci[1]:.1%}]")
    delta = adapt_acc - base_acc
    print(f"  Δ (ADAPT-AI − Baseline) = {delta:+.1%}")

    # ── discordant pairs for McNemar ──────────────────────────────────────────
    b = sum(1 for r in results if r["adapt_ai"]["correct"] and not r["baseline"]["correct"])
    c = sum(1 for r in results if not r["adapt_ai"]["correct"] and r["baseline"]["correct"])

    chi2, sig = mcnemar_test(b, c)

    print(f"\n{'─'*60}")
    print("McNemar's Test (discordant pairs)")
    print(f"{'─'*60}")
    print(f"  ADAPT-AI correct, Baseline wrong : {b}")
    print(f"  ADAPT-AI wrong, Baseline correct : {c}")
    print(f"  χ² (with continuity correction)  : {chi2}")
    print(f"  Result                           : {sig}")

    # ── response time ─────────────────────────────────────────────────────────
    adapt_times  = [r["adapt_ai"]["time"]  for r in results]
    base_times   = [r["baseline"]["time"]  for r in results]

    adapt_avg_t = safe_mean(adapt_times)
    base_avg_t  = safe_mean(base_times)

    print(f"\n{'─'*60}")
    print("Average Response Time")
    print(f"{'─'*60}")
    if adapt_avg_t: print(f"  ADAPT-AI : {adapt_avg_t:.2f}s")
    if base_avg_t:  print(f"  Baseline : {base_avg_t:.2f}s")
    if adapt_avg_t and base_avg_t:
        print(f"  ADAPT-AI is {adapt_avg_t/base_avg_t:.1f}× slower than Baseline")

    # ── cost ──────────────────────────────────────────────────────────────────
    base_costs  = [r["baseline"]["cost"]  for r in results if r["baseline"].get("cost") is not None]
    adapt_costs = [r["adapt_ai"]["cost"]  for r in results if r["adapt_ai"].get("cost") is not None]

    print(f"\n{'─'*60}")
    print("Cost (USD)")
    print(f"{'─'*60}")
    if base_costs:
        print(f"  Baseline total   : ${sum(base_costs):.4f}  (avg ${safe_mean(base_costs):.6f}/q)")
    if adapt_costs:
        print(f"  ADAPT-AI extractor cost (letter extraction only): ${sum(adapt_costs):.4f}")
        print(f"  Note: ADAPT-AI orchestrator internal LLM costs are not exposed in result metadata.")

    # ── divergence analysis ───────────────────────────────────────────────────
    only_adapt = [r for r in results if r["adapt_ai"]["correct"] and not r["baseline"]["correct"]]
    only_base  = [r for r in results if r["baseline"]["correct"]  and not r["adapt_ai"]["correct"]]

    print(f"\n{'─'*60}")
    print(f"Divergence Analysis")
    print(f"{'─'*60}")
    print(f"  ADAPT-AI correct, Baseline wrong ({len(only_adapt)} questions):")
    for r in only_adapt[:5]:
        print(f"    Q{r['id']:3d}: {r['question'][:70]}…  [correct={r['correct']}]")
    if len(only_adapt) > 5:
        print(f"    … and {len(only_adapt)-5} more")

    print(f"\n  ADAPT-AI wrong, Baseline correct ({len(only_base)} questions):")
    for r in only_base[:5]:
        print(f"    Q{r['id']:3d}: {r['question'][:70]}…  [correct={r['correct']}]")
    if len(only_base) > 5:
        print(f"    … and {len(only_base)-5} more")

    # ── per-question comparison table (first 20) ──────────────────────────────
    print(f"\n{'─'*60}")
    print("Per-Question Comparison (first 20 rows)")
    print(f"{'─'*60}")
    header = f"{'Q':>3}  {'Correct':^7}  {'ADAPT':^5}  {'Base':^5}  {'A✓':^4}  {'B✓':^4}"
    print(header)
    print("─" * len(header))
    for r in results[:20]:
        a_ltr = r["adapt_ai"]["letter"] or "?"
        b_ltr = r["baseline"]["letter"] or "?"
        a_ok  = "✓" if r["adapt_ai"]["correct"] else "✗"
        b_ok  = "✓" if r["baseline"]["correct"]  else "✗"
        print(f"{r['id']:>3}  {r['correct']:^7}  {a_ltr:^5}  {b_ltr:^5}  {a_ok:^4}  {b_ok:^4}")

    # ── JSON report ───────────────────────────────────────────────────────────
    report = {
        "n_questions": n,
        "adapt_ai": {
            "n_correct": adapt_n,
            "accuracy": round(adapt_acc, 4),
            "ci_95_lower": round(adapt_ci[0], 4),
            "ci_95_upper": round(adapt_ci[1], 4),
            "avg_time_s": round(adapt_avg_t, 3) if adapt_avg_t else None,
            "extractor_total_cost_usd": round(sum(adapt_costs), 6) if adapt_costs else None,
        },
        "baseline": {
            "n_correct": base_n,
            "accuracy": round(base_acc, 4),
            "ci_95_lower": round(base_ci[0], 4),
            "ci_95_upper": round(base_ci[1], 4),
            "avg_time_s": round(base_avg_t, 3) if base_avg_t else None,
            "total_cost_usd": round(sum(base_costs), 6) if base_costs else None,
        },
        "delta_accuracy": round(delta, 4),
        "mcnemar": {
            "b_adapt_correct_base_wrong": b,
            "c_adapt_wrong_base_correct": c,
            "chi2": chi2,
            "result": sig,
        },
        "n_adapt_only_correct": len(only_adapt),
        "n_base_only_correct": len(only_base),
        "per_question": [
            {
                "id": r["id"],
                "correct": r["correct"],
                "adapt_ai_letter": r["adapt_ai"]["letter"],
                "adapt_ai_correct": r["adapt_ai"]["correct"],
                "adapt_ai_time": r["adapt_ai"]["time"],
                "baseline_letter": r["baseline"]["letter"],
                "baseline_correct": r["baseline"]["correct"],
                "baseline_time": r["baseline"]["time"],
            }
            for r in results
        ],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Saved JSON report → {REPORT_PATH}]")

    # ── Markdown summary ──────────────────────────────────────────────────────
    md = f"""# MedQA Benchmark Summary

**Model**: `claude-3-5-haiku-20241022`
**Questions**: {n} (random sample from MedQA US 5-option USMLE test set, seed=42)

## Accuracy

| System | Correct | Accuracy | 95% CI |
|--------|---------|----------|--------|
| ADAPT-AI (multi-agent) | {adapt_n}/{n} | {adapt_acc:.1%} | [{adapt_ci[0]:.1%} – {adapt_ci[1]:.1%}] |
| Baseline (monolithic) | {base_n}/{n} | {base_acc:.1%} | [{base_ci[0]:.1%} – {base_ci[1]:.1%}] |

**Δ (ADAPT-AI − Baseline)**: {delta:+.1%}

## Statistical Significance (McNemar's Test)

| | |
|--|--|
| ADAPT-AI correct, Baseline wrong | {b} |
| ADAPT-AI wrong, Baseline correct | {c} |
| χ² (continuity-corrected) | {chi2} |
| Result | {sig} |

## Response Time

| System | Avg time/question |
|--------|------------------|
| ADAPT-AI | {f'{adapt_avg_t:.2f}s' if adapt_avg_t else 'N/A'} |
| Baseline | {f'{base_avg_t:.2f}s' if base_avg_t else 'N/A'} |

{"ADAPT-AI is **" + f"{adapt_avg_t/base_avg_t:.1f}×**" + " slower than Baseline." if adapt_avg_t and base_avg_t else ""}

> **Note on ADAPT-AI cost**: The orchestrator's internal LLM calls (Primary Agent, Quality Agent)
> are not directly exposed in the returned metadata. Only the letter-extractor call cost is
> captured ({f'${sum(adapt_costs):.4f} total' if adapt_costs else 'N/A'}).
> Baseline total cost: {f'${sum(base_costs):.4f}' if base_costs else 'N/A'}.

## Divergence

- Questions where **only ADAPT-AI** was correct: **{len(only_adapt)}**
- Questions where **only Baseline** was correct: **{len(only_base)}**

## Architecture Notes

**ADAPT-AI pipeline** runs three agents in sequence:
1. **Primary Agent** — clinical reasoning via RAG (ChromaDB) or RAT (multi-step retrieval)
2. **Compliance Agent** — HIPAA/FDA rule-based validation
3. **Quality Agent** — hallucination detection with one retry loop if needed

The primary agent's answer is then passed to a separate letter-extractor LLM call
to convert clinical prose into a single answer letter.

**Baseline pipeline** is a single `claude-3-5-haiku-20241022` call with a CoT system prompt,
extracting `ANSWER: X` from the response.
"""

    SUMMARY_PATH.write_text(md, encoding="utf-8")
    print(f"[Saved Markdown summary → {SUMMARY_PATH}]")


if __name__ == "__main__":
    main()

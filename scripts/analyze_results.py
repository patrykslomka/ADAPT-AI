"""Analyze domain reasoning benchmark results (healthcare / legal / finance).

Reads data/evaluation/<domain>_benchmark_results.json and produces:
  - Per-metric aggregate comparison (ROUGE-L, concept recall, safety, hallucinations)
  - Per-category breakdowns with delta
  - Wilcoxon signed-rank test for statistical significance on overall_score
  - RAT routing statistics per category
  - Per-question comparison table
  - data/evaluation/<domain>_report.json
  - data/evaluation/<domain>_summary.md

Usage:
    python scripts/analyze_results.py --domain healthcare
    python scripts/analyze_results.py --domain legal
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from scipy.stats import wilcoxon as _scipy_wilcoxon, rankdata as _rankdata, t as _t_dist

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
from adapt_ai.config import settings

DATA_DIR = Path(__file__).parent.parent / "data" / "evaluation"

# Per-domain results/report/summary filenames - uniform <domain>_* naming.
RESULTS_FILES = {
    "healthcare": "healthcare_benchmark_results.json",
    "legal": "legal_benchmark_results.json",
    "finance": "finance_benchmark_results.json",
}
REPORT_FILES = {
    "healthcare": "healthcare_report.json",
    "legal": "legal_report.json",
    "finance": "finance_report.json",
}
SUMMARY_FILES = {
    "healthcare": "healthcare_summary.md",
    "legal": "legal_summary.md",
    "finance": "finance_summary.md",
}

# Category order is derived from the data at runtime (domains differ); this is only a fallback.
CATEGORIES: list[str] = []

METRICS = ["overall_score", "rouge_l", "concept_recall", "safety_score"]


#  statistics 

def safe_mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def safe_std(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1)), 4)


def wilcoxon_signed_rank(diffs: list[float]) -> tuple[float | None, float, str]:
    """Wilcoxon signed-rank test via scipy.

    Returns (W_statistic, p_value, summary_string).
    Uses scipy's exact/normal approximation with tie correction.
    """
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return None, 1.0, "All differences zero - test not applicable."

    stat, p = _scipy_wilcoxon(nz, zero_method="wilcox", correction=False,
                              alternative="two-sided")
    p_val = float(p)
    n = len(nz)
    r = rank_biserial(diffs)

    pos = sum(d > 0 for d in nz)
    neg = sum(d < 0 for d in nz)
    direction = "ADAPT-AI > Baseline" if pos > neg else "Baseline > ADAPT-AI"

    if p_val < 0.001:
        sig = "p < 0.001 (highly significant)"
    elif p_val < 0.01:
        sig = "p < 0.01 (significant)"
    elif p_val < 0.05:
        sig = "p < 0.05 (significant)"
    else:
        sig = f"p = {p_val:.3f} (not significant)"

    summary = f"{sig} ({direction}, W={stat:.1f}, p={p_val:.4g}, r={r:+.2f}, n={n})"
    return round(float(stat), 2), p_val, summary


def rank_biserial(diffs: list[float]) -> float:
    """Matched-pairs rank-biserial effect size for the Wilcoxon signed-rank test.

    Returns a value in [-1, 1]. Positive = ADAPT-AI tends to be higher.
    """
    nz = [d for d in diffs if d != 0.0]
    if not nz:
        return 0.0
    ranks = _rankdata([abs(d) for d in nz])
    w_plus = sum(r for r, d in zip(ranks, nz) if d > 0)
    total = float(ranks.sum())
    return float(2 * w_plus / total - 1) if total > 0 else 0.0


def holm_correct(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down correction.

    Returns adjusted p-values in the same order as the input list.
    Each adjusted value is min(1.0, (m - rank) * raw_p), monotone-enforced.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for step, idx in enumerate(order):
        corrected = (m - step) * pvals[idx]
        running = max(running, corrected)
        adj[idx] = min(1.0, running)
    return adj


def wilson_ci(values: list[float], z: float | None = None) -> tuple[float, float]:
    """95% CI for a mean score via t-distribution."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    m = sum(vals) / n
    std = math.sqrt(sum((x - m) ** 2 for x in vals) / max(n - 1, 1))
    t_crit = z if z is not None else float(_t_dist.ppf(0.975, df=max(n - 1, 1)))
    margin = t_crit * std / math.sqrt(n)
    return (round(max(0.0, m - margin), 4), round(min(1.0, m + margin), 4))


#  helpers ─

def extract_metric(results: list[dict], pipeline: str, metric: str) -> list:
    return [r[pipeline].get(metric) for r in results]


def fmt(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def fmt_delta(a, b) -> str:
    if a is None or b is None:
        return "N/A"
    return f"{a - b:+.3f}"


#  matrix mode ─

def _summarise_results(results: list[dict]) -> dict:
    """Compute summary statistics from a list of benchmark result records."""
    adapt_scores = [r["adapt_ai"].get("overall_score") for r in results
                    if r.get("adapt_ai", {}).get("overall_score") is not None]
    base_scores  = [r["baseline"].get("overall_score") for r in results
                    if r.get("baseline", {}).get("overall_score") is not None]
    adapt_disc   = [1.0 if r["adapt_ai"].get("has_disclaimer") else 0.0
                    for r in results if "adapt_ai" in r]

    if not adapt_scores or not base_scores:
        return {}

    adapt_mean = sum(adapt_scores) / len(adapt_scores)
    base_mean  = sum(base_scores)  / len(base_scores)
    diffs      = [a - b for a, b in zip(adapt_scores, base_scores)]
    delta      = adapt_mean - base_mean

    try:
        W, p_val, _ = wilcoxon_signed_rank(diffs)
        r = rank_biserial(diffs)
    except Exception:
        W, p_val, r = None, 1.0, 0.0

    disc_rate = (sum(adapt_disc) / len(adapt_disc)) if adapt_disc else 0.0

    return {
        "adapt_mean": adapt_mean,
        "base_mean": base_mean,
        "delta": delta,
        "p_overall": p_val,
        "r": r,
        "disclaimer_rate": disc_rate,
        "n": len(adapt_scores),
    }


def _print_matrix_table() -> None:
    """Print a combined model x domain results table from matrix run outputs."""
    MATRIX_DIR = DATA_DIR / "matrix"
    if not MATRIX_DIR.exists():
        print("[matrix] No matrix results found. Run scripts/run_matrix.py first.")
        return

    MODELS  = ["haiku", "sonnet", "qwen7b"]
    DOMAINS = ["healthcare", "legal", "finance"]

    # Collect all available cells
    cells: dict[tuple[str, str], dict] = {}
    for model in MODELS:
        for domain in DOMAINS:
            path = MATRIX_DIR / model / f"{domain}_benchmark_results.json"
            if not path.exists():
                continue
            try:
                raw     = json.loads(path.read_text(encoding="utf-8"))
                results = raw if isinstance(raw, list) else raw.get("results", [])
                cells[(model, domain)] = _summarise_results(results)
            except Exception as e:
                print(f"[warn] Could not read {path}: {e}", file=sys.stderr)

    if not cells:
        print("[matrix] No readable result files found.")
        return

    # Print headline table
    print("\n## Cross-Model Results (ADAPT-AI full vs b1_disclaimer baseline)\n")
    print(f"{'Model':<8} {'Domain':<12} {'ADAPT':>7} {'Base':>7} {'Δ':>7} "
          f"{'r':>6} {'p-corr':>8} {'disclaimer%':>12}")
    print("-" * 75)

    all_p:      list[float]              = []
    cell_order: list[tuple[str, str]]    = []
    for model in MODELS:
        for domain in DOMAINS:
            if (model, domain) in cells:
                cell_order.append((model, domain))
                all_p.append(cells[(model, domain)].get("p_overall", 1.0))

    corrected_p = holm_correct(all_p) if all_p else []

    for i, (model, domain) in enumerate(cell_order):
        c      = cells[(model, domain)]
        adapt  = c.get("adapt_mean",      "N/A")
        base   = c.get("base_mean",       "N/A")
        delta  = c.get("delta",           "N/A")
        r      = c.get("r",               "N/A")
        p_adj  = corrected_p[i] if corrected_p else "N/A"
        disc   = c.get("disclaimer_rate", "N/A")

        adapt_s = f"{adapt:.3f}"  if isinstance(adapt, float) else adapt
        base_s  = f"{base:.3f}"   if isinstance(base,  float) else base
        delta_s = f"{delta:+.3f}" if isinstance(delta, float) else delta
        r_s     = f"{r:+.2f}"    if isinstance(r,     float) else r
        p_s     = f"{p_adj:.4f}" if isinstance(p_adj, float) else p_adj
        disc_s  = f"{disc:.0%}"  if isinstance(disc,  float) else disc

        print(f"{model:<8} {domain:<12} {adapt_s:>7} {base_s:>7} {delta_s:>7} "
              f"{r_s:>6} {p_s:>8} {disc_s:>12}")

    print()


#  main 

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(RESULTS_FILES), default="healthcare",
                        help="Domain whose benchmark results to analyse (default: healthcare)")
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Read matrix results from data/evaluation/matrix/<model>/ and emit combined table",
    )
    args = parser.parse_args()

    if args.matrix:
        _print_matrix_table()
        return

    domain = args.domain

    results_path = DATA_DIR / RESULTS_FILES[domain]
    report_path = DATA_DIR / REPORT_FILES[domain]
    summary_path = DATA_DIR / SUMMARY_FILES[domain]

    if not results_path.exists():
        sys.exit(f"ERROR: {results_path} not found. "
                 f"Run: python scripts/run_benchmark.py --domain {domain}")

    results = json.loads(results_path.read_text(encoding="utf-8"))
    n = len(results)
    if n == 0:
        sys.exit("ERROR: results file is empty.")

    # Categories are domain-specific - derive them from the data, preserving order.
    global CATEGORIES
    CATEGORIES = list(dict.fromkeys(r["category"] for r in results))

    print(f"\n{'='*64}")
    print(f"  {domain.title()} Reasoning Benchmark Analysis  ({n} questions)")
    print(f"{'='*64}")

    #  aggregate metrics ─
    print(f"\n{'─'*64}")
    print("Aggregate Metrics (mean ± SD)")
    print(f"{'─'*64}")
    header = f"  {'Metric':<20} {'ADAPT-AI':>12} {'Baseline':>12} {'Delta':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))

    agg_data: dict = {"adapt_ai": {}, "baseline": {}}

    for metric in METRICS:
        adapt_vals = extract_metric(results, "adapt_ai", metric)
        base_vals = extract_metric(results, "baseline", metric)
        adapt_m = safe_mean(adapt_vals)
        base_m = safe_mean(base_vals)
        adapt_s = safe_std(adapt_vals)
        base_s = safe_std(base_vals)

        a_str = f"{adapt_m:.3f}±{adapt_s:.3f}" if adapt_m is not None and adapt_s is not None else "N/A"
        b_str = f"{base_m:.3f}±{base_s:.3f}" if base_m is not None and base_s is not None else "N/A"
        d_str = fmt_delta(adapt_m, base_m)

        print(f"  {metric:<20} {a_str:>12} {b_str:>12} {d_str:>10}")
        agg_data["adapt_ai"][metric] = {"mean": adapt_m, "std": adapt_s}
        agg_data["baseline"][metric] = {"mean": base_m, "std": base_s}

    # Hallucinations and critical omissions (lower is better)
    for metric, label in [("hallucinations", "hallucinations"), ("critical_omissions", "critical_omissions")]:
        adapt_vals = extract_metric(results, "adapt_ai", metric)
        base_vals = extract_metric(results, "baseline", metric)
        adapt_m = safe_mean(adapt_vals)
        base_m = safe_mean(base_vals)
        a_str = f"{adapt_m:.2f}" if adapt_m is not None else "N/A"
        b_str = f"{base_m:.2f}" if base_m is not None else "N/A"
        d_str = fmt_delta(adapt_m, base_m)
        print(f"  {label:<20} {a_str:>12} {b_str:>12} {d_str:>10}  (↓ better)")
        agg_data["adapt_ai"][metric] = {"mean": adapt_m}
        agg_data["baseline"][metric] = {"mean": base_m}

    #  statistical significance on overall_score ─
    adapt_overall = [r["adapt_ai"].get("overall_score") for r in results]
    base_overall = [r["baseline"].get("overall_score") for r in results]
    diffs = [
        a - b
        for a, b in zip(adapt_overall, base_overall)
        if a is not None and b is not None
    ]

    W, _p_overall, sig_str = wilcoxon_signed_rank(diffs)
    adapt_ci = wilson_ci(adapt_overall)
    base_ci = wilson_ci(base_overall)
    adapt_mean_overall = safe_mean(adapt_overall)
    base_mean_overall = safe_mean(base_overall)

    print(f"\n{'─'*64}")
    print("Statistical Significance - Wilcoxon Signed-Rank Tests")
    print(f"{'─'*64}")
    print(f"  overall_score (n={len(diffs)})")
    print(f"    ADAPT-AI mean: {fmt(adapt_mean_overall)}  95% CI [{adapt_ci[0]:.3f}–{adapt_ci[1]:.3f}]")
    print(f"    Baseline mean: {fmt(base_mean_overall)}  95% CI [{base_ci[0]:.3f}–{base_ci[1]:.3f}]")
    print(f"    Δ: {fmt_delta(adapt_mean_overall, base_mean_overall)}  →  {sig_str}")

    # safety_score (n=30) - ADAPT-AI's strongest advantage
    adapt_safety = [r["adapt_ai"].get("safety_score") for r in results]
    base_safety = [r["baseline"].get("safety_score") for r in results]
    safety_diffs = [a - b for a, b in zip(adapt_safety, base_safety) if a is not None and b is not None]
    W_s, _p_safety, sig_s = wilcoxon_signed_rank(safety_diffs)
    adapt_safety_ci = wilson_ci(adapt_safety)
    base_safety_ci = wilson_ci(base_safety)
    adapt_safety_mean = safe_mean(adapt_safety)
    base_safety_mean = safe_mean(base_safety)
    print(f"\n  safety_score (n={len(safety_diffs)})")
    print(f"    ADAPT-AI mean: {fmt(adapt_safety_mean)}  95% CI [{adapt_safety_ci[0]:.3f}–{adapt_safety_ci[1]:.3f}]")
    print(f"    Baseline mean: {fmt(base_safety_mean)}  95% CI [{base_safety_ci[0]:.3f}–{base_safety_ci[1]:.3f}]")
    print(f"    Δ: {fmt_delta(adapt_safety_mean, base_safety_mean)}  →  {sig_s}")
    print(f"    Note: baseline 0.80 floor reflects absence of mandatory safety disclaimer")

    # compliance_safety category (n=6)
    comp_results = [r for r in results if r["category"] == "compliance_safety"]
    comp_a = [r["adapt_ai"].get("overall_score") for r in comp_results]
    comp_b = [r["baseline"].get("overall_score") for r in comp_results]
    comp_diffs = [a - b for a, b in zip(comp_a, comp_b) if a is not None and b is not None]
    W_c, _p_comp, sig_c = wilcoxon_signed_rank(comp_diffs)
    print(f"\n  overall_score - compliance_safety category (n={len(comp_results)})")
    print(f"    ADAPT-AI mean: {fmt(safe_mean(comp_a))}  Baseline mean: {fmt(safe_mean(comp_b))}")
    print(f"    Δ: {fmt_delta(safe_mean(comp_a), safe_mean(comp_b))}  →  {sig_c}")

    #  per-category breakdown 
    print(f"\n{'─'*64}")
    print("Per-Category Breakdown (mean overall_score)")
    print(f"{'─'*64}")
    cat_data: dict = {}

    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        n_cat = len(cat_results)
        a_vals = [r["adapt_ai"].get("overall_score") for r in cat_results]
        b_vals = [r["baseline"].get("overall_score") for r in cat_results]
        rat_count = sum(1 for r in cat_results if r["adapt_ai"].get("use_rat"))

        a_m = safe_mean(a_vals)
        b_m = safe_mean(b_vals)
        delta = (a_m - b_m) if a_m is not None and b_m is not None else None

        cat_data[cat] = {
            "n": n_cat,
            "adapt_mean": a_m,
            "base_mean": b_m,
            "delta": round(delta, 4) if delta is not None else None,
            "rat_triggered_pct": round(rat_count / n_cat * 100, 1) if n_cat > 0 else None,
        }

        a_str = f"{a_m:.3f}" if a_m is not None else "N/A"
        b_str = f"{b_m:.3f}" if b_m is not None else "N/A"
        d_str = f"{delta:+.3f}" if delta is not None else "N/A"
        rat_str = f"{rat_count}/{n_cat} RAT"
        print(f"  {cat:<25} n={n_cat}  ADAPT={a_str}  Base={b_str}  Δ={d_str}  [{rat_str}]")

    # Per-category detail for concept_recall and safety_score
    print(f"\n{'─'*64}")
    print("Per-Category Detail (concept_recall / safety_score / hallucinations)")
    print(f"{'─'*64}")
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        for metric in ["concept_recall", "safety_score", "hallucinations"]:
            a_m = safe_mean([r["adapt_ai"].get(metric) for r in cat_results])
            b_m = safe_mean([r["baseline"].get(metric) for r in cat_results])
            cat_data[cat][f"adapt_{metric}"] = a_m
            cat_data[cat][f"base_{metric}"] = b_m
        a_cr = fmt(cat_data[cat].get("adapt_concept_recall"))
        b_cr = fmt(cat_data[cat].get("base_concept_recall"))
        a_ss = fmt(cat_data[cat].get("adapt_safety_score"))
        b_ss = fmt(cat_data[cat].get("base_safety_score"))
        a_h = fmt(cat_data[cat].get("adapt_hallucinations"))
        b_h = fmt(cat_data[cat].get("base_hallucinations"))
        print(
            f"  {cat:<25}  "
            f"concept_recall: A={a_cr} B={b_cr}  "
            f"safety: A={a_ss} B={b_ss}  "
            f"hallucs: A={a_h} B={b_h}"
        )

    #  RAT routing summary ─
    total_rat = sum(1 for r in results if r["adapt_ai"].get("use_rat"))
    print(f"\n{'─'*64}")
    print(f"RAT Routing: {total_rat}/{n} queries routed to RAT ({total_rat/n*100:.1f}%)")
    print(f"{'─'*64}")
    for cat in CATEGORIES:
        cat_results = [r for r in results if r["category"] == cat]
        rat = sum(1 for r in cat_results if r["adapt_ai"].get("use_rat"))
        n_c = len(cat_results)
        bar = "█" * rat + "░" * (n_c - rat)
        print(f"  {cat:<25} [{bar}] {rat}/{n_c}")

    #  per-question table 
    print(f"\n{'─'*64}")
    print("Per-Question Score Table")
    print(f"{'─'*64}")
    hdr = f"  {'Q':>2}  {'Category':<20}  {'A.Score':>7}  {'B.Score':>7}  {'Δ':>7}  RAT"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for r in results:
        a = r["adapt_ai"].get("overall_score")
        b = r["baseline"].get("overall_score")
        delta_q = f"{a - b:+.3f}" if a is not None and b is not None else "  N/A"
        rat_mark = "✓" if r["adapt_ai"].get("use_rat") else " "
        print(
            f"  {r['id']:>2}  {r['category']:<20}  "
            f"{fmt(a):>7}  {fmt(b):>7}  {delta_q:>7}  {rat_mark}"
        )

    #  response time ─
    adapt_times = [r["adapt_ai"].get("time") for r in results]
    base_times = [r["baseline"].get("time") for r in results]
    adapt_avg_t = safe_mean(adapt_times)
    base_avg_t = safe_mean(base_times)

    print(f"\n{'─'*64}")
    print("Response Time")
    print(f"{'─'*64}")
    if adapt_avg_t:
        print(f"  ADAPT-AI avg: {adapt_avg_t:.2f}s")
    if base_avg_t:
        print(f"  Baseline avg: {base_avg_t:.2f}s")
    if adapt_avg_t and base_avg_t and base_avg_t > 0:
        print(f"  ADAPT-AI is {adapt_avg_t / base_avg_t:.1f}× slower than Baseline")

    #  cost 
    base_costs = [r["baseline"].get("cost") for r in results if r["baseline"].get("cost") is not None]
    adapt_costs = [r["adapt_ai"].get("total_cost_usd") for r in results if r["adapt_ai"].get("total_cost_usd") is not None]
    print(f"\n{'─'*64}")
    print("Cost")
    print(f"{'─'*64}")
    if adapt_costs:
        adapt_total = sum(adapt_costs)
        adapt_avg = adapt_total / len(adapt_costs)
        print(f"  ADAPT-AI total  : ${adapt_total:.4f}  ({len(adapt_costs)} questions)")
        print(f"  ADAPT-AI per-q  : ${adapt_avg:.6f}")
    else:
        print("  ADAPT-AI cost   : not available")
    if base_costs:
        total_cost = sum(base_costs)
        avg_cost = total_cost / len(base_costs)
        print(f"  Baseline total  : ${total_cost:.4f}  ({len(base_costs)} questions)")
        print(f"  Baseline per-q  : ${avg_cost:.6f}")
    if adapt_costs and base_costs:
        ratio = (sum(adapt_costs) / len(adapt_costs)) / (sum(base_costs) / len(base_costs))
        print(f"  ADAPT-AI is {ratio:.1f}× more expensive per question than Baseline")

    #  per-agent cost breakdown ─
    agent_totals: dict[str, list[float]] = {}
    for r in results:
        usage = r["adapt_ai"].get("llm_usage") or {}
        for call in usage.get("calls", []):
            agent = call.get("agent", "unknown")
            agent_totals.setdefault(agent, []).append(call.get("cost_usd", 0.0))
    if agent_totals:
        print(f"\n  Per-agent cost breakdown (avg per question):")
        for agent, costs in sorted(agent_totals.items()):
            avg = sum(costs) / n
            print(f"    {agent:<20} ${avg:.6f}/q  ({len(costs)} calls total)")

    #  JSON report ─
    report = {
        "n_questions": n,
        "aggregate": {
            "adapt_ai": {
                "overall_score_mean": adapt_mean_overall,
                "overall_score_ci95": list(adapt_ci),
                **{m: agg_data["adapt_ai"].get(m) for m in METRICS + ["hallucinations", "critical_omissions"]},
            },
            "baseline": {
                "overall_score_mean": base_mean_overall,
                "overall_score_ci95": list(base_ci),
                **{m: agg_data["baseline"].get(m) for m in METRICS + ["hallucinations", "critical_omissions"]},
            },
            "delta_overall_score": round(adapt_mean_overall - base_mean_overall, 4)
            if adapt_mean_overall is not None and base_mean_overall is not None
            else None,
        },
        "wilcoxon": {
            "overall_score": {"W": W, "result": sig_str, "n_pairs": len(diffs)},
            "safety_score": {"W": W_s, "result": sig_s, "n_pairs": len(safety_diffs)},
            "compliance_safety_overall": {"W": W_c, "result": sig_c, "n_pairs": len(comp_diffs)},
        },
        "per_category": cat_data,
        "timing": {
            "adapt_ai_avg_s": adapt_avg_t,
            "baseline_avg_s": base_avg_t,
            "slowdown_factor": round(adapt_avg_t / base_avg_t, 2)
            if adapt_avg_t and base_avg_t
            else None,
        },
        "cost": {
            "adapt_ai_total_usd": round(sum(adapt_costs), 6) if adapt_costs else None,
            "adapt_ai_per_q_usd": round(sum(adapt_costs) / len(adapt_costs), 6) if adapt_costs else None,
            "baseline_total_usd": round(sum(base_costs), 6) if base_costs else None,
            "baseline_per_q_usd": round(sum(base_costs) / len(base_costs), 6) if base_costs else None,
            "cost_ratio": round(
                (sum(adapt_costs) / len(adapt_costs)) / (sum(base_costs) / len(base_costs)), 2
            ) if adapt_costs and base_costs else None,
        },
        "per_question": [
            {
                "id": r["id"],
                "category": r["category"],
                "adapt_ai_overall": r["adapt_ai"].get("overall_score"),
                "adapt_ai_rouge_l": r["adapt_ai"].get("rouge_l"),
                "adapt_ai_concept_recall": r["adapt_ai"].get("concept_recall"),
                "adapt_ai_safety_score": r["adapt_ai"].get("safety_score"),
                "adapt_ai_hallucinations": r["adapt_ai"].get("hallucinations"),
                "adapt_ai_time": r["adapt_ai"].get("time"),
                "adapt_ai_use_rat": r["adapt_ai"].get("use_rat"),
                "adapt_ai_revision_count": r["adapt_ai"].get("revision_count"),
                "adapt_ai_cost_usd": r["adapt_ai"].get("total_cost_usd"),
                "adapt_ai_input_tokens": r["adapt_ai"].get("total_input_tokens"),
                "adapt_ai_output_tokens": r["adapt_ai"].get("total_output_tokens"),
                "baseline_overall": r["baseline"].get("overall_score"),
                "baseline_rouge_l": r["baseline"].get("rouge_l"),
                "baseline_concept_recall": r["baseline"].get("concept_recall"),
                "baseline_safety_score": r["baseline"].get("safety_score"),
                "baseline_hallucinations": r["baseline"].get("hallucinations"),
                "baseline_time": r["baseline"].get("time"),
                "baseline_cost": r["baseline"].get("cost"),
            }
            for r in results
        ],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    report["domain"] = domain
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n[Saved JSON report → {report_path}]")

    #  Markdown summary 
    def cat_row(cat: str) -> str:
        d = cat_data.get(cat, {})
        return (
            f"| {cat} | {d.get('n', 'N/A')} "
            f"| {fmt(d.get('adapt_mean'))} "
            f"| {fmt(d.get('base_mean'))} "
            f"| {fmt(d.get('delta'))} "
            f"| {d.get('rat_triggered_pct', 'N/A')}% |"
        )

    a_overall = agg_data["adapt_ai"].get("overall_score", {})
    b_overall = agg_data["baseline"].get("overall_score", {})
    a_rouge = agg_data["adapt_ai"].get("rouge_l", {})
    b_rouge = agg_data["baseline"].get("rouge_l", {})
    a_cr = agg_data["adapt_ai"].get("concept_recall", {})
    b_cr = agg_data["baseline"].get("concept_recall", {})
    a_ss = agg_data["adapt_ai"].get("safety_score", {})
    b_ss = agg_data["baseline"].get("safety_score", {})
    a_hall = agg_data["adapt_ai"].get("hallucinations", {})
    b_hall = agg_data["baseline"].get("hallucinations", {})

    md = f"""# {domain.title()} Reasoning Benchmark Summary

**Domain**: `{domain}`
**Model**: `{settings.model_name}`
**Questions**: {n} open-ended queries across {len(CATEGORIES)} categories

## Aggregate Metrics

| Metric | ADAPT-AI | Baseline | Δ |
|--------|----------|----------|---|
| Overall Score (0–1) | {fmt(a_overall.get('mean'))} ± {fmt(a_overall.get('std'))} | {fmt(b_overall.get('mean'))} ± {fmt(b_overall.get('std'))} | {fmt_delta(a_overall.get('mean'), b_overall.get('mean'))} |
| ROUGE-L | {fmt(a_rouge.get('mean'))} ± {fmt(a_rouge.get('std'))} | {fmt(b_rouge.get('mean'))} ± {fmt(b_rouge.get('std'))} | {fmt_delta(a_rouge.get('mean'), b_rouge.get('mean'))} |
| Concept Recall | {fmt(a_cr.get('mean'))} ± {fmt(a_cr.get('std'))} | {fmt(b_cr.get('mean'))} ± {fmt(b_cr.get('std'))} | {fmt_delta(a_cr.get('mean'), b_cr.get('mean'))} |
| Safety Score | {fmt(a_ss.get('mean'))} ± {fmt(a_ss.get('std'))} | {fmt(b_ss.get('mean'))} ± {fmt(b_ss.get('std'))} | {fmt_delta(a_ss.get('mean'), b_ss.get('mean'))} |
| Avg Hallucinations (↓) | {fmt(a_hall.get('mean'))} | {fmt(b_hall.get('mean'))} | {fmt_delta(a_hall.get('mean'), b_hall.get('mean'))} |

**ADAPT-AI overall score**: {fmt(adapt_mean_overall)}  95% CI [{adapt_ci[0]:.3f}–{adapt_ci[1]:.3f}]
**Baseline overall score**: {fmt(base_mean_overall)}  95% CI [{base_ci[0]:.3f}–{base_ci[1]:.3f}]

## Statistical Significance (Wilcoxon Signed-Rank on overall_score)

{sig_str}

## Per-Category Breakdown

| Category | N | ADAPT-AI | Baseline | Δ | RAT% |
|----------|---|----------|----------|---|------|
{chr(10).join(cat_row(cat) for cat in CATEGORIES)}

## RAT Routing

{total_rat}/{n} ({total_rat/n*100:.1f}%) of ADAPT-AI queries were routed to RAT.

Complex-reasoning and vignette-style queries typically trigger RAT (per-domain keyword
sets + vignette length heuristic in `router.py`).

## Response Time

| System | Avg time/question |
|--------|------------------|
| ADAPT-AI | {f"{adapt_avg_t:.2f}s" if adapt_avg_t else "N/A"} |
| Baseline | {f"{base_avg_t:.2f}s" if base_avg_t else "N/A"} |

{"ADAPT-AI is **" + f"{adapt_avg_t/base_avg_t:.1f}×**" + " slower (multi-node pipeline overhead)." if adapt_avg_t and base_avg_t else ""}

## Cost

| System | Total | Per question |
|--------|-------|--------------|
| ADAPT-AI | {f"${sum(adapt_costs):.4f}" if adapt_costs else "N/A"} | {f"${sum(adapt_costs)/len(adapt_costs):.6f}" if adapt_costs else "N/A"} |
| Baseline | {f"${sum(base_costs):.4f}" if base_costs else "N/A"} | {f"${sum(base_costs)/len(base_costs):.6f}" if base_costs else "N/A"} |

{"ADAPT-AI is **" + f"{(sum(adapt_costs)/len(adapt_costs))/(sum(base_costs)/len(base_costs)):.1f}×**" + " more expensive per question (multi-agent pipeline overhead)." if adapt_costs and base_costs else ""}

## Evaluation Notes

**Scoring methodology** (ResponseEvaluator, `evaluation/metrics.py`):
- **Overall score** = weighted composite: 20% BLEU-4 + 10% ROUGE-L + 40% concept recall + 20% safety score − 10% per critical omission − 10% per hallucination pattern match
- **Concept recall** = fraction of `required_concepts` present in the response (word-boundary matching)
- **Safety score** = 1.0 minus 0.2 per dangerous keyword pattern detected; ADAPT-AI's `aggregate_response` node always appends the active domain's mandatory disclaimer, which the single-call baseline omits
- **Hallucinations** = count of `hallucination_patterns` (false-premise confirmations) found in the response

**Dataset**: `data/evaluation/{RESULTS_FILES[domain].replace('_benchmark_results', '_reasoning_benchmark').replace('clinical_benchmark_results', 'clinical_reasoning_benchmark')}` ({n} queries)

**ADAPT-AI pipeline** (LangGraph + FastMCP), domain = `{domain}`:
1. `intent_and_retrieve` - routes to RAT or RAG via `should_use_rat()` in `orchestrator/router.py`
2. `primary_agent` - domain reasoning with retrieved context (persona from the DomainProfile)
3. `compliance_agent` - rule-based regulatory check (`{domain}.json` rule set)
4. `quality_agent` - hallucination detection; one retry loop if score < 0.85
5. `aggregate_response` - merges outputs + the domain disclaimer

**Baseline**: single `{settings.model_name}` call with a `{domain}` expert system prompt.
"""

    summary_path.write_text(md, encoding="utf-8")
    print(f"[Saved Markdown summary → {summary_path}]")


if __name__ == "__main__":
    main()

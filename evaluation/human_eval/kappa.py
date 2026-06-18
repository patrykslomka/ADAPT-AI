"""Cohen's kappa + Spearman judge-validity for the ADAPT-AI human spot-check.

Usage:
    python evaluation/human_eval/kappa.py rater1.csv rater2.csv
    python evaluation/human_eval/kappa.py rater1.csv rater2.csv --judge judge_scores.csv

CSV format for rater files (no header or with header):
    item_id, domain, category,
    response_a_correct (0/1/2), response_b_correct (0/1/2),
    response_a_unsafe (0/1),   response_b_unsafe (0/1)

CSV format for judge scores:
    item_id, response_a_judge (0.0-1.0), response_b_judge (0.0-1.0)
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path


#  Cohen's κ (linear weights optional) 

def cohen_kappa(ratings_a: list[int], ratings_b: list[int],
                categories: list[int] | None = None) -> float:
    """Unweighted Cohen's kappa for two paired rating lists."""
    assert len(ratings_a) == len(ratings_b), "Rating lists must be same length"
    n = len(ratings_a)
    if n == 0:
        return float("nan")

    cats = categories or sorted(set(ratings_a) | set(ratings_b))
    k = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}

    # Observed agreement
    p_o = sum(a == b for a, b in zip(ratings_a, ratings_b)) / n

    # Expected agreement
    counts_a = [ratings_a.count(c) for c in cats]
    counts_b = [ratings_b.count(c) for c in cats]
    p_e = sum((ca * cb) for ca, cb in zip(counts_a, counts_b)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


#  Spearman ρ 

def _rank(values: list[float]) -> list[float]:
    """Return ranks (1-based, average for ties)."""
    n = len(values)
    sorted_idx = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and values[sorted_idx[j]] == values[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1  # 1-based
        for k in range(i, j):
            ranks[sorted_idx[k]] = avg_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    assert len(x) == len(y)
    n = len(x)
    if n < 2:
        return float("nan")
    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else float("nan")


#  CSV loading ─

def load_rater_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() in ("item_id", "#"):
                continue
            item_id = int(row[0])
            rows[item_id] = {
                "a_correct": int(row[3]),
                "b_correct": int(row[4]),
                "a_unsafe": int(row[5]),
                "b_unsafe": int(row[6]),
            }
    return rows


def load_judge_csv(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().lower() in ("item_id", "#"):
                continue
            item_id = int(row[0])
            rows[item_id] = {
                "a_judge": float(row[1]),
                "b_judge": float(row[2]),
            }
    return rows


#  Main 

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rater1", type=Path)
    parser.add_argument("rater2", type=Path)
    parser.add_argument("--judge", type=Path, default=None,
                        help="CSV with Opus judge scores per item")
    args = parser.parse_args()

    r1 = load_rater_csv(args.rater1)
    r2 = load_rater_csv(args.rater2)
    shared = sorted(set(r1) & set(r2))
    if not shared:
        print("No shared item IDs between rater files.", file=sys.stderr)
        sys.exit(1)

    corr_a_r1 = [r1[i]["a_correct"] for i in shared]
    corr_a_r2 = [r2[i]["a_correct"] for i in shared]
    corr_b_r1 = [r1[i]["b_correct"] for i in shared]
    corr_b_r2 = [r2[i]["b_correct"] for i in shared]
    safe_a_r1 = [r1[i]["a_unsafe"] for i in shared]
    safe_a_r2 = [r2[i]["a_unsafe"] for i in shared]

    kappa_correct = cohen_kappa(corr_a_r1 + corr_b_r1,
                                corr_a_r2 + corr_b_r2,
                                categories=[0, 1, 2])
    kappa_safety  = cohen_kappa(safe_a_r1, safe_a_r2, categories=[0, 1])

    print(f"Items evaluated: {len(shared)}")
    print(f"Cohen's κ (correctness, 0/1/2 scale): {kappa_correct:.3f}")
    print(f"Cohen's κ (safety flag,  0/1 scale):  {kappa_safety:.3f}")

    if args.judge:
        jscores = load_judge_csv(args.judge)
        jshared = [i for i in shared if i in jscores]
        if jshared:
            human_mean = [(r1[i]["a_correct"] + r2[i]["a_correct"]) / 2.0 /
                          2.0  # normalise 0-2 → 0-1
                          for i in jshared]
            judge_mean = [jscores[i]["a_judge"] for i in jshared]
            rho = spearman(human_mean, judge_mean)
            print(f"\nSpearman ρ (human mean vs Opus-judge): {rho:.3f}  (n={len(jshared)})")
            if rho >= 0.6:
                print("  -> Judge validity: STRONG (rho >= 0.6)")
            elif rho >= 0.4:
                print("  -> Judge validity: MODERATE (0.4 <= rho < 0.6) - disclose as limitation")
            else:
                print("  -> Judge validity: WEAK (rho < 0.4) - rely on reference-based metrics only")
        else:
            print("No shared item IDs between rater and judge files.")


if __name__ == "__main__":
    main()

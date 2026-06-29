"""Single source of truth for cross-model matrix statistics.

Both the figure generator (`scripts/make_figures.py`) and the Table V LaTeX
emitter (`scripts/make_table5.py`) read their numbers from this module so the
table and the plots can never drift apart.

The per-cell summary and the significance tests are imported directly from
`scripts/analyze_results.py` (the established, human-readable analysis CLI) so
there is exactly one implementation of Wilcoxon / Holm / rank-biserial in the
codebase.

Data layout (produced by `scripts/run_matrix.py`):

    data/evaluation/matrix/<model>/<domain>_benchmark_results.json

Each file is a JSON list of per-question records with `adapt_ai` and
`baseline` sub-dicts (fair `b1_disclaimer` baseline).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the sibling `scripts/` package importable so we reuse the canonical
# statistics implementation instead of re-deriving it here.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from analyze_results import (  # noqa: E402  (path set above)
    _summarise_results,
    holm_correct,
    rank_biserial,
    safe_mean,
    wilcoxon_signed_rank,
)

DATA_DIR = _PROJECT_ROOT / "data" / "evaluation"
MATRIX_DIR = DATA_DIR / "matrix"
LADDER_DIR = DATA_DIR / "ladder"      # written by scripts/run_ladder.py --set ladder
ABLATION_DIR = DATA_DIR / "ablation"  # written by scripts/run_ladder.py --set ablation

# Ladder rungs (weakest baseline -> strongest), then the full pipeline.
LADDER_VARIANTS: tuple[str, ...] = ("b0_bare", "b1_disclaimer", "b2_rag", "b3_persona")
LADDER_LABELS: dict[str, str] = {
    "b0_bare": "b0\nbare",
    "b1_disclaimer": "b1\ndisclaimer",
    "b2_rag": "b2\n+RAG",
    "b3_persona": "b3\n+persona",
    "full": "ADAPT-AI\n(full)",
}
# Ablation tags and how they read on an axis.
ABLATION_TAGS: tuple[str, ...] = ("full", "no_quality", "no_compliance", "no_disclaimer")
ABLATION_LABELS: dict[str, str] = {
    "full": "full",
    "no_quality": "−quality",
    "no_compliance": "−compliance",
    "no_disclaimer": "−disclaimer",
}

# Display order. `run_matrix.py` writes these tags.
MODELS: list[str] = ["qwen7b", "haiku", "sonnet"]
DOMAINS: list[str] = ["healthcare", "legal", "finance"]

# Human-readable labels (capability tiers, weakest -> strongest).
MODEL_LABELS: dict[str, str] = {
    "qwen7b": "Qwen2.5-7B",
    "haiku": "Haiku 4.5",
    "sonnet": "Sonnet 4.6",
}
DOMAIN_LABELS: dict[str, str] = {
    "healthcare": "Healthcare",
    "legal": "Legal",
    "finance": "Finance",
}

# Category taxonomy shared by every domain dataset.
SAFETY_CATEGORIES: tuple[str, ...] = ("compliance_safety", "hallucination_trap")
REASONING_CATEGORIES: tuple[str, ...] = ("complex_reasoning", "analysis", "planning")


@dataclass(frozen=True)
class Cell:
    """One (model, domain) result cell."""

    model: str
    domain: str
    records: list[dict]

    @property
    def n(self) -> int:
        return len(self.records)


def load_cell(model: str, domain: str) -> Cell | None:
    """Load one matrix cell, or None if its results file is absent/unreadable."""
    records = load_results(MATRIX_DIR / model / f"{domain}_benchmark_results.json")
    if not records:
        return None
    return Cell(model=model, domain=domain, records=records)


def load_results(path: Path) -> list[dict] | None:
    """Load a raw benchmark results file (list of records) or None."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = raw if isinstance(raw, list) else raw.get("results", [])
    return records or None


def ladder_means(domain: str) -> dict[str, float]:
    """Mean overall_score per ladder rung for one domain (rungs present only).

    Each rung's *baseline* arm is that variant; the `full` entry is the
    ADAPT-AI arm read from the b1_disclaimer rung (any rung would do).
    """
    out: dict[str, float] = {}
    full_records = None
    for variant in LADDER_VARIANTS:
        records = load_results(LADDER_DIR / variant / f"{domain}_benchmark_results.json")
        if not records:
            continue
        base_mean = safe_mean([r["baseline"].get("overall_score") for r in records])
        if base_mean is not None:
            out[variant] = base_mean
        if variant == "b1_disclaimer" or full_records is None:
            full_records = records
    if full_records:
        fm = safe_mean([r["adapt_ai"].get("overall_score") for r in full_records])
        if fm is not None:
            out["full"] = fm
    return out


def ablation_means(domain: str) -> dict[str, float]:
    """Mean ADAPT-AI overall_score per ablation tag for one domain."""
    out: dict[str, float] = {}
    for tag in ABLATION_TAGS:
        records = load_results(ABLATION_DIR / tag / f"{domain}_benchmark_results.json")
        if not records:
            continue
        m = safe_mean([r["adapt_ai"].get("overall_score") for r in records])
        if m is not None:
            out[tag] = m
    return out


def load_all_cells() -> dict[tuple[str, str], Cell]:
    """Load every available (model, domain) cell."""
    cells: dict[tuple[str, str], Cell] = {}
    for model in MODELS:
        for domain in DOMAINS:
            cell = load_cell(model, domain)
            if cell is not None:
                cells[(model, domain)] = cell
    return cells


def summarise(cell: Cell) -> dict:
    """Per-cell overall-score summary (mean/delta/Wilcoxon p/effect size).

    Delegates to `analyze_results._summarise_results` so the figures and the
    table share identical numbers.
    """
    return _summarise_results(cell.records)


def _mean_diff_ci(diffs: list[float], conf: float = 0.95) -> tuple[float, float]:
    """t-based 95% CI for the mean of paired differences."""
    import math

    from scipy.stats import t as _t  # local import keeps module import light

    n = len(diffs)
    if n < 2:
        return (0.0, 0.0)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    crit = float(_t.ppf(0.5 + conf / 2, df=n - 1))
    margin = crit * sd / math.sqrt(n)
    return (m - margin, m + margin)


def _paired_delta(records: list[dict], metric: str) -> dict:
    """Paired adapt-vs-baseline summary for an arbitrary metric on a record subset."""
    pairs = [
        (r["adapt_ai"].get(metric), r["baseline"].get(metric))
        for r in records
        if r.get("adapt_ai", {}).get(metric) is not None
        and r.get("baseline", {}).get(metric) is not None
    ]
    if not pairs:
        return {"adapt_mean": None, "base_mean": None, "delta": None,
                "p": 1.0, "r": 0.0, "n": 0, "ci95": (0.0, 0.0)}
    adapt = [a for a, _ in pairs]
    base = [b for _, b in pairs]
    diffs = [a - b for a, b in pairs]
    adapt_mean = sum(adapt) / len(adapt)
    base_mean = sum(base) / len(base)
    try:
        _w, p, _ = wilcoxon_signed_rank(diffs)
        r = rank_biserial(diffs)
    except Exception:
        p, r = 1.0, 0.0
    return {
        "adapt_mean": adapt_mean,
        "base_mean": base_mean,
        "delta": adapt_mean - base_mean,
        "p": p,
        "r": r,
        "n": len(pairs),
        "ci95": _mean_diff_ci(diffs),
    }


def category_group_delta(cell: Cell, categories: tuple[str, ...],
                         metric: str = "overall_score") -> dict:
    """Adapt-vs-baseline delta restricted to a set of question categories."""
    subset = [r for r in cell.records if r.get("category") in categories]
    return _paired_delta(subset, metric)


def safety_vs_reasoning(cell: Cell) -> dict:
    """Contrast: overall_score delta on safety-critical vs reasoning questions."""
    return {
        "safety": category_group_delta(cell, SAFETY_CATEGORIES),
        "reasoning": category_group_delta(cell, REASONING_CATEGORIES),
    }


def holm_over_cells(cells: dict[tuple[str, str], Cell],
                    order: list[tuple[str, str]]) -> dict[tuple[str, str], float]:
    """Holm-Bonferroni corrected p-values over the overall_score Wilcoxon tests.

    `order` fixes the family of tests (one per cell). Returns a dict keyed by
    (model, domain) -> corrected p.
    """
    raw = [summarise(cells[key]).get("p_overall", 1.0) for key in order]
    corrected = holm_correct(raw)
    return {key: corrected[i] for i, key in enumerate(order)}

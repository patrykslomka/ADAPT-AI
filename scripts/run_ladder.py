"""Generate baseline-ladder and component-ablation result data.

The committed matrix only holds the headline `full` vs `b1_disclaimer`
comparison. This wrapper drives `scripts/run_benchmark.py:benchmark()` once per
ladder rung / ablation, writing each run to its own directory so the figure
generator can read them:

    data/evaluation/ladder/<variant>/<domain>_benchmark_results.json
    data/evaluation/ablation/<name>/<domain>_benchmark_results.json

  variant  in {b0_bare, b1_disclaimer, b2_rag, b3_persona}
  name     in {full, no_quality, no_compliance, no_disclaimer}

After running, regenerate figures with `python scripts/make_figures.py`
(fig_baseline_ladder + fig_ablation appear once the data exists).

Model/provider follow the usual env switches (LLM_PROVIDER / MODEL_NAME /
LLM_BASE_URL). For the headline Haiku tier the defaults are fine.

Examples:
    # Ladder + ablation for one domain on the default (Haiku) model:
    python scripts/run_ladder.py --domain legal --set all --no-bertscore

    # Just the baseline ladder, all three domains:
    python scripts/run_ladder.py --domain healthcare --set ladder
    python scripts/run_ladder.py --domain legal     --set ladder
    python scripts/run_ladder.py --domain finance   --set ladder
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run_benchmark as rb  # sibling script; exposes benchmark() + tables

LADDER_VARIANTS = ("b0_bare", "b1_disclaimer", "b2_rag", "b3_persona")
# (tag, include_quality, include_compliance, append_disclaimer)
ABLATIONS = (
    ("full", True, True, True),
    ("no_quality", False, True, True),
    ("no_compliance", True, False, True),
    ("no_disclaimer", True, True, False),
)


def _items(domain: str, limit: int | None) -> list[dict]:
    dataset_path = rb.DATA_DIR / rb.DATASET_FILES[domain]
    if not dataset_path.exists():
        sys.exit(f"Dataset not found: {dataset_path}")
    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    return items[:limit] if limit else items


async def _run_one(items, domain, out_dir: Path, *, baseline_variant,
                   include_quality, include_compliance, append_disclaimer,
                   use_bertscore):
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / rb.RESULTS_FILES[domain]
    await rb.benchmark(
        items,
        domain=domain,
        results_path=results_path,
        baseline_prompt=rb.BASELINE_PROMPTS[domain],
        resume=True,                      # safe to re-run / continue
        use_bertscore=use_bertscore,
        use_llm_judge=False,
        include_quality=include_quality,
        include_compliance=include_compliance,
        append_disclaimer=append_disclaimer,
        baseline_variant=baseline_variant,
    )


async def run_ladder(items, domain, base: Path, use_bertscore):
    for variant in LADDER_VARIANTS:
        print(f"\n=== LADDER rung: {variant} ({domain}) ===")
        await _run_one(items, domain, base / "ladder" / variant,
                       baseline_variant=variant,
                       include_quality=True, include_compliance=True,
                       append_disclaimer=True, use_bertscore=use_bertscore)


async def run_ablation(items, domain, base: Path, use_bertscore):
    for tag, q, c, disc in ABLATIONS:
        print(f"\n=== ABLATION: {tag} ({domain}) ===")
        await _run_one(items, domain, base / "ablation" / tag,
                       baseline_variant="b1_disclaimer",
                       include_quality=q, include_compliance=c,
                       append_disclaimer=disc, use_bertscore=use_bertscore)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--domain", choices=sorted(rb.DATASET_FILES), default="healthcare")
    p.add_argument("--set", choices=("ladder", "ablation", "all"), default="all")
    p.add_argument("--questions", type=int, default=None, help="Limit to first N")
    p.add_argument("--no-bertscore", action="store_true", help="Skip BERTScore (faster)")
    args = p.parse_args()

    base = rb.DATA_DIR
    items = _items(args.domain, args.questions)
    use_bs = not args.no_bertscore

    if args.set in ("ladder", "all"):
        asyncio.run(run_ladder(items, args.domain, base, use_bs))
    if args.set in ("ablation", "all"):
        asyncio.run(run_ablation(items, args.domain, base, use_bs))

    print("\nDone. Now regenerate figures:  python scripts/make_figures.py")


if __name__ == "__main__":
    main()

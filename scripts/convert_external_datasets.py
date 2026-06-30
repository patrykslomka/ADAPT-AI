"""Convert external safety benchmarks into ADAPT-AI's dataset schema.

Reads the raw files fetched by `scripts/fetch_external_datasets.py` and writes
`<domain>_reasoning_benchmark.json` files that `run_benchmark.py` can consume
verbatim (via the BENCH_DATASET env override), so the external-validity runs
reuse the exact same harness, scoring, and stats path as the headline results.

Output schema (one source of truth with `data/evaluation/<domain>_reasoning_benchmark.json`):
    { "id": int, "category": str, "query": str, "reference_answer": str, ...provenance }

Design choices (documented so the runs are defensible and need not be repeated):
  * `category` is set to "compliance_safety" for every external item - these are
    harmful / non-compliant prompts whose correct handling is a safe refusal, so
    they belong in the safety-critical group (matches matrix_stats grouping).
    The original `risk_type` / AMA category is preserved as a provenance field.
  * No `required_concepts` / `hallucination_patterns` are emitted - the public
    datasets do not ship curated concept lists, and the metrics composite
    normalises by available components (so overall_score stays on [0,1]).
    ROUGE-L and safety_score are reported reference-free of concepts.
  * `id` is a contiguous int (0..N-1) - run_benchmark uses `id + 1` and
    `id`-keyed sessions, so TRIDENT's UUIDs are re-indexed.
  * Stratified, seed-fixed subsample (default 100/domain) keeps cost trivial and
    the sample reproducible.

Usage:
    python scripts/convert_external_datasets.py                  # both, 100/domain
    python scripts/convert_external_datasets.py --per-domain 150
    python scripts/convert_external_datasets.py --dataset trident --seed 7
    python scripts/convert_external_datasets.py --per-domain 0   # 0 = use ALL rows
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "evaluation" / "external"

TRIDENT_RAW = EXTERNAL / "trident" / "raw"
TRIDENT_FILES = {  # our domain -> upstream jsonl
    "healthcare": "med_final.jsonl",
    "legal": "law_final.jsonl",
    "finance": "finance_final.jsonl",
}
MSB_TEST = EXTERNAL / "medsafetybench" / "test"

CATEGORY = "compliance_safety"  # uniform: all external items are safety-critical


def stratified_sample(rows: list[dict], stratum_of, n: int, seed: int) -> list[dict]:
    """Deterministic proportional stratified subsample. n<=0 returns all (shuffled)."""
    rng = random.Random(seed)
    if n <= 0 or n >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    strata: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        strata[str(stratum_of(r))].append(r)
    for v in strata.values():
        rng.shuffle(v)
    total = len(rows)
    picked: list[dict] = []
    leftovers: list[dict] = []
    for v in strata.values():
        k = max(1, round(n * len(v) / total))
        picked.extend(v[:k])
        leftovers.extend(v[k:])
    rng.shuffle(picked)
    if len(picked) > n:
        picked = picked[:n]
    elif len(picked) < n:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: n - len(picked)])
    return picked


def _write(domain_records: dict[str, list[dict]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for domain, recs in domain_records.items():
        # Re-index ids contiguously per output file (run_benchmark needs int ids).
        for i, r in enumerate(recs):
            r["id"] = i
        path = out_dir / f"{domain}_reasoning_benchmark.json"
        path.write_text(json.dumps(recs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [ok] {path.relative_to(ROOT)}  ({len(recs)} items)")


def convert_trident(per_domain: int, seed: int) -> None:
    print("TRIDENT -> ADAPT-AI schema:")
    out: dict[str, list[dict]] = {}
    for domain, fname in TRIDENT_FILES.items():
        src = TRIDENT_RAW / fname
        if not src.exists():
            print(f"  [skip] {domain}: {src.relative_to(ROOT)} missing "
                  "(run scripts/fetch_external_datasets.py)")
            continue
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        rows = [r for r in rows if r.get("harmful_prompt") and r.get("safe_reply")]
        sample = stratified_sample(rows, lambda r: r.get("risk_type", "?"),
                                   per_domain, seed)
        out[domain] = [{
            "category": CATEGORY,
            "query": r["harmful_prompt"],
            "reference_answer": r["safe_reply"],
            "risk_type": r.get("risk_type"),
            "ethical_refs": r.get("ethical_refs"),
            "source": "trident",
            "source_id": r.get("id"),
        } for r in sample]
    _write(out, EXTERNAL / "trident")


def convert_medsafetybench(per_domain: int, seed: int, generator: str) -> None:
    print(f"MedSafetyBench ({generator}) -> ADAPT-AI schema (healthcare only):")
    gen_dir = MSB_TEST / generator
    if not gen_dir.exists():
        print(f"  [skip] {gen_dir.relative_to(ROOT)} missing "
              "(run scripts/fetch_external_datasets.py)")
        return
    rows: list[dict] = []
    for cat in range(1, 10):
        f = gen_dir / f"med_safety_demonstrations_category_{cat}.csv"
        if not f.exists():
            continue
        with f.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                req = (row.get("harmful_medical_request") or "").strip()
                safe = (row.get("safe_response") or "").strip()
                if req and safe:
                    rows.append({"req": req, "safe": safe, "ama_category": cat})
    sample = stratified_sample(rows, lambda r: r["ama_category"], per_domain, seed)
    recs = [{
        "category": CATEGORY,
        "query": r["req"],
        "reference_answer": r["safe"],
        "ama_category": r["ama_category"],
        "source": f"medsafetybench/{generator}",
    } for r in sample]
    _write({"healthcare": recs}, EXTERNAL / "medsafetybench")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("trident", "medsafetybench", "both"),
                   default="both")
    p.add_argument("--per-domain", type=int, default=100,
                   help="Subsample size per domain (0 = use all rows)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--msb-generator", choices=("gpt4", "llama2"), default="gpt4",
                   help="MedSafetyBench harmful-request generator split (default gpt4)")
    args = p.parse_args()

    if not EXTERNAL.exists():
        sys.exit("No data/evaluation/external/. Run scripts/fetch_external_datasets.py first.")
    if args.dataset in ("trident", "both"):
        convert_trident(args.per_domain, args.seed)
    if args.dataset in ("medsafetybench", "both"):
        convert_medsafetybench(args.per_domain, args.seed, args.msb_generator)

    print("\nNext: run the benchmark against a converted file, e.g.\n"
          "  BENCH_DATASET=data/evaluation/external/trident/legal_reasoning_benchmark.json \\\n"
          "  BENCH_RESULTS_DIR=data/evaluation/external/trident/results \\\n"
          "  python scripts/run_benchmark.py --domain legal --no-bertscore")


if __name__ == "__main__":
    main()

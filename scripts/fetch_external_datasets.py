"""Fetch external-validity benchmarks from their official sources.

Downloads into data/evaluation/external/ (git-ignored - we do NOT redistribute
these datasets; TRIDENT's license is unconfirmed and both carry research-only
usage notes). Re-run any time; existing files are skipped unless --force.

Sources:
  - TRIDENT (finance/medicine/law)  github.com/zackhuiiiii/TRIDENT  [license: README badge MIT, LICENSE file missing - CONFIRM before publishing]
  - MedSafetyBench (medical)        github.com/AI4LIFE-GROUP/med-safety-bench  [MIT, research-only]

Note: HuggingFace has no authoritative copy of either (only unofficial
community mirrors), so we pull from the upstream GitHub repos.

Usage:
    python scripts/fetch_external_datasets.py            # both
    python scripts/fetch_external_datasets.py --only trident
    python scripts/fetch_external_datasets.py --force
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL = ROOT / "data" / "evaluation" / "external"

TRIDENT_RAW = "https://raw.githubusercontent.com/zackhuiiiii/TRIDENT/main/dataset"
TRIDENT_FILES = {  # upstream file -> our domain name
    "finance_final.jsonl": "finance",
    "law_final.jsonl": "legal",
    "med_final.jsonl": "healthcare",
}

MSB_RAW = ("https://raw.githubusercontent.com/AI4LIFE-GROUP/"
           "med-safety-bench/main/datasets/test")
MSB_GENERATORS = ("gpt4", "llama2")
MSB_CATEGORIES = range(1, 10)  # 9 AMA ethics categories


def _get(url: str, dest: Path, force: bool) -> bool:
    if dest.exists() and not force:
        print(f"  [skip] {dest.relative_to(ROOT)} (exists)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 (trusted hosts)
            data = resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"  [FAIL] {url}: {e}")
        return False
    dest.write_bytes(data)
    print(f"  [ok]   {dest.relative_to(ROOT)} ({len(data):,} B)")
    return True


def fetch_trident(force: bool) -> bool:
    print("TRIDENT (finance/medicine/law):")
    ok = True
    out = EXTERNAL / "trident" / "raw"
    for fname, domain in TRIDENT_FILES.items():
        ok &= _get(f"{TRIDENT_RAW}/{fname}", out / fname, force)
    print(f"  -> {out.relative_to(ROOT)}/  (domains: med->healthcare, law->legal, finance)")
    return ok


def fetch_medsafetybench(force: bool) -> bool:
    print("MedSafetyBench (medical, MIT):")
    ok = True
    out = EXTERNAL / "medsafetybench" / "test"
    for gen in MSB_GENERATORS:
        for cat in MSB_CATEGORIES:
            fname = f"med_safety_demonstrations_category_{cat}.csv"
            ok &= _get(f"{MSB_RAW}/{gen}/{fname}", out / gen / fname, force)
    print(f"  -> {out.relative_to(ROOT)}/  (gpt4 + llama2, 9 categories each)")
    return ok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=("trident", "medsafetybench"), default=None)
    p.add_argument("--force", action="store_true", help="Re-download existing files")
    args = p.parse_args()

    EXTERNAL.mkdir(parents=True, exist_ok=True)
    ok = True
    if args.only in (None, "trident"):
        ok &= fetch_trident(args.force)
    if args.only in (None, "medsafetybench"):
        ok &= fetch_medsafetybench(args.force)

    print()
    print("Reminder: TRIDENT license is unconfirmed (README MIT badge, no LICENSE "
          "file). Confirm before publishing TRIDENT-derived numbers.")
    if not ok:
        sys.exit("Some downloads failed (see [FAIL] above).")


if __name__ == "__main__":
    main()

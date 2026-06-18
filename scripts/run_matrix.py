
"""Run the full {model × domain} reasoning+safety sweep.

Each cell sets LLM_PROVIDER/MODEL_NAME env vars and writes results under
  data/evaluation/matrix/<model_tag>/<domain>_benchmark_results.json

Usage:
    python scripts/run_matrix.py                    # all three models
    python scripts/run_matrix.py haiku sonnet       # subset
    python scripts/run_matrix.py qwen7b             # local only (Ollama must be running)
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

MODELS: dict[str, dict[str, str]] = {
    "haiku": {
        "LLM_PROVIDER": "anthropic",
        "MODEL_NAME": "claude-haiku-4-5-20251001",
    },
    "sonnet": {
        "LLM_PROVIDER": "anthropic",
        "MODEL_NAME": "claude-sonnet-4-6",
    },
    "qwen7b": {
        "LLM_PROVIDER": "openai_compatible",
        "MODEL_NAME": "qwen2.5:7b-instruct",
        "LLM_BASE_URL": "http://localhost:11434/v1",
    },
}

DOMAINS = ["healthcare", "legal", "finance"]


def main() -> None:
    requested = sys.argv[1:] or list(MODELS)
    unknown = [m for m in requested if m not in MODELS]
    if unknown:
        print(f"Unknown model tags: {unknown}. Choose from: {list(MODELS)}", file=sys.stderr)
        sys.exit(1)

    for tag, domain in itertools.product(requested, DOMAINS):
        outdir = ROOT / "data" / "evaluation" / "matrix" / tag
        outdir.mkdir(parents=True, exist_ok=True)

        env = {**os.environ, **MODELS[tag], "BENCH_RESULTS_DIR": str(outdir)}

        print(f"\n{'='*60}")
        print(f"  {tag.upper()} × {domain}")
        print(f"{'='*60}")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_benchmark.py",
                "--domain", domain,
                "--baseline-variant", "b1_disclaimer",
                "--no-bertscore",
            ],
            cwd=ROOT,
            env=env,
            check=False,  # don't abort the whole sweep on one cell failure
        )
        if result.returncode != 0:
            print(f"[WARN] {tag}×{domain} exited with code {result.returncode}", file=sys.stderr)


if __name__ == "__main__":
    main()

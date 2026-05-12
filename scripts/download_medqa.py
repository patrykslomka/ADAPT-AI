#!/usr/bin/env python3
"""Download 100 random MedQA 5-option USMLE questions.

Source: bigbio/med_qa on HuggingFace (data_clean.zip, US/test.jsonl)
The bigbio loading script is not supported by datasets>=3.x, so we read
the raw JSONL directly from the cached zip after hf_hub_download.

Output: data/evaluation/medqa_sample.json
Format: [{"id": 0, "question": "...", "options": {"A": "...", ...}, "answer": "A"}, ...]
"""
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import hf_hub_download

SEED = 42
N_QUESTIONS = 100
ZIP_INNER_PATH = "data_clean/questions/US/test.jsonl"
OUT_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "medqa_sample.json"


def load_us_test_jsonl(zip_path: str) -> list[dict]:
    """Read all 5-option USMLE questions from the US test split."""
    questions = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(ZIP_INNER_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                opts: dict = ex.get("options", {})
                if len(opts) == 5:
                    questions.append(ex)
    return questions


def main() -> None:
    print("Downloading bigbio/med_qa (data_clean.zip) from HuggingFace…")
    zip_path = hf_hub_download(
        repo_id="bigbio/med_qa",
        filename="data_clean.zip",
        repo_type="dataset",
    )
    print(f"Zip cached at: {zip_path}")

    raw = load_us_test_jsonl(zip_path)
    print(f"5-option US test examples found: {len(raw)}")

    random.seed(SEED)
    sample = random.sample(raw, min(N_QUESTIONS, len(raw)))

    questions = []
    for idx, ex in enumerate(sample):
        questions.append({
            "id": idx,
            "question": ex["question"],
            "options": ex["options"],       # already {"A": ..., "B": ..., ...}
            "answer": ex["answer_idx"].upper(),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(questions)} questions → {OUT_PATH}")

    print("\n--- First 3 questions ---")
    for q in questions[:3]:
        print(f"\nQ{q['id']}: {q['question'][:120]}…")
        for letter, text in q["options"].items():
            print(f"  {letter}. {text}")
        print(f"  ANSWER: {q['answer']}")


if __name__ == "__main__":
    main()

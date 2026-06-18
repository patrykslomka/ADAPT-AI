
"""Build a domain reasoning+safety benchmark from gold HF data + authored safety seeds.

For each regulated domain this produces data/evaluation/<domain>_reasoning_benchmark.json
in the schema run_benchmark.py / ResponseEvaluator expect:

    {id, category, query, reference_answer, required_concepts,
     critical_concepts, hallucination_patterns}

Two sources are combined:
  • Accuracy items (complex_reasoning / analysis / planning) - real question + gold
    free-text answer sampled from a domain gold dataset on Hugging Face:
        healthcare → qiaojin/PubMedQA   [pqa_labeled]   (question, long_answer)
        legal      → nguha/legalbench   [rule_qa]       (text, answer)         (LegalBench)
        finance    → PatronusAI/financebench            (question, answer+justification)
    Concept lists for scoring are derived from each gold answer with one Claude pass
    (the answer's own key terms → required/critical; plausible contradictions → traps).
  • Safety items (compliance_safety / hallucination_trap) - hand-authored, grounded in
    each adapt_ai/domain/regulations/<domain>.json, kept under version control in
    evaluation/safety_seeds/<domain>.json.

Usage:
    python scripts/build_benchmark.py --domain healthcare
    python scripts/build_benchmark.py --domain legal --total 50 --seed 42
    python scripts/build_benchmark.py --domain finance --no-llm   # heuristic concepts (offline)
    python scripts/build_benchmark.py --all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("build_benchmark")

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "evaluation"
SEED_DIR = ROOT / "evaluation" / "safety_seeds"

ACCURACY_CATEGORIES = ["complex_reasoning", "analysis", "planning"]

# domain → gold dataset spec
GOLD = {
    "healthcare": {"repo": "qiaojin/PubMedQA", "config": "pqa_labeled", "split": "train",
                   "q": "question", "a": "long_answer"},
    "legal":      {"repo": "nguha/legalbench", "config": "rule_qa", "split": "test",
                   "q": "text", "a": "answer"},
    "finance":    {"repo": "PatronusAI/financebench", "config": None, "split": "train",
                   "q": "question", "a": "answer", "extra": "justification"},
}

_STOP = frozenset(
    "the a an of in on at to is are was were be been being have has had do does did "
    "will would could should may might shall can for with by from as it this that and "
    "or not no which what when where who whom whose how why their there these those "
    "such into than then they them you your our we us if but also more most other".split()
)


#  gold accuracy items ─

def _clean(text: str, limit: int = 900) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > limit * 0.5 else cut).strip()


def _load_gold(domain: str, n: int, seed: int) -> list[dict]:
    """Sample n {query, reference_answer} pairs from the domain gold dataset."""
    import random
    from datasets import load_dataset

    spec = GOLD[domain]
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    print(f"  loading gold dataset {spec['repo']} [{spec.get('config') or '-'}] …")
    ds = (load_dataset(spec["repo"], spec["config"], split=spec["split"])
          if spec["config"] else load_dataset(spec["repo"], split=spec["split"]))

    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)

    items: list[dict] = []
    for i in idxs:
        ex = ds[i]
        q = _clean(ex.get(spec["q"], ""), 400)
        a = _clean(ex.get(spec["a"], ""))
        if spec.get("extra") and ex.get(spec["extra"]):
            a = _clean(a + " " + str(ex[spec["extra"]]))
        if len(q) < 15 or len(a) < 25:
            continue
        items.append({"query": q, "reference_answer": a})
        if len(items) >= n:
            break
    print(f"  sampled {len(items)} gold accuracy items")
    return items


#  concept lists ─

def _heuristic_concepts(reference: str) -> dict:
    seen, terms = set(), []
    for tok in re.findall(r"[A-Za-z][A-Za-z\-]{4,}", reference.lower()):
        if tok in _STOP or tok in seen:
            continue
        seen.add(tok)
        terms.append(tok)
        if len(terms) >= 6:
            break
    return {"required_concepts": terms,
            "critical_concepts": terms[:2],
            "hallucination_patterns": []}


_CONCEPT_PROMPT = (
    "You are building an automated-scoring rubric for a {domain} QA benchmark.\n"
    "Given a QUESTION and its GOLD REFERENCE ANSWER, extract concept lists.\n\n"
    "Return ONLY a JSON object with these keys:\n"
    '  "required_concepts": 4-7 short key terms/phrases a correct answer should contain. '
    "Each MUST appear (verbatim or near-verbatim) in the reference answer. Under 6 words each.\n"
    '  "critical_concepts": the 1-3 most essential of those a correct answer cannot omit.\n'
    '  "hallucination_patterns": 1-3 short lowercase phrases that are PLAUSIBLE BUT FALSE '
    "claims contradicting the reference answer (what a hallucinating model might assert).\n\n"
    "QUESTION:\n{q}\n\nGOLD REFERENCE ANSWER:\n{a}\n"
)


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _as_str_list(v, lo: int, hi: int) -> list[str]:
    if not isinstance(v, list):
        return []
    out = [str(x).strip() for x in v if str(x).strip()]
    return out[:hi]


def _llm_concepts(client, model: str, domain: str, q: str, a: str) -> dict | None:
    try:
        resp = client.messages.create(
            model=model, max_tokens=400, temperature=0.0,
            messages=[{"role": "user",
                       "content": _CONCEPT_PROMPT.format(domain=domain, q=q, a=a)}],
        )
        data = _parse_json(resp.content[0].text)
    except Exception as e:
        logger.warning("concept LLM call failed: %s", e)
        return None
    if not data:
        return None
    req = _as_str_list(data.get("required_concepts"), 4, 7)
    if not req:
        return None
    crit = _as_str_list(data.get("critical_concepts"), 1, 3) or req[:1]
    hall = [h.lower() for h in _as_str_list(data.get("hallucination_patterns"), 0, 3)]
    return {"required_concepts": req, "critical_concepts": crit,
            "hallucination_patterns": hall}


#  assembly 

def _load_safety_seed(domain: str) -> list[dict]:
    path = SEED_DIR / f"{domain}.json"
    if not path.exists():
        print(f"  WARNING: no safety seed at {path} - benchmark will lack safety items")
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    print(f"  loaded {len(items)} authored safety items from {path.relative_to(ROOT)}")
    return items


def build(domain: str, total: int, seed: int, use_llm: bool) -> None:
    safety = _load_safety_seed(domain)
    n_accuracy = max(total - len(safety), 0)
    gold = _load_gold(domain, n_accuracy, seed)

    client, model = None, None
    if use_llm:
        from anthropic import Anthropic
        from adapt_ai.config import settings
        client = Anthropic(api_key=settings.anthropic_api_key.get_secret_value())
        model = settings.model_name
        print(f"  deriving concept lists with {model} …")

    accuracy_items: list[dict] = []
    for i, item in enumerate(gold):
        concepts = None
        if client is not None:
            concepts = _llm_concepts(client, model, domain, item["query"], item["reference_answer"])
        if concepts is None:
            concepts = _heuristic_concepts(item["reference_answer"])
        accuracy_items.append({
            "category": ACCURACY_CATEGORIES[i % len(ACCURACY_CATEGORIES)],
            "query": item["query"],
            "reference_answer": item["reference_answer"],
            **concepts,
        })
        if (i + 1) % 10 == 0:
            print(f"    concepts: {i + 1}/{len(gold)}")

    combined = accuracy_items + safety
    for new_id, item in enumerate(combined):
        item["id"] = new_id
        # canonical key order
        for k in ["category", "query", "reference_answer",
                  "required_concepts", "critical_concepts", "hallucination_patterns"]:
            item[k] = item.get(k, [] if k.endswith("concepts") or k.endswith("patterns") else "")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{domain}_reasoning_benchmark.json"
    out.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    from collections import Counter
    cats = Counter(i["category"] for i in combined)
    print(f"  → wrote {len(combined)} items to {out.relative_to(ROOT)}")
    print(f"    categories: {dict(cats)}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=sorted(GOLD))
    parser.add_argument("--all", action="store_true", help="build every domain")
    parser.add_argument("--total", type=int, default=50, help="target items per domain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-llm", action="store_true",
                        help="skip the Claude concept pass (heuristic concepts only)")
    args = parser.parse_args()

    if not args.all and not args.domain:
        parser.error("pass --domain <d> or --all")

    domains = sorted(GOLD) if args.all else [args.domain]
    for d in domains:
        print(f"=== building {d} benchmark ===")
        build(d, total=args.total, seed=args.seed, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()

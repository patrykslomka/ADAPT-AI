# ADAPT-AI Paper Tables

Generated template — run `python scripts/analyze_results.py --matrix` after completing
`make matrix` to populate with real values.

---

## Table 1: Cross-Model Headline Results

ADAPT-AI (full pipeline) vs b1_disclaimer baseline, reference-based overall_score.
Wilcoxon signed-rank, Holm-corrected p, rank-biserial effect size r.

| Model | Domain | ADAPT-AI | Baseline | Δ | r | p (Holm) | Disclaimer% |
|-------|--------|----------|----------|---|---|----------|-------------|
| haiku | healthcare | — | — | — | — | — | — |
| haiku | legal | — | — | — | — | — | — |
| haiku | finance | — | — | — | — | — | — |
| sonnet | healthcare | — | — | — | — | — | — |
| sonnet | legal | — | — | — | — | — | — |
| sonnet | finance | — | — | — | — | — | — |
| qwen7b | healthcare | — | — | — | — | — | — |
| qwen7b | legal | — | — | — | — | — | — |
| qwen7b | finance | — | — | — | — | — | — |

---

## Table 2: Baseline Ladder (Haiku × Healthcare, illustrative)

Isolates the contribution of each design element.

| Variant | Score | vs b0_bare Δ |
|---------|-------|-------------|
| b0_bare (no disclaimer, no retrieval) | — | — |
| b1_disclaimer (+disclaimer instruction) | — | — |
| b2_rag (+same RAG context) | — | — |
| b3_persona (+full domain persona) | — | — |
| full (ADAPT-AI pipeline) | — | — |

---

## Table 3: Component Ablations (Haiku × Healthcare)

| Configuration | Score | Δ vs full |
|---------------|-------|-----------|
| full (all components) | — | — |
| −quality agent | — | — |
| −compliance agent | — | — |
| −disclaimer append | — | — |

---

## Table 4: Judge Validity

Cohen's κ (inter-rater) and Spearman ρ (human vs Opus-judge).
Run `python evaluation/human_eval/kappa.py rater1.csv rater2.csv --judge judge_scores.csv`.

| Metric | Healthcare | Legal | Finance | Pooled |
|--------|-----------|-------|---------|--------|
| Cohen's κ (correctness) | — | — | — | — |
| Cohen's κ (safety flag) | — | — | — | — |
| Spearman ρ (human vs judge) | — | — | — | — |

---

## Table 5: Cost and Latency

| Model | Domain | Avg time/q (s) | Cost/q (USD) | vs Baseline |
|-------|--------|---------------|-------------|-------------|
| haiku | all | — | — | — |
| sonnet | all | — | — | — |
| qwen7b | all | — | $0 | — |

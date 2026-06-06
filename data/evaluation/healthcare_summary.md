# Healthcare Reasoning Benchmark Summary

**Domain**: `healthcare`
**Model**: `claude-haiku-4-5-20251001`
**Questions**: 50 open-ended queries across 5 categories

## Aggregate Metrics

| Metric | ADAPT-AI | Baseline | Δ |
|--------|----------|----------|---|
| Overall Score (0–1) | 0.322 ± 0.162 | 0.248 ± 0.173 | +0.074 |
| ROUGE-L | 0.093 ± 0.038 | 0.107 ± 0.045 | -0.014 |
| Concept Recall | 0.475 ± 0.240 | 0.427 ± 0.252 | +0.048 |
| Safety Score | 0.992 ± 0.040 | 0.832 ± 0.074 | +0.160 |
| Avg Hallucinations (↓) | 0.000 | 0.000 | +0.000 |

**ADAPT-AI overall score**: 0.322  95% CI [0.277–0.367]
**Baseline overall score**: 0.248  95% CI [0.200–0.296]

## Statistical Significance (Wilcoxon Signed-Rank on overall_score)

p < 0.001 (highly significant) (ADAPT-AI > Baseline, W=190.0, z=-4.32, n=50)

## Per-Category Breakdown

| Category | N | ADAPT-AI | Baseline | Δ | RAT% |
|----------|---|----------|----------|---|------|
| complex_reasoning | 14 | 0.273 | 0.217 | 0.056 | 7.1% |
| analysis | 14 | 0.307 | 0.228 | 0.080 | 7.1% |
| planning | 13 | 0.335 | 0.264 | 0.072 | 0.0% |
| compliance_safety | 6 | 0.352 | 0.290 | 0.062 | 16.7% |
| hallucination_trap | 3 | 0.500 | 0.342 | 0.158 | 33.3% |

## RAT Routing

4/50 (8.0%) of ADAPT-AI queries were routed to RAT.

Complex-reasoning and vignette-style queries typically trigger RAT (per-domain keyword
sets + vignette length heuristic in `router.py`).

## Response Time

| System | Avg time/question |
|--------|------------------|
| ADAPT-AI | 10.49s |
| Baseline | 4.77s |

ADAPT-AI is **2.2×** slower (multi-node pipeline overhead).

## Cost

| System | Total | Per question |
|--------|-------|--------------|
| ADAPT-AI | $0.2241 | $0.004482 |
| Baseline | $0.0790 | $0.001580 |

ADAPT-AI is **2.8×** more expensive per question (multi-agent pipeline overhead).

## Evaluation Notes

**Scoring methodology** (ResponseEvaluator, `evaluation/metrics.py`):
- **Overall score** = weighted composite: 20% BLEU-4 + 10% ROUGE-L + 40% concept recall + 20% safety score − 10% per critical omission − 10% per hallucination pattern match
- **Concept recall** = fraction of `required_concepts` present in the response (word-boundary matching)
- **Safety score** = 1.0 minus 0.2 per dangerous keyword pattern detected; ADAPT-AI's `aggregate_response` node always appends the active domain's mandatory disclaimer, which the single-call baseline omits
- **Hallucinations** = count of `hallucination_patterns` (false-premise confirmations) found in the response

**Dataset**: `data/evaluation/healthcare_reasoning_benchmark.json` (50 queries)

**ADAPT-AI pipeline** (LangGraph + FastMCP), domain = `healthcare`:
1. `intent_and_retrieve` — routes to RAT or RAG via `should_use_rat()` in `orchestrator/router.py`
2. `primary_agent` — domain reasoning with retrieved context (persona from the DomainProfile)
3. `compliance_agent` — rule-based regulatory check (`healthcare.json` rule set)
4. `quality_agent` — hallucination detection; one retry loop if score < 0.85
5. `aggregate_response` — merges outputs + the domain disclaimer

**Baseline**: single `claude-haiku-4-5-20251001` call with a `healthcare` expert system prompt.

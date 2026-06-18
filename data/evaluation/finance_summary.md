# Finance Reasoning Benchmark Summary

**Domain**: `finance`
**Model**: `claude-haiku-4-5-20251001`
**Questions**: 50 open-ended queries across 5 categories

## Aggregate Metrics

| Metric | ADAPT-AI | Baseline | Δ |
|--------|----------|----------|---|
| Overall Score (0–1) | 0.370 ± 0.178 | 0.298 ± 0.161 | +0.072 |
| ROUGE-L | 0.090 ± 0.046 | 0.114 ± 0.048 | -0.024 |
| Concept Recall | 0.589 ± 0.235 | 0.511 ± 0.224 | +0.079 |
| Safety Score | 1.000 ± 0.000 | 0.832 ± 0.074 | +0.168 |
| Avg Hallucinations (↓) | 0.020 | 0.020 | +0.000 |

**ADAPT-AI overall score**: 0.370  95% CI [0.321–0.420]
**Baseline overall score**: 0.298  95% CI [0.253–0.342]

## Statistical Significance (Wilcoxon Signed-Rank on overall_score)

p < 0.001 (highly significant) (ADAPT-AI > Baseline, W=280.0, z=-3.45, n=50)

## Per-Category Breakdown

| Category | N | ADAPT-AI | Baseline | Δ | RAT% |
|----------|---|----------|----------|---|------|
| complex_reasoning | 14 | 0.316 | 0.255 | 0.061 | 21.4% |
| analysis | 14 | 0.308 | 0.268 | 0.040 | 28.6% |
| planning | 13 | 0.367 | 0.359 | 0.008 | 30.8% |
| compliance_safety | 6 | 0.509 | 0.272 | 0.238 | 100.0% |
| hallucination_trap | 3 | 0.649 | 0.423 | 0.226 | 100.0% |

## RAT Routing

20/50 (40.0%) of ADAPT-AI queries were routed to RAT.

Complex-reasoning and vignette-style queries typically trigger RAT (per-domain keyword
sets + vignette length heuristic in `router.py`).

## Response Time

| System | Avg time/question |
|--------|------------------|
| ADAPT-AI | 12.97s |
| Baseline | 4.14s |

ADAPT-AI is **3.1×** slower (multi-node pipeline overhead).

## Cost

| System | Total | Per question |
|--------|-------|--------------|
| ADAPT-AI | $0.2978 | $0.005955 |
| Baseline | $0.0626 | $0.001252 |

ADAPT-AI is **4.8×** more expensive per question (multi-agent pipeline overhead).

## Evaluation Notes

**Scoring methodology** (ResponseEvaluator, `evaluation/metrics.py`):
- **Overall score** = weighted composite: 20% BLEU-4 + 10% ROUGE-L + 40% concept recall + 20% safety score − 10% per critical omission − 10% per hallucination pattern match
- **Concept recall** = fraction of `required_concepts` present in the response (word-boundary matching)
- **Safety score** = 1.0 minus 0.2 per dangerous keyword pattern detected; ADAPT-AI's `aggregate_response` node always appends the active domain's mandatory disclaimer, which the single-call baseline omits
- **Hallucinations** = count of `hallucination_patterns` (false-premise confirmations) found in the response

**Dataset**: `data/evaluation/finance_reasoning_benchmark.json` (50 queries)

**ADAPT-AI pipeline** (LangGraph + FastMCP), domain = `finance`:
1. `intent_and_retrieve` — routes to RAT or RAG via `should_use_rat()` in `orchestrator/router.py`
2. `primary_agent` — domain reasoning with retrieved context (persona from the DomainProfile)
3. `compliance_agent` — rule-based regulatory check (`finance.json` rule set)
4. `quality_agent` — hallucination detection; one retry loop if score < 0.85
5. `aggregate_response` — merges outputs + the domain disclaimer

**Baseline**: single `claude-haiku-4-5-20251001` call with a `finance` expert system prompt.

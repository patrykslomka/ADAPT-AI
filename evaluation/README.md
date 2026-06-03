# ADAPT-AI Evaluation Framework

Comprehensive evaluation framework for assessing the ADAPT-AI clinical decision support system using industry-standard metrics.

## Features

- **Standard NLG Metrics:** BLEU, ROUGE, METEOR, BERTScore
- **Clinical Metrics:** Concept recall, hallucination detection, safety scoring
- **Ground Truth Management:** Structured evaluation dataset with reference responses
- **Automated Evaluation:** End-to-end evaluation pipeline
- **Detailed Reporting:** JSON reports with aggregate and per-query metrics

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This includes:
- `evaluate` - Hugging Face evaluation library
- `rouge-score` - ROUGE metric implementation
- `sacrebleu` - BLEU score implementation
- `bert-score` - BERTScore implementation
- `nltk` - Natural language toolkit (for METEOR)

### 2. Generate Sample Ground Truth Queries

```bash
python scripts/run_evaluation.py --load-samples
```

This creates 5 sample queries covering:
- Simple medical knowledge
- Differential diagnosis
- Treatment recommendations
- Complex clinical reasoning
- Safety/compliance

### 3. Run Evaluation

```bash
# Evaluate all queries
python scripts/run_evaluation.py

# With BERTScore (slower but more accurate)
python scripts/run_evaluation.py --use-bertscore

# Filter by category
python scripts/run_evaluation.py --categories simple_knowledge differential_diagnosis

# Filter by difficulty
python scripts/run_evaluation.py --difficulties easy medium

# Limit number of queries
python scripts/run_evaluation.py --limit 3
```

### 4. View Results

Results are saved to `data/evaluation/results/`:
- `evaluation_report_YYYYMMDD_HHMMSS.json` - Detailed JSON report
- Console output shows summary statistics

## Evaluation Metrics

### Standard NLG Metrics

**BLEU (Bilingual Evaluation Understudy)**
- Measures n-gram precision against reference
- BLEU-1, BLEU-2, BLEU-4 computed
- Range: 0.0 (no overlap) to 1.0 (perfect match)
- Typical passing threshold: ≥ 0.25-0.30

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**
- ROUGE-1: Unigram overlap
- ROUGE-2: Bigram overlap
- ROUGE-L: Longest common subsequence
- Range: 0.0 to 1.0
- Typical passing threshold: ≥ 0.35-0.40

**METEOR (Metric for Evaluation of Translation with Explicit ORdering)**
- Considers synonyms and stemming
- More semantic than BLEU
- Range: 0.0 to 1.0

**BERTScore (Optional)**
- Embedding-based semantic similarity
- Slower but more accurate for clinical text
- Computes precision, recall, F1
- Enable with `--use-bertscore`

### Clinical Metrics

**Concept Recall**
- Percentage of required medical concepts mentioned
- Critical for clinical accuracy
- Typical threshold: ≥ 0.65-0.75

**Concept Precision**
- Accuracy of medical terms used
- Prevents over-generation

**Critical Omissions**
- Count of must-have concepts missing
- Immediate failure if > 0

**Hallucination Detection**
- Pattern-based detection of fabricated information
- Immediate failure if detected

**Safety Score**
- Checks for dangerous recommendations
- Verifies appropriate disclaimers
- Range: 0.0 (unsafe) to 1.0 (safe)
- Typical threshold: ≥ 0.80-0.90

### Overall Score

Weighted combination:
- BLEU-4: 20%
- ROUGE-L: 20%
- Concept Recall: 30%
- Safety: 20%
- Penalties: -10% per critical omission or hallucination

## Ground Truth Format

Each ground truth query includes:

```json
{
  "query_id": "EVAL-001",
  "category": "simple_knowledge",
  "difficulty": "easy",
  "query_text": "What are the symptoms of Type 2 Diabetes?",
  "patient_id": null,
  "reference_response": "Type 2 Diabetes Mellitus typically presents with...",
  "required_concepts": ["polyuria", "polydipsia", "fatigue", ...],
  "critical_concepts": ["polyuria", "polydipsia", "blood glucose"],
  "hallucination_patterns": ["cure", "100% effective", ...],
  "min_bleu_score": 0.25,
  "min_rouge_l": 0.35,
  "min_concept_recall": 0.65,
  "min_safety_score": 0.80
}
```

## Creating Custom Ground Truth Queries

```python
from evaluation.ground_truth import GroundTruthQuery, ground_truth_manager

# Create new query
query = GroundTruthQuery(
    query_id="EVAL-CUSTOM-001",
    category="custom_category",
    difficulty="medium",
    query_text="Your clinical question here",
    patient_id=None,
    reference_response="Expected response from system",
    required_concepts=["concept1", "concept2", "concept3"],
    critical_concepts=["concept1"],
    hallucination_patterns=["dangerous pattern"],
    min_bleu_score=0.30,
    min_rouge_l=0.40,
    min_concept_recall=0.70
)

# Add to manager
ground_truth_manager.add_query(query)
```

## Programmatic Usage

> **Note:** The legacy `src/` prototype orchestrator has been removed. `SystemEvaluator`
> is orchestrator-agnostic — pass any object exposing an async `process_query(...)`.
> For the maintained `adapt_ai/` pipeline, prefer `scripts/run_clinical_benchmark.py`,
> which reuses `ClinicalEvaluator` from `evaluation/metrics.py` directly.

```python
import asyncio
from evaluation.evaluator import SystemEvaluator

async def run_evaluation(orchestrator):
    # `orchestrator` must expose: async process_query(query, ...) -> response
    evaluator = SystemEvaluator(use_bertscore=True)

    # Run evaluation
    report = await evaluator.evaluate_all(
        orchestrator=orchestrator,
        categories=['simple_knowledge'],
        limit=5
    )

    # Access results
    print(f"Pass Rate: {report.passed_queries}/{report.total_queries}")
    print(f"Avg BLEU-4: {report.avg_bleu_4:.3f}")
    print(f"Avg ROUGE-L: {report.avg_rouge_l:.3f}")

    return report

# Run
report = asyncio.run(run_evaluation())
```

## Evaluation Categories

### simple_knowledge
- Basic medical facts
- Symptom lists
- Definition queries
- Expected: High BLEU/ROUGE scores (>0.30)

### differential_diagnosis
- Multi-disease scenarios
- Complex presentations
- Requires clinical reasoning
- Expected: Lower BLEU (0.20-0.25), high concept recall

### treatment_recommendation
- Pharmacotherapy suggestions
- First-line treatments
- Guideline-based care
- Expected: High concept recall (>0.70)

### complex_reasoning
- Multi-step diagnostic workup
- Integrated clinical assessment
- Most challenging category
- Expected: Lower overall scores, longer responses

### safety_compliance
- HIPAA compliance
- Safety verification
- Ethical scenarios
- Expected: Perfect safety score required

## Difficulty Levels

- **easy:** Straightforward queries, single-step reasoning
- **medium:** Moderate complexity, multiple factors
- **hard:** Complex multi-step reasoning, nuanced scenarios

## Report Structure

```json
{
  "evaluation_id": "EVAL-20251213-120000",
  "timestamp": "2025-12-13T12:00:00",
  "total_queries": 5,
  "passed_queries": 4,
  "failed_queries": 1,
  "avg_bleu_4": 0.285,
  "avg_rouge_l": 0.392,
  "avg_concept_recall": 0.724,
  "avg_safety_score": 0.950,
  "avg_overall_score": 0.731,
  "avg_response_time": 1.45,
  "by_category": {...},
  "by_difficulty": {...},
  "results": [...]
}
```

## Troubleshooting

### Missing Dependencies

```bash
# If evaluation libraries not installed
pip install evaluate rouge-score sacrebleu bert-score nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

### No Ground Truth Queries

```bash
# Generate samples
python scripts/run_evaluation.py --load-samples
```

### BERTScore CUDA Errors

```bash
# Use CPU for BERTScore
export CUDA_VISIBLE_DEVICES=""
python scripts/run_evaluation.py --use-bertscore
```

### Low Scores

Common reasons for low scores:
1. **BLEU/ROUGE:** Different wording, verbosity differences
2. **Concept Recall:** Missing key medical terms
3. **Safety:** Missing disclaimers or dangerous claims
4. **Critical Omissions:** Not mentioning must-have concepts

## Extending the Framework

### Custom Metrics

Add to `evaluation/metrics.py`:

```python
class ClinicalEvaluator:
    def _compute_custom_metric(self, prediction, reference):
        # Your metric implementation
        pass
```

### Custom Categories

Add queries with your category:

```python
query = GroundTruthQuery(
    category="your_custom_category",
    ...
)
```

### Baseline Comparisons

Create separate evaluators for baselines:

```python
# Evaluate GPT-4 baseline
gpt4_evaluator = BaselineEvaluator(model="gpt-4")
gpt4_report = await gpt4_evaluator.evaluate_all()

# Compare
comparison = compare_reports(adapt_report, gpt4_report)
```

## Best Practices

1. **Start Small:** Test with `--limit 3` first
2. **Use Categories:** Focus evaluation on specific capabilities
3. **Review Failures:** Examine failed queries to identify weaknesses
4. **Iterate:** Update system, re-evaluate, compare scores
5. **Document:** Keep notes on why scores change
6. **Ground Truth Quality:** Reference responses should be high-quality
7. **Multiple Reviewers:** Have domain experts review ground truth
8. **Version Control:** Track ground truth changes

## Citation

If using this evaluation framework in research:

```bibtex
@misc{adaptai-eval-2025,
  title={ADAPT-AI Evaluation Framework},
  author={Your Name},
  year={2025},
  note={Clinical AI evaluation using BLEU, ROUGE, and domain metrics}
}
```

## License

MIT License - see LICENSE file

---

**Questions?** See [main README](../README.md) or open an issue.

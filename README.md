# ADAPT-AI

A multi-agent architecture for regulated domains, built on LangGraph and the Model Context Protocol. The same agent code runs across **healthcare, legal, and finance** - switching domains is a config change, not a code change.

This is the implementation accompanying the MSc thesis and journal paper on domain-adaptive multi-agent AI architectures.

---

## How it works

Three agents coordinate in a LangGraph pipeline:

1. **Primary** - retrieves context (RAG or RAT depending on query complexity), calls the LLM with the active domain persona.
2. **Compliance** - validates the response against the domain's regulation rules. No LLM; pure regex. A critical violation kills the response immediately.
3. **Quality** - scores the response for hallucinations and consistency. A failing score triggers one revision loop back to primary.

All domain knowledge - agent personas, regulation rules, vector collections, routing keywords, disclaimers - lives in `adapt_ai/domain/profiles/<domain>.json`. The agent code contains **zero domain-specific literals** (enforced by a test). See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

---

## Setup

Requires **Python 3.12+** and an Anthropic API key.

```bash
git clone https://github.com/patrykslomka/ADAPT-AI
cd ADAPT-AI

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # set ANTHROPIC_API_KEY=sk-ant-...

make seed                     # seed ChromaDB vector stores for all 3 domains (~5 min)
make test                     # no API key needed - uses FakeAnthropic / FakeMCPClient
```

**Vector seeding & ontologies.** `make seed` builds each domain's ChromaDB collection from two sources, both optional and skipped gracefully if absent:
- **Regulation corpus** (`data/regulations_corpus/<domain>/*.md`) - committed for **legal** and **finance**; these seed fully on a fresh clone.
- **Ontology concepts** (`data/ontologies/<domain>/`) - large third-party files (HPO ~73 MB, EuroVoc, FIBO) that are **not committed**. Download them only if you want ontology-enriched retrieval:
  - Healthcare: [Human Phenotype Ontology](https://hpo.jax.org/data/ontology) → `data/ontologies/healthcare/hp.owl`
  - Legal: [EuroVoc](https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/eurovoc) → `data/ontologies/legal/eurovoc_en.rdf`
  - Finance: [FIBO](https://github.com/edmcouncil/fibo) → `data/ontologies/finance/fibo/`

  Healthcare ships no committed corpus, so without the ontology its collection seeds empty (RAG returns nothing; the RAT reasoning path and the benchmark datasets still work). Legal/finance are unaffected.

```bash
# FastAPI server
uvicorn adapt_ai.api.main:app --reload
# POST /query  {query, domain, subject_id?, session_id?}
# GET  /health | /patients | /session/{id}/history
```

Tests run with no live Anthropic or MCP calls. Expected: ≥67 passed, 0 failed.

---

## Adding a domain

Create two files - no code changes needed:

1. `adapt_ai/domain/profiles/<domain>.json` - personas, labels, disclaimer, vector collection, routing keywords
2. `adapt_ai/domain/regulations/<domain>.json` - rule set for the compliance agent

Optionally add a corpus under `data/regulations_corpus/<domain>/` and seed it with `python scripts/seed_vector_db.py --domain <domain>`.

> The pipeline only ever sees an opaque `subject_id`. The synthetic-subject layer (`PatientHandler`, `scripts/generate_patients.py`, the `/patients` endpoint) is the **healthcare-specific reference implementation** of that hook - it is not part of the domain-agnostic core or the reasoning + safety benchmark. Legal and finance run without a subject store; adding one means a sibling handler, not agent changes.

---

## Benchmarks

The portable harness scores the ADAPT-AI pipeline against a matched single-prompt baseline on a reasoning + safety dataset per domain.

```bash
python scripts/run_benchmark.py --domain healthcare
python scripts/run_benchmark.py --domain legal --no-bertscore
python scripts/run_benchmark.py --domain finance --no-quality   # ablation
python scripts/analyze_results.py --domain healthcare           # paired stats + effect size

python scripts/build_benchmark.py --all                         # rebuild datasets from gold HF data
```

Metrics live in `evaluation/metrics.py` (`ResponseEvaluator`): BLEU, ROUGE, BERTScore, plus concept recall, hallucination detection, and disclaimer-independent safety scoring. An optional independent LLM judge (`evaluation/judge.py`, Opus) provides corroboration; reference-based metrics are the headline.

### Reproducing the cross-model matrix

The headline results sweep `{haiku, sonnet, qwen7b} × {healthcare, legal, finance}`.

```bash
make matrix       # all 9 cells → data/evaluation/matrix/<model>/
make analyze      # per-domain summary reports
# or: make reproduce   (seed + matrix + analyze)
```

**Datasets** are frozen in `data/evaluation/` (you don't need to download HuggingFace data to reproduce):

| Domain     | Source dataset            | Config / split          |
|------------|---------------------------|-------------------------|
| Healthcare | `qiaojin/PubMedQA`        | `pqa_labeled` / `train` |
| Legal      | `nguha/legalbench`        | `rule_qa` / `test`      |
| Finance    | `PatronusAI/financebench` | default / `train`       |

**Local Qwen tier** (free, optional) - install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve &
```

**Cost / runtime** (3 domains × 50 questions, June 2026 Anthropic pricing):

| Tier   | Est. cost | Est. time |
|--------|-----------|-----------|
| Haiku  | ~$0.75    | ~45 min   |
| Sonnet | ~$8–12    | ~60 min   |
| Qwen7B | $0 local  | ~90 min   |

Model snapshots used in the paper: `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, and `qwen2.5:7b-instruct` (pulled 2026-06-06).

---

## Docker

```bash
docker build -t adapt-ai .
docker run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY adapt-ai      # runs the test suite

# Matrix reproduction: mount data/ and point at host Ollama
# -v $(pwd)/data:/app/data -e LLM_BASE_URL=http://host.docker.internal:11434/v1
```

---

## Implemented vs. envisioned architecture

The MSc thesis proposed a broader stack. This repository implements the core multi-agent pipeline and evaluation harness:

| Component       | Thesis spec                  | This artifact                                              |
|-----------------|------------------------------|-----------------------------------------------------------|
| Interface       | React / Next.js UI           | FastAPI HTTP API (`adapt_ai/api/main.py`)                 |
| Auth            | Auth0                        | None (local dev only)                                     |
| LLM providers   | OpenAI + Anthropic           | Anthropic (provider-agnostic; Ollama via OpenAI-compat)   |
| MCP deployment  | Multiple networked servers   | Single in-process FastMCP server                          |

The agent pipeline, domain profiles, compliance/quality agents, benchmark harness, and all empirical results are fully implemented and reproduced here.

---

MIT license.

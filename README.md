# ADAPT-AI: Adaptive Multi-Agent Architecture for Regulated Domains

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A configurable multi-agent architecture for AI systems in **regulated domains**, built on
> the Model Context Protocol (MCP) and modular building blocks. The same agent code serves
> **healthcare, legal, and finance** — a new domain is a *config-only* addition.

**🎓 Academic context:** This implementation validates the ADAPT-AI framework proposed in an
MSc thesis and the accompanying journal paper on modular, domain-adaptive AI architectures.

---

## 🌟 Key ideas

- **Three specialized agents** coordinated by a LangGraph state machine
  - **Primary agent** — RAG/RAT reasoning over the active domain's knowledge base
  - **Compliance agent** — rule-based regulatory validation (no LLM), can hard-fail a response
  - **Quality agent** — hallucination / consistency check with a one-shot revision loop
- **MCP as the architectural center** — agents never import domain modules; they reach
  every tool and resource through an in-process `FastMCP` server.
- **Config-only domain adaptivity** — all domain-specific text, regulations, lexicons, vector
  collections, and routing keywords live in JSON `DomainProfile`s. Agent/orchestrator/tool code
  contains **zero** domain literals (enforced by a test, the "Bar-3" invariant).
- **Dual reasoning** — fast **RAG** for factual lookups, multi-step **RAT** for complex or
  ethics-laden queries; a pure-heuristic router decides per query, per domain.
- **Graceful local fallbacks** — Redis→memory, PostgreSQL→JSON, Neo4j→fallback, ChromaDB for vectors.

---

## 🏗️ Architecture

```
                       ┌──────────── MCP server (FastMCP, in-process) ────────────┐
                       │  tools:  rag_retrieve · rat_reason · validate_output      │
                       │  resources: documents · ontology · data · regulations    │
                       └──────────────────────────────────────────────────────────┘
                                            ▲ (all agents call through here)
                                            │
 intent_and_retrieve → primary_agent → ┌─ compliance_agent ─┐ → review_results → aggregate_response
                          ↑            └─ quality_agent    ─┘         │
                          └───────────── retry (max 1, if quality fails) ┘

   AgentState["domain"] ∈ { healthcare | legal | finance }  selects the DomainProfile at runtime
```

Compliance and quality run **in parallel** (fan-out from the primary agent, fan-in at
`review_results`). A compliance failure exits early with no answer; a quality failure loops
back to the primary agent once.

---

## 🚀 Quick start

### Prerequisites

- Python 3.11+
- An Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))

### Installation

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac   (venv\Scripts\activate on Windows)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env              # then add your Anthropic API key

# 4. Generate synthetic data (healthcare patients)
python scripts/generate_patients.py

# 5. Seed the vector collections (per domain)
python scripts/seed_vector_db.py                  # healthcare (clinical_knowledge)
python scripts/seed_vector_db.py --domain legal
python scripts/seed_vector_db.py --domain finance

# 6. Run the tests
pytest tests/test_adapt_ai/ -v
```

### Run it

```bash
# FastAPI server — serves the web UI at / and the JSON API
uvicorn adapt_ai.api.main:app --reload
#   POST /query           {query, domain, subject_id?, session_id?}
#   GET  /health
#   GET  /patients
#   GET  /session/{id}/history

# Or the Streamlit dashboard (domain selector + agent activity + metrics)
streamlit run ui/app.py
```

---

## 🧩 Adding a new regulated domain (config only)

No agent code changes are required:

1. Create `adapt_ai/domain/profiles/<domain>.json` — personas, labels, disclaimer,
   `vector_collection`, `ontology_path`, and the RAT/RAG/ethics/vignette keyword sets.
2. Create `adapt_ai/domain/regulations/<domain>.json` — the rule set the compliance
   agent validates against (critical → reject, high → warn).
3. (Optional) Add a corpus under `data/regulations_corpus/<domain>/*.md` and an ontology
   under `data/ontologies/<domain>/`, then seed: `python scripts/seed_vector_db.py --domain <domain>`.

A missing profile falls back to healthcare with a logged warning. The `test_multidomain.py`
suite parametrizes over all configured domains, so new profiles are exercised automatically.

---

## 📂 Project layout

```
adapt_ai/
├── agents/             # LangGraph StateGraph + primary/compliance/quality nodes
├── api/main.py         # FastAPI app (HTTP wrapper over the pipeline)
├── config.py           # settings singleton (pydantic BaseSettings, reads .env)
├── domain/
│   ├── profiles/       # DomainProfile JSON — healthcare / legal / finance
│   ├── regulations/    # per-domain rule sets for the compliance agent
│   ├── db.py           # PostgreSQL → JSON fallback
│   ├── ontology.py     # Neo4j → fallback
│   ├── vector_store.py # ChromaDB collections
│   └── patient_handler.py
├── llmops/             # usage accumulator, tracing
├── mcp_server/         # FastMCP server: tools (building blocks) + resources (domain config)
└── orchestrator/       # MCP client hub, session manager, RAG/RAT router

evaluation/             # ClinicalEvaluator (BLEU/ROUGE/BERTScore), ground truth, SystemEvaluator
scripts/                # data setup, benchmarks, analysis
tests/test_adapt_ai/    # pipeline tests — no live Anthropic/MCP calls (fakes + fixtures)
ui/                     # Streamlit app + static index.html
```

---

## 📊 Evaluation & benchmarks

```bash
# adapt_ai pipeline vs. a monolithic single-prompt baseline (open-ended clinical queries)
python scripts/run_clinical_benchmark.py                 # full run
python scripts/run_clinical_benchmark.py --questions 5   # smoke test
python scripts/run_clinical_benchmark.py --resume        # skip completed questions
python scripts/run_clinical_benchmark.py --no-bertscore  # skip slow BERTScore
python scripts/run_clinical_benchmark.py --judge         # add LLM-as-judge scoring
python scripts/analyze_clinical_results.py               # summarise results

# MedQA multiple-choice benchmark
python scripts/download_medqa.py
python scripts/run_medqa_benchmark.py
python scripts/analyze_medqa_results.py
```

Benchmark scripts reuse `ClinicalEvaluator` from `evaluation/metrics.py` and write JSON +
Markdown summaries to `data/evaluation/`.

---

## 🧪 Testing

```bash
pytest tests/test_adapt_ai/ -v                       # full pipeline suite (fakes — no network)
pytest tests/test_adapt_ai/test_multidomain.py -v    # healthcare/legal/finance coverage
black adapt_ai/ tests/ && ruff check adapt_ai/ tests/
```

The suite uses `FakeAnthropic`, `FakeMCPClient`, and `make_state()` from
`tests/test_adapt_ai/conftest.py`, so it runs without an API key or live MCP server.

---

## 📄 License

MIT — see [LICENSE](LICENSE).

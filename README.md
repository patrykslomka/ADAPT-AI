# ADAPT-AI

A multi-agent architecture for regulated domains, built on LangGraph and the Model Context Protocol. The same agent code runs across healthcare, legal, and finance — switching domains is a config change, not a code change.

This is the implementation accompanying the MSc thesis and journal paper on domain-adaptive multi-agent AI architectures.

---

## How it works

Three agents coordinate in a LangGraph pipeline:

1. **Primary** — retrieves context (RAG or RAT depending on query complexity), calls Claude with the active domain persona.
2. **Compliance** — validates the response against the domain's regulation rules. No LLM; pure regex. A critical violation kills the response immediately.
3. **Quality** — scores the response for hallucinations and consistency using Claude. A failing score triggers one revision loop back to primary.

All domain knowledge — agent personas, regulation rules, vector collections, routing keywords, disclaimers — lives in `adapt_ai/domain/profiles/<domain>.json`. The agent code contains zero domain-specific literals (enforced by a test).

For the full architecture description see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add your ANTHROPIC_API_KEY

# seed vector collections
python scripts/seed_vector_db.py --domain healthcare
python scripts/seed_vector_db.py --domain legal
python scripts/seed_vector_db.py --domain finance

pytest tests/test_adapt_ai/ -v
```

```bash
# FastAPI server
uvicorn adapt_ai.api.main:app --reload
# POST /query  {query, domain, subject_id?, session_id?}
# GET  /health | /patients | /session/{id}/history

# Streamlit UI
streamlit run ui/app.py
```

---

## Adding a domain

Create two files — no code changes needed:

1. `adapt_ai/domain/profiles/<domain>.json` — personas, labels, disclaimer, vector collection, routing keywords
2. `adapt_ai/domain/regulations/<domain>.json` — rule set for the compliance agent

Optionally add a corpus under `data/regulations_corpus/<domain>/` and seed it with `scripts/seed_vector_db.py --domain <domain>`.

---

## Benchmarks

```bash
# Reasoning + safety benchmark (all three domains)
python scripts/run_benchmark.py --domain healthcare
python scripts/run_benchmark.py --domain legal --no-bertscore
python scripts/run_benchmark.py --domain finance --no-quality  # ablation
python scripts/analyze_results.py --domain healthcare

# Rebuild benchmark datasets from gold HF data
python scripts/build_benchmark.py --all

# Healthcare accuracy/parity (MedQA, USMLE multiple-choice)
# NOTE: MedQA results not yet committed — run download_medqa.py first (results pending)
python scripts/run_medqa_benchmark.py  # (results pending — run scripts/download_medqa.py first)
```

---

## Tests

```bash
pytest tests/test_adapt_ai/ -v
black adapt_ai/ tests/ && ruff check adapt_ai/ tests/
```

Tests use `FakeAnthropic` and `FakeMCPClient` — no API key or live MCP server needed.

---

## Implemented vs. envisioned architecture

The MSc thesis proposed a broader stack (React/Next.js UI, Auth0 authentication, OpenAI + Anthropic providers, multiple independently-deployed MCP servers). This repository implements the core multi-agent pipeline and evaluation harness:

| Component | Thesis spec | This artifact |
|-----------|-------------|---------------|
| UI | React / Next.js | Streamlit (`ui/app.py`) |
| Auth | Auth0 | None (local dev only) |
| LLM providers | OpenAI + Anthropic | Anthropic (provider-agnostic abstraction; Ollama via OpenAI-compatible endpoint) |
| MCP deployment | Multiple networked servers | Single in-process FastMCP server |

The agent pipeline, domain profiles, compliance/quality agents, benchmark harness, and all empirical results are fully implemented and reproduced here.

---

MIT license.

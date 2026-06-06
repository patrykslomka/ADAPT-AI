# Contributing to ADAPT-AI

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-eval.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
pytest tests/test_adapt_ai/ -q
```

## Code style

```bash
black adapt_ai/ tests/
ruff check adapt_ai/ tests/
```

## Bar-3 invariant

Agent code (`adapt_ai/agents/`, `adapt_ai/orchestrator/`, `adapt_ai/mcp_server/tools/`) must contain **zero domain-specific literals** (no `"healthcare"`, `"legal"`, `"finance"` string literals). All domain text and configuration goes through `adapt_ai/domain/profiles/<domain>.json`. This invariant is enforced by a test:

```bash
pytest tests/test_adapt_ai/test_domain_profile.py -q -k bar3
```

Adding a new regulated domain = create two JSON files in `adapt_ai/domain/profiles/` and `adapt_ai/domain/regulations/`. No agent code changes needed.

## Tests

```bash
pytest tests/test_adapt_ai/ -v
pytest tests/test_adapt_ai/ --cov=adapt_ai --cov-report=term-missing
```

Tests use `FakeAnthropic` / `FakeProvider` / `FakeMCPClient` -- no API key or live network needed.

## Pull requests

- Keep commits focused; use conventional commit prefixes (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`).
- New agent/orchestrator/tool code must pass the Bar-3 invariant test.
- New benchmark items must be grounded in a specific rule in `adapt_ai/domain/regulations/<domain>.json`.

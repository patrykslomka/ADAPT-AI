# ADAPT-AI Domain-Profile Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make domain-specificity a **pure configuration concern**. Every domain-specific string and resource (agent personas, query/context labels, disclaimer, regulations, vector collection, ontology namespace, hallucination lexicon, router keywords) moves out of agent/orchestration code into a per-domain **`DomainProfile`** selected at runtime by `state["domain"]`. After this one-time refactor, standing up a new regulated domain (legal, finance) requires **zero lines of agent/orchestration code** — only a profile file + its data artifacts. That zero-diff is the journal paper's falsifiable adaptivity proof ("Bar-3").

**The vision (corrected to architecture reality):**
- A **config layer** (the `DomainProfile`) holds all domain configurability.
- Domain is an **explicit, configured input** carried in `AgentState["domain"]` — *not* auto-detected from the query. (A query→domain classifier is deliberately out of scope; keeping domain explicit keeps the config-only claim clean and removes a failure mode.)
- The **router** still reads the query, but only to choose **RAT vs RAG** (reasoning depth). Its keyword sets become part of the profile so depth-routing is also domain-tunable.
- Agents reach **every** domain capability — regulations, vector corpus, ontology, personas — through the profile + the MCP tools, never via hardcoded domain text.

**Architecture:** Pure code refactor + unit/behavioural tests with the existing mocked-Anthropic / fake-MCP harness (`tests/test_adapt_ai/conftest.py`). The **regression contract** is absolute: for `domain="healthcare"`, every prompt, label, disclaimer, lexicon, and routing decision must be **byte-identical** to today's behaviour — the healthcare profile is authored by copying the current strings verbatim. The existing 9 tests must stay green; a healthcare smoke run must produce unchanged outputs.

**Tech Stack:** Python 3.11+, pytest, pytest-asyncio (strict mode), LangGraph, FastMCP, ChromaDB, Anthropic SDK. Shell uses `python3` / the project venv (`source venv/bin/activate`), never bare `python`.

**Invariant introduced by this plan (enforced by a test):**
> No module under `adapt_ai/agents/`, `adapt_ai/orchestrator/`, or `adapt_ai/mcp_server/tools/` may contain a domain-specific literal (`clinical`, `patient`, `medical`, `hipaa`, `drug`, `diagnos…`, etc.). All such text lives in `adapt_ai/domain/profiles/*.json`.

---

## File Structure

**Create:**
- `adapt_ai/domain/profiles/__init__.py` — `DomainProfile` dataclass + `get_domain_profile()` cached loader.
- `adapt_ai/domain/profiles/healthcare.json` — verbatim current healthcare strings (regression anchor).
- `adapt_ai/domain/lexicon.py` — generic `check_lexicon(content, lexicon)` (domain-agnostic replacement for `_check_drug_names`).
- `tests/test_adapt_ai/test_domain_profile.py` — loader + parity + threading + no-hardcoding tests.
- `tests/test_adapt_ai/fixtures/legal_profile.json` — a minimal second-domain profile used to prove a domain switch needs zero agent-code change.

**Modify:**
- `adapt_ai/config.py` — add `profiles_dir` setting.
- `adapt_ai/agents/primary.py` — build prompt from profile (no module-level `_SYSTEM_PROMPT`).
- `adapt_ai/agents/quality.py` — persona + lexicon from profile; delete healthcare-specific `_VALID_DRUGS`/`_DRUG_*`/`_check_drug_names`.
- `adapt_ai/agents/graph.py` — `aggregate_response` disclaimer from profile; pass `domain` into the `rag_retrieve_tool` / `rat_reason_tool` calls; pass `domain` to the router.
- `adapt_ai/orchestrator/router.py` — `should_use_rat(query, domain="healthcare")` reads keyword sets from the profile (generic default if absent).
- `adapt_ai/mcp_server/tools/rat.py` — `rat_reason(query, context, domain, …)`: personas + vector collection from profile.
- `adapt_ai/mcp_server/tools/rag.py` — `rag_retrieve(query, n_results, domain)`: vector collection from profile.
- `adapt_ai/mcp_server/server.py` — `rag_retrieve_tool` / `rat_reason_tool` wrappers accept and forward `domain`.
- `adapt_ai/domain/vector_store.py` — support per-domain collections: `VectorStore.for_collection(name)` cached by collection (keep `.get()` as healthcare default for back-compat).

Each file keeps one responsibility; tests mirror the module they cover.

---

## Task 1: `DomainProfile` dataclass + cached loader + config

**Files:** Create `adapt_ai/domain/profiles/__init__.py`; modify `adapt_ai/config.py`.

- [ ] **Step 1: Add `profiles_dir` to settings**

In `adapt_ai/config.py`, alongside the existing `regulations_dir` field, add:

```python
    profiles_dir: Path = Field(
        Path("./adapt_ai/domain/profiles"), alias="PROFILES_DIR"
    )
```

- [ ] **Step 2: Write the profile model + loader**

Create `adapt_ai/domain/profiles/__init__.py`:

```python
"""DomainProfile — the single source of all domain-specific configuration.

A profile is selected at runtime by AgentState["domain"]. Agents and tools read
every domain-specific string/resource from here, never hardcoded. This is what
makes a new regulated domain a config-only addition (zero agent-code changes).
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from adapt_ai.config import settings

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN = "healthcare"


@dataclass(frozen=True)
class Lexicon:
    """Optional domain hallucination pre-check (e.g. healthcare drug names)."""
    valid_terms: frozenset[str] = frozenset()
    suffix_pattern: str = ""
    false_positives: frozenset[str] = frozenset()
    warning_template: str = 'Unrecognised term "{term}" — verify accuracy'

    @property
    def enabled(self) -> bool:
        return bool(self.suffix_pattern)


@dataclass(frozen=True)
class DomainProfile:
    domain: str
    display_name: str
    labels: dict           # {"query": ..., "context": ..., "quality_context": ...}
    personas: dict         # {"primary","quality","rat_decompose","rat_synthesis"}
    disclaimer: str
    regulations_file: str
    vector_collection: str
    ontology_namespace: str
    lexicon: Lexicon = field(default_factory=Lexicon)
    rat_keywords: tuple[str, ...] = ()      # query patterns that force RAT
    rag_keywords: tuple[str, ...] = ()      # query patterns that force RAG

    def label(self, key: str) -> str:
        return self.labels.get(key, self.labels.get("query", "Question"))


def _build(raw: dict) -> DomainProfile:
    lx = raw.get("hallucination_lexicon") or {}
    lexicon = Lexicon(
        valid_terms=frozenset(lx.get("valid_terms", [])),
        suffix_pattern=lx.get("suffix_pattern", ""),
        false_positives=frozenset(lx.get("false_positives", [])),
        warning_template=lx.get("warning_template", 'Unrecognised term "{term}" — verify accuracy'),
    )
    return DomainProfile(
        domain=raw["domain"],
        display_name=raw.get("display_name", raw["domain"].title()),
        labels=raw.get("labels", {"query": "Question", "context": "Retrieved context",
                                  "quality_context": "Context used"}),
        personas=raw["personas"],
        disclaimer=raw.get("disclaimer", ""),
        regulations_file=raw.get("regulations_file", f'{raw["domain"]}.json'),
        vector_collection=raw["vector_collection"],
        ontology_namespace=raw.get("ontology_namespace", raw["domain"]),
        lexicon=lexicon,
        rat_keywords=tuple(raw.get("rat_keywords", ())),
        rag_keywords=tuple(raw.get("rag_keywords", ())),
    )


@lru_cache(maxsize=None)
def get_domain_profile(domain: Optional[str] = None) -> DomainProfile:
    """Load (and cache) the profile for `domain`. Falls back to healthcare with a
    warning if the requested profile file is missing — mirrors the compliance
    agent's `state.get("domain", "healthcare")` default so behaviour is uniform."""
    domain = domain or DEFAULT_DOMAIN
    path = settings.profiles_dir / f"{domain}.json"
    if not path.exists():
        if domain != DEFAULT_DOMAIN:
            logger.warning("No profile for domain %r — falling back to %r", domain, DEFAULT_DOMAIN)
            return get_domain_profile(DEFAULT_DOMAIN)
        raise FileNotFoundError(f"Required default profile missing: {path}")
    return _build(json.loads(path.read_text(encoding="utf-8")))
```

> `lru_cache` makes profiles effectively singletons; tests that write fixture profiles must call `get_domain_profile.cache_clear()` (handled in Task 8).

- [ ] **Step 3: Commit**

```bash
git add adapt_ai/config.py adapt_ai/domain/profiles/__init__.py
git commit -m "feat: add DomainProfile model + cached loader (config layer for domain adaptivity)"
```

---

## Task 2: Author the healthcare profile (verbatim regression anchor)

**Files:** Create `adapt_ai/domain/profiles/healthcare.json`.

- [ ] **Step 1: Copy current strings verbatim**

Create `adapt_ai/domain/profiles/healthcare.json`. The `personas`, `labels`, `disclaimer`, and `hallucination_lexicon` MUST reproduce today's exact text from `primary.py`, `quality.py`, `rat.py`, and `graph.py` so healthcare behaviour is unchanged.

```json
{
  "domain": "healthcare",
  "display_name": "Healthcare",
  "labels": {
    "query": "Clinical question",
    "context": "Retrieved clinical context",
    "quality_context": "Clinical context used"
  },
  "personas": {
    "primary": "You are an expert clinical diagnostic assistant supporting healthcare providers.\n\nYour role:\n1. Analyse patient presentations and medical history.\n2. Generate evidence-based differential diagnoses.\n3. Recommend appropriate diagnostic work-ups.\n4. Suggest treatment considerations based on established guidelines.\n\nWhen answering a multiple-choice question (A / B / C / D / E):\n- Reason step-by-step through the clinical scenario.\n- Eliminate incorrect options explicitly.\n- End your response with exactly: ANSWER: X\n  (where X is the single letter of the best choice).\n\nYou are providing decision support for qualified healthcare providers — not making diagnoses.",
    "quality": "You are a medical quality assurance specialist. Evaluate a clinical response for accuracy.\n\nGiven the original question and the AI response, assess:\n1. Does the response directly address the question?\n2. Is the clinical reasoning sound and consistent with medical evidence?\n3. If an answer choice (A/B/C/D/E) is stated, is it plausible given the clinical picture?\n4. Are there any hallucinations, contradictions, or unsupported claims?\n\nRespond ONLY with a JSON object in this exact format:\n{\n  \"passed\": true or false,\n  \"score\": 0.0-1.0,\n  \"issues\": [\"issue1\", \"issue2\"],\n  \"feedback\": \"brief corrective feedback if failed, else empty string\"\n}\n\nScore >= 0.85 -> passed=true. Be strict: fail if the response contains factual errors, omits critical safety information, contradicts established clinical guidelines, makes unsupported claims, or has any issue rated \"Major\" or \"Critical\". Scores of 0.6-0.84 should be passed=false with corrective feedback.",
    "rat_decompose": "You are a medical reasoning assistant. Break down a clinical question into 2-3 focused sub-questions to guide targeted information retrieval. Output only the sub-questions, one per line.",
    "rat_synthesis": "You are a clinical reasoning expert. Use the retrieved medical information and your reasoning to answer the clinical question accurately. Think step by step. If the question has answer choices (A/B/C/D/E), conclude with 'ANSWER: X' where X is the letter of the best choice."
  },
  "disclaimer": "*AI-generated clinical decision support. Healthcare providers must verify all recommendations.*",
  "regulations_file": "healthcare.json",
  "vector_collection": "clinical_knowledge",
  "ontology_namespace": "healthcare",
  "hallucination_lexicon": {
    "suffix_pattern": "\\b[A-Z][a-z]+(?:mycin|cillin|prazole|olol|pine|statin|pril|sartan|mab|nib|vir|tide|zole|oxin)\\b",
    "warning_template": "Unrecognised drug name \"{term}\" — verify accuracy",
    "valid_terms": ["__PASTE_THE_EXISTING_VALID_DRUGS_SET_HERE__"],
    "false_positives": ["medicine", "routine", "baseline", "pipeline", "outline", "guideline", "timeline", "crystalline", "membrane", "cocaine", "codeine"]
  }
}
```

> **Implementation note:** replace `__PASTE_THE_EXISTING_VALID_DRUGS_SET_HERE__` with the full lowercase list currently in `quality.py:_VALID_DRUGS` (≈130 entries), copied verbatim, so the drug check is byte-identical for healthcare.

- [ ] **Step 2: Verify it loads and round-trips**

Run: `source venv/bin/activate && python3 -c "from adapt_ai.domain.profiles import get_domain_profile as g; p=g('healthcare'); print(p.display_name, p.vector_collection, p.lexicon.enabled, len(p.lexicon.valid_terms))"`
Expected: `Healthcare clinical_knowledge True 130` (count may differ slightly — must be >100).

- [ ] **Step 3: Commit**

```bash
git add adapt_ai/domain/profiles/healthcare.json
git commit -m "feat: add healthcare domain profile (verbatim current strings, regression anchor)"
```

---

## Task 3: Generic hallucination lexicon checker

**Files:** Create `adapt_ai/domain/lexicon.py`.

- [ ] **Step 1: Write the domain-agnostic checker**

Create `adapt_ai/domain/lexicon.py`:

```python
"""Generic lexicon-based hallucination pre-check, configured per domain.

Replaces the healthcare-specific drug-name check. A domain that defines a
`hallucination_lexicon` in its profile gets a cheap regex pre-screen; a domain
without one gets no pre-check (the LLM quality pass still runs)."""
from __future__ import annotations
import re
from functools import lru_cache

from adapt_ai.domain.profiles import Lexicon


@lru_cache(maxsize=16)
def _compiled(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def check_lexicon(content: str, lexicon: Lexicon) -> list[str]:
    """Return warnings for terms matching the lexicon's suffix pattern that are
    not in `valid_terms` (and not known false positives). Empty if disabled."""
    if not lexicon.enabled:
        return []
    matches = {m.lower() for m in _compiled(lexicon.suffix_pattern).findall(content)}
    unknown = matches - lexicon.valid_terms - lexicon.false_positives
    return [lexicon.warning_template.format(term=t) for t in sorted(unknown)]
```

- [ ] **Step 2: Commit**

```bash
git add adapt_ai/domain/lexicon.py
git commit -m "feat: add generic per-domain lexicon hallucination check"
```

---

## Task 4: Refactor `primary.py` to read the profile

**Files:** Modify `adapt_ai/agents/primary.py`.

- [ ] **Step 1: Delete the hardcoded prompt; load profile per request**

Remove the module-level `_SYSTEM_PROMPT`. Add `from adapt_ai.domain.profiles import get_domain_profile`. Inside `primary_agent`, after reading state:

```python
        profile = get_domain_profile(state.get("domain"))
        user_content = f'{profile.label("query")}:\n{query}'
        if context:
            user_content += f'\n\n{profile.label("context")}:\n{context}'
        # (feedback / revision blocks unchanged — they are domain-neutral)
```

And change the API call's `system=_SYSTEM_PROMPT` → `system=profile.personas["primary"]`.

- [ ] **Step 2: Verify no domain literal remains**

Run: `grep -niE "clinic|patient|medical|diagnos" adapt_ai/agents/primary.py` → expect **no matches** (docstring included — update it to "Primary Domain Agent — domain reasoning via MCP tool calls.").

---

## Task 5: Refactor `quality.py` to profile + generic lexicon

**Files:** Modify `adapt_ai/agents/quality.py`.

- [ ] **Step 1: Delete healthcare-specific lexicon code**

Remove `_VALID_DRUGS`, `_DRUG_SUFFIX_RE`, `_DRUG_FALSE_POSITIVES`, and `_check_drug_names` (they now live in the healthcare profile + `lexicon.py`). Remove the module-level `_QUALITY_SYSTEM`.

- [ ] **Step 2: Load profile + run the generic check**

Add imports: `from adapt_ai.domain.profiles import get_domain_profile` and `from adapt_ai.domain.lexicon import check_lexicon`. Inside `quality_agent`:

```python
        profile = get_domain_profile(state.get("domain"))
        warnings = check_lexicon(primary_response, profile.lexicon)

        evaluation_prompt = (
            f"Original question:\n{query}\n\n"
            f'{profile.label("quality_context")}:\n{context[:800] if context else "None"}\n\n'
        )
        if warnings:
            evaluation_prompt += (
                "[Pre-check flags — verify these in the response below:]\n"
                + "\n".join(f"  • {w}" for w in warnings)
                + "\n\n"
            )
        evaluation_prompt += f"AI response to evaluate:\n{primary_response}"
```

Change `system=_QUALITY_SYSTEM` → `system=profile.personas["quality"]`. Leave the JSON-parse, scoring, and error fallbacks unchanged (domain-neutral) — but replace the literal "well-structured clinical response" fallback feedback strings with "well-structured response".

- [ ] **Step 3: Verify**

Run: `grep -niE "clinic|medical|drug" adapt_ai/agents/quality.py` → no matches (update docstring to drop "clinical").

---

## Task 6: Per-domain retrieval — `rag.py`, `rat.py`, `server.py`, `vector_store.py`

**Files:** Modify those four.

- [ ] **Step 1: Vector store supports per-domain collections**

In `adapt_ai/domain/vector_store.py`, add a collection-keyed accessor that reuses the existing seeded-collection logic **without passing an embedding function** (must match whatever seeded each collection):

```python
    @classmethod
    def for_collection(cls, collection_name: str) -> "VectorStore":
        # cache one VectorStore per collection name; reuse existing connect logic
        ...
```

Keep `VectorStore.get()` as a thin wrapper returning `for_collection(settings.chroma_collection)` so any untouched caller still hits healthcare. *(Match the existing class's construction signature when implementing.)*

- [ ] **Step 2: `rag_retrieve` takes a domain**

In `adapt_ai/mcp_server/tools/rag.py`:

```python
from adapt_ai.domain.profiles import get_domain_profile
from adapt_ai.domain.vector_store import VectorStore

async def rag_retrieve(query: str, n_results: int = 5, domain: str = "healthcare") -> str:
    store = VectorStore.for_collection(get_domain_profile(domain).vector_collection)
    docs = store.query(query.strip(), n_results=n_results)
    return store.format_context(docs)
```

- [ ] **Step 3: `rat_reason` takes a domain; personas + collection from profile**

In `adapt_ai/mcp_server/tools/rat.py`, change the signature to `rat_reason(query, context="", domain="healthcare", max_steps=None)`. Load `profile = get_domain_profile(domain)`. Replace:
- the decompose `system=...` → `profile.personas["rat_decompose"]`
- the decompose user content `f"Clinical question:\n{query}"` → `f'{profile.label("query")}:\n{query}'`
- `store = VectorStore.get()` → `store = VectorStore.for_collection(profile.vector_collection)`
- the synthesis `system=(...)` → `profile.personas["rat_synthesis"]`
- the synthesis user content `f"Retrieved clinical context:\n{combined_context}"` → `f'{profile.label("context")}:\n{combined_context}'`

- [ ] **Step 4: MCP tool wrappers forward `domain`**

In `adapt_ai/mcp_server/server.py`, the `rag_retrieve_tool` and `rat_reason_tool` wrappers must accept a `domain` argument and forward it to `rag_retrieve` / `rat_reason`. *(Match the existing `@mcp.tool` wrapper style; `validate_output_tool` already takes `domain` — mirror it.)*

- [ ] **Step 5: Verify**

Run: `grep -niE "clinic|medical" adapt_ai/mcp_server/tools/rat.py adapt_ai/mcp_server/tools/rag.py` → no matches (docstrings updated).

---

## Task 7: Orchestration — `graph.py` (disclaimer + tool calls) and `router.py`

**Files:** Modify `adapt_ai/agents/graph.py`, `adapt_ai/orchestrator/router.py`.

- [ ] **Step 1: Router reads keyword sets from the profile**

In `adapt_ai/orchestrator/router.py`, change `should_use_rat(query)` → `should_use_rat(query, domain="healthcare")`. Load `profile = get_domain_profile(domain)`; if `profile.rat_keywords` / `profile.rag_keywords` are non-empty, use them; otherwise fall back to the current generic heuristics (long/vignette → RAT; explicit factual lookup → RAG). Move today's clinical regexes into the healthcare profile's `rat_keywords`/`rag_keywords` arrays.

- [ ] **Step 2: `intent_and_retrieve` passes domain to router + tools**

In `adapt_ai/agents/graph.py`, in `make_retrieval_node`:
- `use_rat = should_use_rat(query)` → `use_rat = should_use_rat(query, state.get("domain", "healthcare"))`
- `rat_reason_tool` call args → add `"domain": state.get("domain", "healthcare")`
- `rag_retrieve_tool` call args → add `"domain": state.get("domain", "healthcare")`

- [ ] **Step 3: `aggregate_response` disclaimer from profile**

In `aggregate_response`, replace the hardcoded clinical disclaimer block with:

```python
    profile = get_domain_profile(state.get("domain"))
    if profile.disclaimer:
        parts.append(f"\n---\n{profile.disclaimer}")
```

Add `from adapt_ai.domain.profiles import get_domain_profile` at the top of `graph.py`.

- [ ] **Step 4: Verify**

Run: `grep -rniE "clinic|patient|medical|hipaa|drug|healthcare" adapt_ai/agents/ adapt_ai/orchestrator/ adapt_ai/mcp_server/tools/ --include=*.py` → **no matches** (outside docstrings/comments, which should also be cleaned). This grep is the Bar-3 invariant.

---

## Task 8: Tests — regression + second-domain proof + no-hardcoding guard

**Files:** Create `tests/test_adapt_ai/test_domain_profile.py`, `tests/test_adapt_ai/fixtures/legal_profile.json`.

- [ ] **Step 1: Minimal legal fixture profile** (proves a 2nd domain needs no agent-code change)

`tests/test_adapt_ai/fixtures/legal_profile.json`: same shape as healthcare but legal personas/labels/disclaimer, `"vector_collection": "legal_knowledge"`, no lexicon. (Short personas are fine — this is a wiring test, not a quality test.)

- [ ] **Step 2: Tests**

Create `tests/test_adapt_ai/test_domain_profile.py` covering:
1. **Loader parity:** `get_domain_profile("healthcare")` exposes the verbatim primary/quality personas (assert key phrases) + `vector_collection == "clinical_knowledge"`.
2. **Lexicon parity:** `check_lexicon("Patient given Zzytomycin.", healthcare.lexicon)` flags `zzytomycin`; a real drug ("aspirin") does not flag; disabled lexicon returns `[]`.
3. **Primary threading:** with the legal fixture loaded (monkeypatch `settings.profiles_dir` to include the fixture, then `get_domain_profile.cache_clear()`), `primary_agent` on `make_state(domain="legal")` sends the **legal** persona as `system` and uses the legal query label — asserted via the `FakeAnthropic` recorded `calls`. Same code path, zero agent edits.
4. **Disclaimer threading:** `aggregate_response` appends the legal disclaimer for `domain="legal"` and the clinical one for `domain="healthcare"`.
5. **No-hardcoding guard (the Bar-3 invariant):** walk `adapt_ai/agents/*.py`, `adapt_ai/orchestrator/*.py`, `adapt_ai/mcp_server/tools/*.py`; assert none contains the banned substrings (`clinical`, `patient`, `medical`, `hipaa`, `drug`, `diagnos`) in code or strings. This test *fails until the refactor is complete* and *guards against regressions forever*.

- [ ] **Step 3: Run the full suite**

Run: `source venv/bin/activate && python3 -m pytest tests/test_adapt_ai/ -v`
Expected: all prior 9 + new tests PASS.

- [ ] **Step 4: Healthcare behavioural regression (live, cheap)**

Run: `source venv/bin/activate && python3 scripts/run_clinical_benchmark.py --questions 3 --no-bertscore`
Expected: completes; ADAPT-AI responses are clinically framed and carry the unchanged disclaimer — i.e. healthcare behaviour is visibly identical to pre-refactor.

- [ ] **Step 5: Commit**

```bash
git add tests/test_adapt_ai/test_domain_profile.py tests/test_adapt_ai/fixtures/legal_profile.json
git commit -m "test: lock domain-profile threading + no-hardcoding (Bar-3) invariant"
```

---

## Task 9: Refactor commit for the agent/tool changes

- [ ] **Step 1: Commit the refactor**

```bash
git add adapt_ai/agents/primary.py adapt_ai/agents/quality.py adapt_ai/agents/graph.py \
        adapt_ai/orchestrator/router.py adapt_ai/mcp_server/tools/rat.py \
        adapt_ai/mcp_server/tools/rag.py adapt_ai/mcp_server/server.py \
        adapt_ai/domain/vector_store.py
git commit -m "refactor: drive all domain text/resources from DomainProfile (config-only adaptivity)"
```

> Commit ordering note: Tasks 4–7 leave the tree red (the no-hardcoding test in Task 8 fails) until all are done. Implement Tasks 1–7, then add Task 8 tests, then make Task 9's refactor commit and Task 8's test commit together so no commit is broken. (Or stage everything and split into the two commits at the end.)

---

## Self-Review

**1. Vision coverage:**
- "Config file for all domain configurability" → `DomainProfile` (Task 1–2). ✅
- "Domain drives which ontology / knowledge / tools" → vector_collection + ontology_namespace + regulations_file in profile; rag/rat select per domain (Task 6). ✅
- Correction surfaced: domain is **explicit** (`state["domain"]`), not query-detected; router only does RAT/RAG. Documented in Goal. ✅

**2. Bar-3 (the paper's claim) is now testable:** Task 8 Step 2.5 is a programmatic no-hardcoding guard; after this refactor a new domain = 1 profile JSON + 4 data artifacts, **zero** agent-code lines. The legal fixture test proves the switch works with no code edits.

**3. Regression safety:** healthcare profile is verbatim (Task 2); existing 9 tests must pass (Task 8.3); a 3-question live healthcare run confirms identical framing + disclaimer (Task 8.4). The `lru_cache` gotcha is called out (cache_clear in tests).

**4. Placeholder scan:** one intentional placeholder — the `_VALID_DRUGS` paste in Task 2 (explicit instruction to copy the existing ≈130-item set verbatim). Everything else is real code. The few "match existing signature" notes (vector_store/server) are because those internals weren't read line-by-line; the integration contract is exact.

**5. Out of scope (YAGNI):** query→domain auto-classifier; Neo4j migration; renaming `clinical_knowledge`; UI. Legal/finance *data* (regs JSON, benchmark, corpus, ontology) is the separate dataset task (`2026-05-31-legal-finance-dataset-spec.md`).

---

## Notes for downstream
- After this refactor, the legal/finance ports = author `legal.json`/`finance.json` profiles + their 4 data artifacts; capture the `git diff --stat` showing zero lines under `adapt_ai/agents/**`, `adapt_ai/orchestrator/**`, `adapt_ai/mcp_server/tools/**` → the portability/Bar-3 table.
- Seeding `legal_knowledge` / `finance_knowledge` ChromaDB collections must use the **same embedder** that seeded `clinical_knowledge` (see `scripts/seed_vector_db.py`); the profile only names the collection.
- Re-run full healthcare clinical + MedQA after the refactor to confirm no metric drift, then proceed to the quality-agent ablation across domains.

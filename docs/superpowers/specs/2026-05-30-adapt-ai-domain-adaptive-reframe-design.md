# ADAPT-AI Journal Paper — Contribution Reframe & Multi-Domain Empirical Validation

**Date:** 2026-05-30
**Status:** Design (brainstorming) — pending spec review
**Author:** Patryk Slomka (JADS MSc), supervisors Dr. Damian Tamburri, Dr. Marco Tonnarelli
**Source thesis:** `/mnt/c/Users/patry/Desktop/Thesis/FinalThesisReport_PatrykSlomka_2049498.pdf` — *"ADAPT-AI: A Configurable Multi-Agent Architecture for AI Factories Using Model Context Protocol and Modular Building Blocks"* (Dec 15, 2025, defended 8.5/10).

**Context:** The thesis is done. This document scopes the **IEEE journal paper** that extends it. No hard deadline, but the work should produce demonstrable progress.

---

## 1. Why this document exists

The thesis claim, as held informally, drifted to:

> "Using agents + MCP + ontology produces *better results* than a plain LLM."

A back-to-basics interview exposed two problems that would sink the paper at peer review:

1. **On the metric a reader assumes ("better" = accuracy), the project's own data refutes it.** MedQA: −2% vs baseline. Clinical overall: +16% but *not statistically significant* (Wilcoxon p ≥ 0.05, n=30). The system is also 3.86× slower and more expensive.
2. **The identity-defining claim — domain adaptivity — was empirically unproven (n=1 domain: healthcare).**

Two facts from the thesis itself reframe everything:

- **The thesis was qualitative.** Its evidence base was a systematic literature review (75 papers) + Action Design Research + semi-structured practitioner interviews. Secondary Objective 4 is literally *"creating the **qualitative** evaluation."* The thesis never ran a quantitative benchmark. Its abstract claim that "ontologies improve accuracy and reduce hallucinations" is **literature-derived (RQ2.2)**, not self-measured.
- **The thesis's spine is adaptivity, not accuracy.** RQ1 asks how MCP/modular blocks "enable **adaptive AI system design** ... across different domains." Research Gap #3 is the "lack of generalizable frameworks with systematic domain adaptation ... across industries." The Significance section explicitly names "healthcare, **finance and legal**."

**So the journal's job is not to re-argue the thesis — it is to add the empirical/quantitative layer the thesis lacked, and to do it honestly.** The honest empirical headline is **safety/compliance**, demonstrated to **replicate across multiple regulated domains** via **configuration-only** adaptation. That fills Research Gap #3 with evidence and is faithful to RQ1.

---

## 2. The reframed contribution

Three previously-conflated claims, now separated into **scope**, **content**, and an **honest trade-off**:

| Axis | Claim | How evidenced |
|------|-------|---------------|
| **Scope** | ADAPT-AI is **domain-adaptive across regulated, high-stakes domains** (NOT "any domain"). | Config-only ports across **3 domains**: healthcare → legal → finance. |
| **Content (empirical headline)** | The decoupled compliance/quality layer delivers **architecturally-enforced safety/compliance** a monolithic LLM does not. | Existing data (1.0 vs 0.82); must **replicate** in legal and finance. |
| **Honest trade-off** | Accuracy is at **parity / no-regression**, at an explicit **latency and $ cost**. | Defensive only. Note the nuance vs the thesis's literature-claim that ontologies *improve* accuracy: our own data shows parity. |

**Paper spine (one sentence):**

> *ADAPT-AI is a domain-adaptive MCP architecture whose demonstrated benefit is architecturally-enforced safety/compliance — shown to replicate as the same architecture is ported across three regulated domains (healthcare, legal, finance) by configuration changes alone, at accuracy parity and an explicit latency/cost overhead.*

### Scope discipline (must hold throughout)
- Never write "any domain" / unqualified "domain-agnostic" as an empirical claim. Use "domain-adaptive across regulated, high-stakes domains."
- Never present accuracy as a win — only as no-regression.
- The empirical headline is **safety**, replicated across all three domains.

---

## 3. The core experiment: one architecture, three regulated domains

The project's iron architectural rule (verify it still holds in current `adapt_ai/` code): **agents never import domain code; all domain access flows through MCP tools/resources** (regulations JSON, ontology, vector corpus). This rule *is* the falsifiable basis of the adaptivity claim.

Healthcare already exists. The experiment adds **legal** and **finance** as two further regulated domains, each brought up through configuration only.

### 3.1 Falsifiable test ("Bar 3" — committed to)
For **each** new domain, evaluate on three axes:

1. **Architectural (binary):** can the domain be stood up by changing **only** config/resource files — **zero lines** in agent/orchestration code? A git diff is the proof. *If agent/orchestration logic must change, the adaptivity claim is FALSIFIED for that domain and reported as such.*
2. **Cost of adaptation (quantitative portability):** files changed, lines changed, wall-clock hours per domain. The concrete "how adaptive?" number — and a trend across domains 2 and 3.
3. **Effect replication:** does the safety advantage (≈1.0 vs ≈0.82 gap) reappear in legal and in finance? Replication across *three* domains is far stronger evidence than across one or two.

### 3.2 Honest-reporting commitment
All outcomes reported truthfully, per domain: adaptivity holds or broke at X; safety replicates or does not; the quality-agent ablation verdict (§4), whatever it is. A partial/negative result is acceptable and publishable; a dishonest positive is not.

### 3.3 Uniform benchmark schema across all three domains
A single, shared benchmark structure (the existing `clinical_reasoning_benchmark.json` shape: `query`, `reference_answer`, `required_concepts`, `critical_concepts`, `hallucination_patterns`, `category`) is used for **all three** domains, so the evaluator runs unchanged. **This uniformity is itself part of the adaptivity evidence** — same harness, same schema, only the domain config differs.

Work per domain = a full config stack: regulation schema (JSON), ontology, vector corpus, and a labelled benchmark set. The user has access to the candidate sources below; the open task is **structuring** them into the uniform schema for healthcare, legal, **and** finance.

| Domain | Regulations | Ontology | Corpus | Benchmark seed (prefer existing gold answers) |
|--------|-------------|----------|--------|-----------------------------------------------|
| Healthcare (exists) | HIPAA/FDA JSON | current (NetworkX/graph) | clinical ChromaDB | existing 30 + MedQA |
| Legal | e.g. GDPR / contract / unauthorized-practice rules | LKIF or graph placeholder | EUR-Lex / CUAD | **LegalBench**, **CUAD** (gold clause labels) |
| Finance | e.g. MiFID II / SEC / FINRA suitability | finance ontology / placeholder | filings / regulatory text | **FinQA / ConvFinQA / TAT-QA** (gold answers) |

**Construct-validity guardrail:** prefer datasets with **existing gold answers** over self-authored reference answers. Self-authoring legal/finance references without domain expertise is a validity threat (mirrors the thesis's Threats to Validity §7.3). Where references must be authored, document the procedure; consider lightweight expert review, echoing the thesis's practitioner-interview method.

**Safety must be expressible per domain.** The compliance/safety angle must be scorable in each domain, e.g.: clinical (no unsafe advice, disclaimers) → legal (refuse binding advice / unauthorized practice, cite correct statutes) → finance (suitability, mandatory disclosures, no guaranteed-return claims). If safety cannot be operationalised in a domain, that domain is a poor fit — flag early.

---

## 4. Quality-agent ablation

**Problem:** the quality agent fired **0 times across 30 questions** (incl. `hallucination_trap`). A reviewer will ask why it exists. Concluding "redundant" now would repeat the n=1 error in reverse.

**Resolution:** decide its fate by **ablation**, across **all three** domains:
- Run the pipeline **with** and **without** the quality agent, per domain; measure whether outputs/scores change.
- Report the data-driven verdict: **redundant** (parsimony finding) / **domain-dependent** (e.g. fires in finance, not healthcare — a notable result) / **needed**.
- Separately verify the gate isn't merely mis-tuned (threshold) so "redundant" is not a hidden bug.

This converts a perceived weakness into a respectable minimal-sufficient-agent-set / ablation result.

---

## 5. Integrity fixes (paper cannot ship without these)

Verify each against current code first (memories are ~17 days old).

1. **Cost tracking (`adapt_ai`):** currently `null`. The latency/$ trade-off claim is unsupportable without it. Confirm the usage accumulator (`adapt_ai/llmops/usage.py`, keyed by `session_id`) captures **all** internal LLM calls (RAT steps, retries, quality agent) and surfaces a per-query total. (Open item #3.)
2. **MedQA null-answer bug (Q27–44):** a data-integrity hole. Fix and re-run, or the accuracy table is invalid — required because accuracy is kept as the no-regression defense (§2).
3. **Statistical power:** n=30 / p ≥ 0.05 is too weak for any claim *except* safety (large effect). Report effect sizes + CIs; consider larger n; mark non-significant results plainly.

---

## 6. Evaluation methodology (target tables)

- **Per-domain, per-category** (×3 domains): ADAPT-AI vs monolithic baseline on safety, concept recall/F1, ROUGE-L, hallucination count, overall, latency, **cost**.
- **Cross-domain replication:** safety advantage side-by-side across healthcare / legal / finance — the headline figure.
- **Portability table:** files/LOC/hours per domain + explicit "0 agent-code lines changed" + diff references.
- **Ablation table:** with/without quality agent × {healthcare, legal, finance}.
- **Statistics:** paired tests (Wilcoxon / McNemar), effect sizes, CIs; non-significant results marked.

---

## 7. Success & falsification criteria

**Success** if the paper can state, with evidence, *either*:
- "Adaptivity holds: legal and finance added via config only; safety advantage replicated across all three domains" — the strong positive result; **or**
- "Adaptivity holds architecturally, but the safety effect is domain-dependent" — still an honest, publishable finding about the architecture's limits.

**Falsified (and reported as such)** if standing up a new domain required changing agent/orchestration code — the decoupling does not deliver adaptivity.

The user's explicit wish: *fair and interesting.* Negative/partial results are in scope.

---

## 8. Reusing the thesis (do not rewrite from scratch)

The journal should **port and tighten** thesis text, not regenerate it. Reusable, near-ready sections (verify exact wording in the PDF):
- **Abstract / Introduction / Problem Statement** (thesis §1) — condense.
- **RQs + Research Gap + Objectives** (thesis §1.4, §3.5, §3.6) — RQ1 and Gap #3 directly motivate the empirical study; reframe RQs as the journal's empirically-tested questions.
- **Background** (thesis §2: RAG, RAT, ontologies, building blocks, MCP, agentic AI, AI factories) — reusable related-work spine.
- **Domain applications** (thesis §3.4.2 healthcare, §3.4.3 financial systems) — directly support the three-domain choice.
- **Theoretical framework / 6-layer architecture** (thesis §5.2) — the architecture description.
- **Threats to Validity** (thesis §7.1–7.3) — base for the journal's Limitations; the journal's new empirical layer *answers* the prior external-validity threat.

**New for the journal (what the thesis lacked):** the quantitative multi-domain benchmark, the config-only portability evidence, the safety-replication result, the quality-agent ablation, and the honest accuracy/cost trade-off — i.e. everything in §§2–6 here.

---

## 9. Out of scope (YAGNI for this paper)

- React/Next.js UI (Layer 1), Auth0 (open items #6, #7).
- Neo4j migration unless a domain genuinely needs it (#4) — keep ontology parity across domains.
- A2A protocol comparison (#8), MCP context-injection security study (#9), longitudinal production deployment (#11).
- A **fourth** domain. Three regulated domains are sufficient and well-motivated by the thesis.
- Any attempt to make accuracy a *win*.

---

## 10. Immediate next steps (feed into implementation plan)

1. Verify current-code status: no-domain-import rule, cost tracking, MedQA parsing, quality-gate behaviour.
2. Select & licence-check legal + finance datasets (prefer gold-answer sources); design the uniform benchmark schema; restructure the healthcare set to match.
3. Build legal config stack (regulations / ontology / corpus / benchmark); then finance.
4. Port + run each domain: capture config-only diff, portability metrics, per-domain results.
5. Run the quality-agent ablation across all three domains.
6. Fix integrity items (cost, MedQA, stats) before drafting results tables.
7. Draft results around §2 spine; port thesis text per §8.

> Detailed sequencing, owners, and verification steps belong in the implementation plan (writing-plans skill), not here.

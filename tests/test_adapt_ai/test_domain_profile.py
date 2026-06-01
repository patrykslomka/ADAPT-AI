"""Tests for the DomainProfile config layer (Bar-3 invariant + threading + regression).

Five test groups:
1. Loader parity   — healthcare profile exposes verbatim current strings
2. Lexicon parity  — check_lexicon behaves identically to the old _check_drug_names
3. Primary threading — domain switch changes system prompt with zero agent-code edits
4. Disclaimer threading — aggregate_response uses the profile disclaimer
5. Bar-3 guard     — no domain-specific literals remain in agent/orchestrator/tools code
"""
from __future__ import annotations
import asyncio
import json
import re
from pathlib import Path

import pytest

from adapt_ai.domain.profiles import get_domain_profile, DomainProfile
from adapt_ai.domain.lexicon import check_lexicon


FIXTURES = Path(__file__).parent / "fixtures"
ADAPT_AI_ROOT = Path(__file__).parent.parent.parent / "adapt_ai"


# ── 1. Loader parity ──────────────────────────────────────────────────────────

def test_healthcare_profile_loads():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert isinstance(p, DomainProfile)
    assert p.domain == "healthcare"
    assert p.vector_collection == "clinical_knowledge"
    assert p.display_name == "Healthcare"


def test_healthcare_primary_persona_verbatim():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert "clinical diagnostic assistant" in p.personas["primary"]
    assert "ANSWER: X" in p.personas["primary"]
    assert "healthcare providers" in p.personas["primary"]


def test_healthcare_quality_persona_verbatim():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert "medical quality assurance" in p.personas["quality"]
    assert '"passed": true or false' in p.personas["quality"]
    assert "0.85" in p.personas["quality"]


def test_healthcare_rat_personas_verbatim():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert "medical reasoning assistant" in p.personas["rat_decompose"]
    assert "clinical reasoning expert" in p.personas["rat_synthesis"]


def test_healthcare_labels():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert p.label("query") == "Clinical question"
    assert p.label("context") == "Retrieved clinical context"
    assert p.label("quality_context") == "Clinical context used"


def test_healthcare_disclaimer():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    assert "AI-generated clinical decision support" in p.disclaimer
    assert "Healthcare providers must verify" in p.disclaimer


def test_missing_profile_falls_back_to_healthcare(tmp_path, monkeypatch):
    """A missing profile falls back to healthcare, not a crash."""
    get_domain_profile.cache_clear()
    from adapt_ai import config as cfg
    monkeypatch.setattr(cfg.settings, "profiles_dir", tmp_path)
    # Copy healthcare.json so the fallback itself works
    import shutil
    src = ADAPT_AI_ROOT / "domain" / "profiles" / "healthcare.json"
    shutil.copy(src, tmp_path / "healthcare.json")
    result = get_domain_profile("nonexistent_domain")
    assert result.domain == "healthcare"
    get_domain_profile.cache_clear()


# ── 2. Lexicon parity ─────────────────────────────────────────────────────────

def test_unknown_drug_suffix_flagged():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    warnings = check_lexicon("Patient given Zzytomycin.", p.lexicon)
    assert any("zzytomycin" in w.lower() for w in warnings)


def test_known_drug_not_flagged():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    warnings = check_lexicon("Administer aspirin 300 mg.", p.lexicon)
    # "aspirin" does not match the suffix pattern (no listed suffix) — no warnings
    assert warnings == []


def test_false_positive_not_flagged():
    get_domain_profile.cache_clear()
    p = get_domain_profile("healthcare")
    # "Guideline" matches the -line suffix pattern but is a known false positive
    warnings = check_lexicon("See the Guideline for details.", p.lexicon)
    assert all("guideline" not in w.lower() for w in warnings)


def test_disabled_lexicon_returns_empty():
    """A profile with no suffix_pattern returns no warnings."""
    from adapt_ai.domain.profiles import Lexicon
    empty = Lexicon()
    assert not empty.enabled
    assert check_lexicon("Zzytomycin overdose.", empty) == []


# ── 3. Primary threading — domain switch with zero agent-code edits ──────────

@pytest.mark.asyncio
async def test_primary_agent_uses_legal_persona(tmp_path, monkeypatch):
    """Switch to the legal fixture profile — primary_agent must send the legal
    system persona. No changes to agent code are required, only the profile."""
    get_domain_profile.cache_clear()

    # Point profiles_dir to fixtures so the loader finds legal.json
    import shutil
    from adapt_ai import config as cfg
    monkeypatch.setattr(cfg.settings, "profiles_dir", tmp_path)
    shutil.copy(FIXTURES / "legal_profile.json", tmp_path / "legal.json")
    # Also copy healthcare for the fallback path
    shutil.copy(ADAPT_AI_ROOT / "domain" / "profiles" / "healthcare.json", tmp_path / "healthcare.json")

    from tests.test_adapt_ai.conftest import FakeAnthropic, FakeMCPClient, make_state
    from adapt_ai.agents.primary import make_primary_node

    fake_ant = FakeAnthropic(text="ANSWER: B")
    mcp = FakeMCPClient()
    node = make_primary_node(mcp, fake_ant)

    state = make_state(
        query="Who bears liability for a breach of contract?",
        domain="legal",
    )
    result = await node(state)

    # The system prompt sent to the API must be the legal persona, not healthcare
    calls = fake_ant.messages.calls
    assert calls, "No API call recorded"
    system_sent = calls[0]["system"]
    assert "legal" in system_sent.lower(), "Expected legal persona in system prompt"
    assert "clinical" not in system_sent.lower(), "Healthcare persona leaked into legal domain"

    # The user message should use the legal query label
    user_msg = calls[0]["messages"][0]["content"]
    assert user_msg.startswith("Legal question:")

    get_domain_profile.cache_clear()


@pytest.mark.asyncio
async def test_primary_agent_uses_healthcare_persona():
    """Healthcare domain still sends the healthcare persona — regression guard."""
    get_domain_profile.cache_clear()

    from tests.test_adapt_ai.conftest import FakeAnthropic, FakeMCPClient, make_state
    from adapt_ai.agents.primary import make_primary_node

    fake_ant = FakeAnthropic(text="ANSWER: A")
    mcp = FakeMCPClient()
    node = make_primary_node(mcp, fake_ant)

    state = make_state(query="What is first-line for hypertension?", domain="healthcare")
    await node(state)

    system_sent = fake_ant.messages.calls[0]["system"]
    assert "clinical diagnostic assistant" in system_sent
    assert user_msg_starts_with(fake_ant, "Clinical question:")


def user_msg_starts_with(fake_ant, prefix: str) -> bool:
    return fake_ant.messages.calls[0]["messages"][0]["content"].startswith(prefix)


# ── 4. Disclaimer threading ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_aggregate_response_healthcare_disclaimer():
    """aggregate_response appends the healthcare disclaimer for domain='healthcare'."""
    from adapt_ai.agents.graph import aggregate_response
    from adapt_ai.llmops.usage import new_accumulator
    get_domain_profile.cache_clear()

    sid = "test-disclaimer-hc"
    new_accumulator(sid)
    state = {
        "session_id": sid,
        "domain": "healthcare",
        "primary_response": "Take aspirin.",
        "compliance_result": {},
        "quality_result": {"score": 0.9},
        "agent_statuses": {},
    }
    result = await aggregate_response(state)
    assert "Healthcare providers must verify" in result["final_response"]


@pytest.mark.asyncio
async def test_aggregate_response_legal_disclaimer(tmp_path, monkeypatch):
    """aggregate_response appends the legal disclaimer for domain='legal'."""
    get_domain_profile.cache_clear()

    import shutil
    from adapt_ai import config as cfg
    monkeypatch.setattr(cfg.settings, "profiles_dir", tmp_path)
    shutil.copy(FIXTURES / "legal_profile.json", tmp_path / "legal.json")
    shutil.copy(ADAPT_AI_ROOT / "domain" / "profiles" / "healthcare.json", tmp_path / "healthcare.json")

    from adapt_ai.agents.graph import aggregate_response
    from adapt_ai.llmops.usage import new_accumulator

    sid = "test-disclaimer-legal"
    new_accumulator(sid)
    state = {
        "session_id": sid,
        "domain": "legal",
        "primary_response": "The defendant breached the duty.",
        "compliance_result": {},
        "quality_result": {"score": 0.9},
        "agent_statuses": {},
    }
    result = await aggregate_response(state)
    assert "legal research support" in result["final_response"].lower()
    assert "Healthcare providers" not in result["final_response"]

    get_domain_profile.cache_clear()


# ── 5. Bar-3 guard — no domain hardcoding in agent/orchestrator/tools ─────────

_BANNED = re.compile(
    r"\b(clinical|patient|medical|hipaa|drug|diagnos)",
    re.IGNORECASE,
)

_SCAN_DIRS = [
    ADAPT_AI_ROOT / "agents",
    ADAPT_AI_ROOT / "orchestrator",
    ADAPT_AI_ROOT / "mcp_server" / "tools",
]


def _scan_files():
    for d in _SCAN_DIRS:
        yield from d.glob("*.py")


def test_no_domain_hardcoding_in_agent_code():
    """Bar-3 invariant: no domain-specific literals in agents/orchestrator/tools.

    This test fails if anyone re-introduces hardcoded domain text.
    All domain strings must live in adapt_ai/domain/profiles/*.json.
    """
    violations: list[str] = []
    for path in _scan_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if _BANNED.search(line):
                violations.append(f"{path.relative_to(ADAPT_AI_ROOT.parent)}:{lineno}: {line.strip()}")

    assert not violations, (
        "Domain-specific literals found in agent/orchestrator/tools code "
        "(Bar-3 violation). Move them to a DomainProfile JSON:\n"
        + "\n".join(violations)
    )

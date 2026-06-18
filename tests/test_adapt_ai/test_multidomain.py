"""Multi-domain port coverage - healthcare / legal / finance.

These tests lock the three production domain profiles, their regulation rule
sets, the per-domain compliance validation tool, and per-domain routing.
Proving legal+finance work through the *same* code paths as healthcare is the
evidence that a new regulated domain is a config-only addition (zero agent-code
changes - see also the Bar-3 guard in test_domain_profile.py).
"""
from __future__ import annotations
import json
import re

import pytest

from adapt_ai.config import settings
from adapt_ai.domain.profiles import get_domain_profile, DomainProfile
from adapt_ai.orchestrator.router import should_use_rat
from adapt_ai.mcp_server.tools.validation import validate_output

DOMAINS = ["healthcare", "legal", "finance"]
PERSONA_KEYS = {"primary", "quality", "rat_decompose", "rat_synthesis"}
LABEL_KEYS = {"query", "context", "quality_context"}
_VALID_SEVERITIES = {"critical", "high", "medium", "low"}


#  Profiles 

@pytest.mark.parametrize("domain", DOMAINS)
def test_profile_loads_and_is_complete(domain):
    get_domain_profile.cache_clear()
    p = get_domain_profile(domain)
    assert isinstance(p, DomainProfile)
    assert p.domain == domain
    assert p.display_name
    assert PERSONA_KEYS <= set(p.personas), f"{domain} missing personas"
    assert all(p.personas[k].strip() for k in PERSONA_KEYS)
    assert LABEL_KEYS <= set(p.labels), f"{domain} missing labels"
    assert p.disclaimer.strip()
    assert p.vector_collection


def test_each_domain_has_a_distinct_vector_collection():
    """Domains must not share a vector collection - they index different corpora."""
    get_domain_profile.cache_clear()
    collections = {d: get_domain_profile(d).vector_collection for d in DOMAINS}
    assert len(set(collections.values())) == len(DOMAINS), collections


#  Regulation rule sets 

@pytest.mark.parametrize("domain", DOMAINS)
def test_regulations_file_exists_and_is_well_formed(domain):
    get_domain_profile.cache_clear()
    p = get_domain_profile(domain)
    path = settings.regulations_dir / p.regulations_file
    assert path.exists(), f"missing regulations file for {domain}: {path}"

    data = json.loads(path.read_text(encoding="utf-8"))
    rules = data.get("regulations", [])
    assert rules, f"{domain} regulations file has no rules"
    for rule in rules:
        assert rule.get("id"), f"{domain} rule missing id"
        assert rule.get("severity") in _VALID_SEVERITIES, rule.get("id")
        assert rule.get("description"), f"{rule['id']} missing description"
        for pattern in rule.get("patterns", []):
            re.compile(pattern)  # must be a valid regex or this raises


#  Compliance validation (the safety bar) - per domain ─

@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["legal", "finance"])
async def test_disclosed_ssn_is_rejected(domain):
    """A raw SSN is a critical privacy/confidentiality violation in both new
    domains (LEGAL-001 / FIN-001) and must hard-fail compliance."""
    result = await validate_output(
        "The client SSN is 123-45-6789 and the matter is ongoing.",
        domain=domain,
    )
    assert result["passed"] is False
    assert result["status"] == "rejected"
    assert any(i["severity"] == "critical" for i in result["issues"])


@pytest.mark.asyncio
async def test_finance_guaranteed_return_is_flagged():
    """FIN-002 (no guaranteed returns) is a high-severity warning."""
    result = await validate_output(
        "This fund guarantees a 12% annual return for every investor.",
        domain="finance",
    )
    assert result["status"] == "warning"
    assert any(i["rule_id"] == "FIN-002" for i in result["issues"])


@pytest.mark.asyncio
async def test_legal_guaranteed_outcome_is_flagged():
    """LEGAL-002 (no guaranteed outcome) is a high-severity warning."""
    result = await validate_output(
        "Retain us and I guarantee we will win this case at trial.",
        domain="legal",
    )
    assert result["status"] == "warning"
    assert any(i["rule_id"] == "LEGAL-002" for i in result["issues"])


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", DOMAINS)
async def test_clean_educational_text_passes(domain):
    """Benign, identifier-free educational text passes compliance in every domain
    (the 'educational' exception waives the disclaimer-required rules)."""
    result = await validate_output(
        "This is a general informational overview provided for educational purposes.",
        domain=domain,
    )
    assert result["passed"] is True
    assert not any(i["severity"] in {"critical", "high"} for i in result["issues"])


#  Routing - domain keyword sets drive RAT vs RAG 

def test_legal_vignette_routes_to_rat():
    assert should_use_rat("The plaintiff sued the defendant for breach.", "legal") is True


def test_finance_vignette_routes_to_rat():
    assert should_use_rat(
        "The investor's portfolio lost value after the transaction settled.", "finance"
    ) is True


def test_finance_ethics_query_routes_to_rat():
    assert should_use_rat("Is it appropriate to recommend this fund to a client?", "finance") is True


def test_legal_factual_lookup_routes_to_rag():
    assert should_use_rat("What is the statute of limitations?", "legal") is False

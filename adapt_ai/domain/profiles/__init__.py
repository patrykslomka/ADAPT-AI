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
    valid_terms: frozenset = field(default_factory=frozenset)
    suffix_pattern: str = ""
    false_positives: frozenset = field(default_factory=frozenset)
    warning_template: str = 'Unrecognised term "{term}" — verify accuracy'

    @property
    def enabled(self) -> bool:
        return bool(self.suffix_pattern)


@dataclass(frozen=True)
class DomainProfile:
    domain: str
    display_name: str
    labels: dict
    personas: dict
    disclaimer: str
    regulations_file: str
    vector_collection: str
    ontology_path: str        # path to OWL/RDF file, relative to project root
    lexicon: Lexicon = field(default_factory=Lexicon)
    rat_keywords: tuple = ()
    rag_keywords: tuple = ()
    ethics_keywords: tuple = ()
    vignette_keywords: tuple = ()

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
        ontology_path=raw.get("ontology_path", ""),
        lexicon=lexicon,
        rat_keywords=tuple(raw.get("rat_keywords", ())),
        rag_keywords=tuple(raw.get("rag_keywords", ())),
        ethics_keywords=tuple(raw.get("ethics_keywords", ())),
        vignette_keywords=tuple(raw.get("vignette_keywords", ())),
    )


@lru_cache(maxsize=None)
def get_domain_profile(domain: Optional[str] = None) -> DomainProfile:
    """Load (and cache) the profile for `domain`.

    Falls back to healthcare with a warning if the requested profile file is
    missing — mirrors the compliance agent's state.get("domain", "healthcare")
    default so behaviour is uniform.
    """
    domain = domain or DEFAULT_DOMAIN
    path = settings.profiles_dir / f"{domain}.json"
    if not path.exists():
        if domain != DEFAULT_DOMAIN:
            logger.warning("No profile for domain %r — falling back to %r", domain, DEFAULT_DOMAIN)
            return get_domain_profile(DEFAULT_DOMAIN)
        raise FileNotFoundError(f"Required default profile missing: {path}")
    return _build(json.loads(path.read_text(encoding="utf-8")))

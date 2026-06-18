"""Generic lexicon-based hallucination pre-check, configured per domain.

Replaces the healthcare-specific drug-name check. A domain that defines a
`hallucination_lexicon` in its profile gets a cheap regex pre-screen; a domain
without one gets no pre-check (the LLM quality pass still runs).
"""
from __future__ import annotations
import re
from functools import lru_cache

from adapt_ai.domain.profiles import Lexicon


@lru_cache(maxsize=16)
def _compiled(pattern: str) -> "re.Pattern[str]":
    return re.compile(pattern)


def check_lexicon(content: str, lexicon: Lexicon) -> list[str]:
    """Return warnings for terms matching the lexicon's suffix pattern that are
    not in valid_terms (and not known false positives). Empty list if disabled.
    """
    if not lexicon.enabled:
        return []
    matches = {m.lower() for m in _compiled(lexicon.suffix_pattern).findall(content)}
    unknown = matches - lexicon.valid_terms - lexicon.false_positives
    return [lexicon.warning_template.format(term=t) for t in sorted(unknown)]

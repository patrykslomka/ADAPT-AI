"""Intent detection and routing - decides which building block to invoke."""
from __future__ import annotations
import re

from adapt_ai.domain.profiles import get_domain_profile

# Generic fallback keyword sets (used when a domain profile has no keywords).
_GENERIC_RAT = re.compile(
    r"\b(why|mechanism|explain|compare|differentiate|how does|management|"
    r"treatment plan|work-?up|differential|multi-step|complex|reason|"
    r"analyse|analyze)\b",
    re.IGNORECASE,
)
_GENERIC_ETHICS = re.compile(
    r"\b(ethics|policy|fraud|consent|disclose|obligation|"
    r"should (i|you|we)|can i|is it (ok|appropriate|legal|ethical))\b",
    re.IGNORECASE,
)
_GENERIC_RAG = re.compile(
    r"\b(what is|define|list|name|first-?line|dose|contraindication|"
    r"side effect|interaction|normal range|reference value)\b",
    re.IGNORECASE,
)


def _build_re(keywords: tuple) -> "re.Pattern | None":
    if not keywords:
        return None
    return re.compile(
        r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
        re.IGNORECASE,
    )


def should_use_rat(query: str, domain: str = "healthcare") -> bool:
    """Return True if the query warrants multi-step RAT reasoning.

    Long vignette-style questions always use RAT because they require
    integrating multiple pieces of information. Domain keyword sets are read
    from the DomainProfile; generic fallbacks apply when none are set.
    """
    profile = get_domain_profile(domain)

    rat_re = _build_re(profile.rat_keywords) or _GENERIC_RAT
    rag_re = _build_re(profile.rag_keywords) or _GENERIC_RAG
    ethics_re = _build_re(profile.ethics_keywords) or _GENERIC_ETHICS
    vignette_re = _build_re(profile.vignette_keywords)

    # Long vignette-style question → always RAT
    if len(query) > 300:
        return True
    if vignette_re and vignette_re.search(query):
        return True
    # Ethics/compliance always needs RAT even if the query starts with "What is"
    if ethics_re.search(query):
        return True
    # Explicit reasoning keywords → RAT (unless the query is clearly a factual lookup)
    if rat_re.search(query) and not rag_re.search(query):
        return True
    return False

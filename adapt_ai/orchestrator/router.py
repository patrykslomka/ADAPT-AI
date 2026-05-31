"""Intent detection and routing — decides which building block to invoke."""
from __future__ import annotations
import re

# Keywords that signal complex multi-step reasoning → RAT (blocked by RAG keywords)
_RAT_KEYWORDS = re.compile(
    r"\b(why|mechanism|pathophysiology|explain|compare|differentiate|"
    r"how does|management|treatment plan|work-?up|differential|"
    r"multi-step|complex|reason|analyse|analyze)\b",
    re.IGNORECASE,
)

# Ethics/compliance queries always require RAT regardless of surface form.
# "What is the HIPAA policy?" looks like RAG but needs deep reasoning.
_ETHICS_KEYWORDS = re.compile(
    r"\b(ethics|policy|hipaa|fraud|malpractice|consent|disclose|obligation|"
    r"should (i|you|we)|can i|is it (ok|appropriate|legal|ethical))\b",
    re.IGNORECASE,
)

# Indicators of a direct factual lookup → RAG
_RAG_KEYWORDS = re.compile(
    r"\b(what is|define|list|name|which drug|first-?line|dose|contraindication|"
    r"side effect|interaction|normal range|reference value)\b",
    re.IGNORECASE,
)

# Clinical vignettes (long questions with patient data) need RAT
_VIGNETTE_PATTERN = re.compile(
    r"\b(year-old|yo |presents|history of|physical exam|vital signs|"
    r"laboratory|blood pressure|mmhg|pulse|temperature|oxygen)\b",
    re.IGNORECASE,
)


def should_use_rat(query: str) -> bool:
    """Return True if the query warrants multi-step RAT reasoning.

    Clinical vignettes (USMLE-style) always use RAT because they require
    integrating multiple pieces of information — demographics, vitals, labs,
    and answer choice discrimination.
    """
    # Long vignette-style question → always RAT
    if len(query) > 300 or _VIGNETTE_PATTERN.search(query):
        return True
    # Ethics/compliance always needs RAT even if the query starts with "What is"
    if _ETHICS_KEYWORDS.search(query):
        return True
    # Explicit reasoning keywords → RAT (unless the query is clearly a factual lookup)
    if _RAT_KEYWORDS.search(query) and not _RAG_KEYWORDS.search(query):
        return True
    return False

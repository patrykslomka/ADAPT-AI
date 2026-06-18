# tests/test_adapt_ai/test_baseline_ladder.py
"""Tests for the baseline ladder in run_benchmark.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.run_benchmark import build_baseline_prompt, BASELINE_VARIANTS


def test_ladder_has_five_named_variants():
    assert set(BASELINE_VARIANTS) == {"b0_bare", "b1_disclaimer", "b2_rag", "b3_persona", "full"}


def test_disclaimer_instruction_added_from_b1_onward():
    assert "disclaimer" not in build_baseline_prompt("b0_bare", "healthcare").lower()
    for v in ("b1_disclaimer", "b2_rag", "b3_persona"):
        assert "disclaimer" in build_baseline_prompt(v, "healthcare").lower()


def test_b3_uses_full_domain_persona():
    from adapt_ai.domain.profiles import get_domain_profile
    persona = get_domain_profile("legal").personas["primary"]
    result = build_baseline_prompt("b3_persona", "legal")
    assert result.startswith(persona[:40])

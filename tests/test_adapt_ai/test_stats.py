"""Tests for scipy-backed stats in analyze_results.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.analyze_results import wilcoxon_signed_rank, rank_biserial, holm_correct


def test_wilcoxon_significant_positive_direction():
    """ADAPT consistently higher → significant, positive direction reported."""
    diffs = [0.1, 0.2, 0.05, 0.15, 0.08, 0.12, 0.2, 0.07, 0.09, 0.11]
    W, p, summary = wilcoxon_signed_rank(diffs)
    assert p < 0.05
    assert "ADAPT-AI > Baseline" in summary


def test_rank_biserial_in_unit_range():
    diffs = [0.1, -0.05, 0.2, 0.0, 0.15]
    r = rank_biserial(diffs)
    assert -1.0 <= r <= 1.0


def test_holm_correction_monotone_and_inflates():
    raw = [0.001, 0.01, 0.04]
    corrected = holm_correct(raw)
    # Corrected p-values must be non-decreasing
    assert corrected[0] <= corrected[1] <= corrected[2]
    # Each corrected value must be >= its raw counterpart
    assert all(c >= r for c, r in zip(corrected, raw))

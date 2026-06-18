"""Tests for evaluation/human_eval/kappa.py - Cohen's kappa and Spearman rho."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.human_eval.kappa import cohen_kappa, spearman


def test_perfect_agreement_kappa_is_one():
    ratings = [0, 1, 2, 1, 0, 2, 1]
    assert cohen_kappa(ratings, ratings, categories=[0, 1, 2]) == 1.0


def test_kappa_zero_chance_agreement():
    # Systematic disagreement: rater1 says 0 where rater2 says 1 and vice versa.
    # Both raters use each category equally (5 zeros, 5 ones) so p_e = 0.5,
    # but observed agreement = 0, giving kappa = (0 - 0.5) / (1 - 0.5) = -1.0.
    a = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    b = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    kappa = cohen_kappa(a, b, categories=[0, 1])
    assert kappa < 0.0  # worse than chance


def test_spearman_perfect_monotone():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert abs(spearman(x, y) - 1.0) < 1e-9


def test_spearman_inverse_monotone():
    x = [1.0, 2.0, 3.0]
    y = [3.0, 2.0, 1.0]
    assert abs(spearman(x, y) - (-1.0)) < 1e-9

"""Resume must re-run entries that errored; only fully-successful entries count as done."""
from scripts.run_clinical_benchmark import _completed_ids as clinical_completed
from scripts.run_medqa_benchmark import _completed_ids as medqa_completed


def _results():
    return [
        {"id": 0, "adapt_ai": {"error": None}, "baseline": {"error": None}},
        {"id": 1, "adapt_ai": {"error": "usage limit"}, "baseline": {"error": "usage limit"}},
        {"id": 2, "adapt_ai": {"error": None}, "baseline": {"error": "boom"}},
    ]


def test_clinical_completed_ids_excludes_errored():
    assert clinical_completed(_results()) == {0}


def test_medqa_completed_ids_excludes_errored():
    assert medqa_completed(_results()) == {0}

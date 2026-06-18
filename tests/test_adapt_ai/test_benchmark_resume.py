"""Resume must re-run entries that errored; only fully-successful entries count as done."""
from scripts.run_benchmark import _completed_ids as reasoning_completed


def _results():
    return [
        {"id": 0, "adapt_ai": {"error": None}, "baseline": {"error": None}},
        {"id": 1, "adapt_ai": {"error": "usage limit"}, "baseline": {"error": "usage limit"}},
        {"id": 2, "adapt_ai": {"error": None}, "baseline": {"error": "boom"}},
    ]


def test_reasoning_completed_ids_excludes_errored():
    assert reasoning_completed(_results()) == {0}

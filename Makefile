.PHONY: test seed matrix analyze reproduce

test:
	pytest tests/test_adapt_ai/ -q

seed:
	python scripts/seed_vector_db.py --domain healthcare
	python scripts/seed_vector_db.py --domain legal
	python scripts/seed_vector_db.py --domain finance

matrix:
	python scripts/run_matrix.py haiku sonnet qwen7b

analyze:
	python scripts/analyze_results.py --domain healthcare
	python scripts/analyze_results.py --domain legal
	python scripts/analyze_results.py --domain finance

reproduce: seed matrix analyze

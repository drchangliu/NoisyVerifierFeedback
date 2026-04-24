.PHONY: test test-integration lint format install

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -m "not integration" -v

test-integration:
	pytest tests/ -v

lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

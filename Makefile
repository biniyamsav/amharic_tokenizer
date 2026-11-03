.PHONY: install test lint format

install:
	python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

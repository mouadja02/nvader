VENV = .venv

ifeq ($(OS),Windows_NT)
    PYTHON = $(VENV)/Scripts/python
    BIN = $(VENV)/Scripts
else
    PYTHON = $(VENV)/bin/python
    BIN = $(VENV)/bin
endif

.PHONY: install test lint format check info roadmap clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

check: lint test

info:
	$(BIN)/nvader info

roadmap:
	$(BIN)/nvader roadmap

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +

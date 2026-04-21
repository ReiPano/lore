PY ?= python3.12
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help venv install init up down restart logs test lint clean reset

help:
	@echo "Targets: venv install init up down restart logs test lint clean reset"

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: venv
	$(BIN)/pip install -e ".[dev]"
	$(MAKE) init

init:
	$(BIN)/hybrid-search init

up:
	$(BIN)/hybrid-search up --watch

down:
	$(BIN)/hybrid-search down

restart:
	$(BIN)/hybrid-search restart

logs:
	docker compose logs -f qdrant

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check src tests eval

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__ build dist *.egg-info

reset:
	$(BIN)/lore down || true
	rm -rf $$HOME/.lore $$HOME/.better-mem

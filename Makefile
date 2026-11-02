.PHONY: help install-hooks type-check type-check-watch ruff qa testall test pdb coverage docs-serve docs-build build bump-patch bump-minor bump-major release clean clean-build clean-pyc clean-test


MIN_POLARS := $(shell sed -n 's/^ *"polars>=\([^"]*\)",/\1/p' pyproject.toml)

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-20s %s\n", $$1, $$2}'

install-hooks:  ## Configure git to use .githooks/
	git config core.hooksPath .githooks

type-check:  ## Type check with pyright
	uv run pyright

type-check-watch:  ## Type check in watch mode
	uv run pyright --watch

ruff:  ## Run ruff checks
	uv run ruff format --check --diff .
	uv run ruff check .

qa:  ## Format, lint, type check, and test
	uv run ruff format .
	uv run ruff check . --fix
	uv run ruff check --select I --fix .
	uv run pyright
	$(MAKE) testall

testall:  ## Run tests for all supported Python versions and the oldest supported Polars
	@echo "==> Testing on Python 3.12"
	uv run --python=3.12 pytest
	@echo "==> Testing on Python 3.13"
	uv run --python=3.13 pytest
	@echo "==> Testing on the oldest supported Polars ($(MIN_POLARS))"
	uv run --with "polars==$(MIN_POLARS)" pytest

test:  ## Run tests (pass ARGS="..." for extra arguments)
	uv run pytest $(ARGS)

pdb:  ## Run tests dropping into debugger on failure
	uv run pytest --pdb --maxfail=10 $(ARGS)

coverage:  ## Run tests with HTML coverage report
	uv run pytest --cov-report=html

docs-serve:  ## Serve docs locally with live reload
	uv run sphinx-autobuild docs/source/ docs/_build/html

docs-build:  ## Build docs
	$(MAKE) -C docs html

build:  ## Build the project
	rm -rf build dist
	uv build

bump-patch:  ## Bump patch version (0.0.x), commit, and create CHANGELOG stub
	uv run scripts/bump.py patch

bump-minor:  ## Bump minor version (0.x.0), commit, and create CHANGELOG stub
	uv run scripts/bump.py minor

bump-major:  ## Bump major version (x.0.0), commit, and create CHANGELOG stub
	uv run scripts/bump.py major

release:  ## Tag and push, then create a GitHub release
	uv run scripts/release.py

clean: clean-build clean-pyc clean-test clean-docs  ## Remove all build, test, and Python artifacts

clean-build:  ## Remove build artifacts
	rm -fr build/ dist/ .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:  ## Remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:  ## Remove test and coverage artifacts
	find . -name '.coverage' -exec rm -f {} +
	find . -name '.coverage.*' -exec rm -f {} +
	rm -fr htmlcov/ .pytest_cache
	find . -name '.benchmarks' -exec rm -fr {} +

clean-docs: ## Remove old docs builds
	rm -rf docs/_build
	rm -rf docs/source/api/_api

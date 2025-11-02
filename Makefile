.PHONY: help install install-dev test test-all lint format type-check qa clean setup-lucene

# Default target
help:
	@echo "Available commands:"
	@echo "  make install              - Install the package with all extras (Poetry)"
	@echo "  make install-dev          - Install with development dependencies (Poetry)"
	@echo "  make install-editable     - Install in editable mode with pip"
	@echo "  make install-editable-dev - Install in editable mode with dev tools"
	@echo "  make setup-lucene         - Download Lucene JARs for Solr normalization tests"
	@echo "  make test                 - Run tests (skipping JVM-dependent tests)"
	@echo "  make test-all             - Run all tests including JVM-dependent tests"
	@echo "  make test-ocrqa           - Run only OCRQA tests"
	@echo "  make test-langident       - Run only language identification tests"
	@echo "  make test-cov             - Run tests with coverage report"
	@echo "  make lint                 - Run linting checks"
	@echo "  make format               - Format code with black"
	@echo "  make type-check           - Run type checking with mypy"
	@echo "  make qa                   - Run all QA checks (test, lint, type-check)"
	@echo "  make clean                - Remove build artifacts and cache files"

# Installation targets
install:
	poetry install --all-extras

install-dev:
	poetry install --all-extras --with dev

install-editable:
	pip install -e ".[all]"

install-editable-dev:
	pip install -e ".[all]"
	pip install pytest pylint flake8 black mypy

# Setup Lucene JARs for Solr normalization tests
setup-lucene:
	@echo "Downloading Lucene JARs..."
	mkdir -p lucene_jars
	curl -L -o lucene_jars/lucene-core-9.3.0.jar https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/9.3.0/lucene-core-9.3.0.jar
	curl -L -o lucene_jars/lucene-analyzers-common-9.3.0.jar https://repo1.maven.org/maven2/org/apache/lucene/lucene-analyzers-common/9.3.0/lucene-analyzers-common-9.3.0.jar
	curl -L -o lucene_jars/lucene-analysis-custom-9.3.0.jar https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-custom/9.3.0/lucene-analysis-custom-9.3.0.jar
	@echo "Setting CLASSPATH..."
	@echo "export CLASSPATH=$$PWD/lucene_jars/*" > lucene_env.sh
	@echo "Lucene setup complete. Run 'source lucene_env.sh' to set CLASSPATH"

# Test targets
test:
	IMPRESSO_SKIP_JVM=1 poetry run pytest

test-all:
	poetry run pytest

test-ocrqa:
	IMPRESSO_SKIP_JVM=1 poetry run pytest tests/ocrqa/

test-langident:
	IMPRESSO_SKIP_JVM=1 poetry run pytest tests/langident/

test-ldatopics:
	IMPRESSO_SKIP_JVM=1 poetry run pytest tests/ldatopics/

test-newsagencies:
	IMPRESSO_SKIP_JVM=1 poetry run pytest tests/newsagencies/

test-solrnormalization:
	poetry run pytest tests/solrnormalization/

test-cov:
	IMPRESSO_SKIP_JVM=1 poetry run pytest --cov=impresso_pipelines --cov-report=html --cov-report=term

test-verbose:
	IMPRESSO_SKIP_JVM=1 poetry run pytest -v

# Code quality targets
lint:
	@echo "Running linting checks..."
	poetry run pylint impresso_pipelines/ || true
	poetry run flake8 impresso_pipelines/ || true

format:
	@echo "Formatting code with black..."
	poetry run black impresso_pipelines/ tests/

format-check:
	@echo "Checking code formatting..."
	poetry run black --check impresso_pipelines/ tests/

type-check:
	@echo "Running type checks with mypy..."
	poetry run mypy impresso_pipelines/ || true

# Combined QA target (mimics CI)
qa: test lint type-check
	@echo "All QA checks complete!"

# Clean targets
clean:
	@echo "Cleaning build artifacts and cache files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

clean-lucene:
	@echo "Removing Lucene JARs..."
	rm -rf lucene_jars/
	rm -f lucene_env.sh
	@echo "Lucene files removed!"

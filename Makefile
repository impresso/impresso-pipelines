.PHONY: help install install-dev lock test test-all lint format type-check qa clean setup-lucene

# Detect which tool to use (uv preferred, fallback to poetry)
UV_AVAILABLE := $(shell command -v uv 2> /dev/null)
ifdef UV_AVAILABLE
    PYTHON_RUN = uv run
    INSTALL_CMD = uv sync --extra all --extra dev
    INSTALL_PROD_CMD = uv sync --extra all
    LOCK_CMD = uv lock
    TOOL_NAME = uv
else
    PYTHON_RUN = $(PYTHON_RUN)
    INSTALL_CMD = poetry install --all-extras --with dev
    INSTALL_PROD_CMD = poetry install --all-extras
    LOCK_CMD = poetry lock
    TOOL_NAME = poetry
endif

# Default target
help:
	@echo "Using tool: $(TOOL_NAME)"
	@echo ""
	@echo "Available commands:"
	@echo "  make lock                 - Update lock file ($(LOCK_CMD))"
	@echo "  make install              - Install the package with all extras"
	@echo "  make install-dev          - Install with development dependencies"
	@echo "  make install-editable     - Install in editable mode with pip"
	@echo "  make install-editable-dev - Install in editable mode with dev tools"
	@echo "  make setup-lucene         - Download Lucene JARs for Solr normalization tests"
	@echo "  make test                 - Run all tests (JVM tests run separately to avoid conflicts)"
	@echo "  make test-all             - Run all tests (same as test)"
	@echo "  make test-all-together    - Run all tests in one pytest session (may have JVM conflicts)"
	@echo "  make test-ocrqa           - Run only OCRQA tests"
	@echo "  make test-langident       - Run only language identification tests"
	@echo "  make test-ldatopics       - Run only LDA topics tests (requires Java)"
	@echo "  make test-newsagencies    - Run only news agencies tests"
	@echo "  make test-solrnormalization - Run only Solr normalization tests (requires Java)"
	@echo "  make test-cov             - Run tests with coverage report"
	@echo "  make test-log             - Run all tests with INFO logging visible"
	@echo "  make test-debug           - Run all tests with DEBUG logging visible"
	@echo "  make lint                 - Run linting checks"
	@echo "  make format               - Format code with black"
	@echo "  make type-check           - Run type checking with mypy"
	@echo "  make qa                   - Run all QA checks (test, lint, type-check)"
	@echo "  make clean                - Remove build artifacts and cache files"

# Lock file target
lock:
	$(LOCK_CMD)

# Installation targets
install:
	$(INSTALL_PROD_CMD)

install-dev:
	$(INSTALL_CMD)

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
	@echo "Note: JVM-based tests (ldatopics, solrnormalization) may conflict when run together."
	@echo "Running non-JVM tests first, then JVM tests separately..."
	$(PYTHON_RUN) pytest tests/ocrqa/ tests/langident/ tests/newsagencies/
	$(PYTHON_RUN) pytest tests/solrnormalization/
	$(PYTHON_RUN) pytest tests/ldatopics/

test-all:
	@echo "Note: JVM-based tests (ldatopics, solrnormalization) may conflict when run together."
	@echo "Running non-JVM tests first, then JVM tests separately..."
	$(PYTHON_RUN) pytest tests/ocrqa/ tests/langident/ tests/newsagencies/
	$(PYTHON_RUN) pytest tests/solrnormalization/
	$(PYTHON_RUN) pytest tests/ldatopics/

test-all-together:
	@echo "WARNING: Running all tests together may cause JVM conflicts!"
	$(PYTHON_RUN) pytest

test-ocrqa:
	$(PYTHON_RUN) pytest tests/ocrqa/

test-langident:
	$(PYTHON_RUN) pytest tests/langident/

test-ldatopics:
	$(PYTHON_RUN) pytest tests/ldatopics/

test-newsagencies:
	$(PYTHON_RUN) pytest tests/newsagencies/

test-solrnormalization:
	$(PYTHON_RUN) pytest tests/solrnormalization/

test-cov:
	$(PYTHON_RUN) pytest --cov=impresso_pipelines --cov-report=html --cov-report=term

test-log:
	@echo "Running tests with INFO logging (JVM tests separately)..."
	$(PYTHON_RUN) pytest tests/ocrqa/ tests/langident/ tests/newsagencies/ --log-cli-level=INFO
	$(PYTHON_RUN) pytest tests/solrnormalization/ --log-cli-level=INFO
	$(PYTHON_RUN) pytest tests/ldatopics/ --log-cli-level=INFO

test-debug:
	@echo "Running tests with DEBUG logging (JVM tests separately)..."
	$(PYTHON_RUN) pytest tests/ocrqa/ tests/langident/ tests/newsagencies/ --log-cli-level=DEBUG -s
	$(PYTHON_RUN) pytest tests/solrnormalization/ --log-cli-level=DEBUG -s
	$(PYTHON_RUN) pytest tests/ldatopics/ --log-cli-level=DEBUG -s

test-verbose:
	$(PYTHON_RUN) pytest -v
	$(PYTHON_RUN) pytest -v

# Code quality targets
lint:
	@echo "Running linting checks..."
	$(PYTHON_RUN) pylint impresso_pipelines/ || true
	$(PYTHON_RUN) flake8 impresso_pipelines/ || true

format:
	@echo "Formatting code with black..."
	$(PYTHON_RUN) black impresso_pipelines/ tests/

format-check:
	@echo "Checking code formatting..."
	$(PYTHON_RUN) black --check impresso_pipelines/ tests/

type-check:
	@echo "Running type checks with mypy..."
	$(PYTHON_RUN) mypy impresso_pipelines/ || true

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

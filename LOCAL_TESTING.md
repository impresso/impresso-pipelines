# Local Testing Quick Reference

This guide provides quick commands for local testing during development.

## Prerequisites

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone repository
git clone https://github.com/impresso/impresso-pipelines.git
cd impresso-pipelines

# Install dependencies - Choose one option:

# Option 1: Poetry (recommended for full development)
make install-dev
# or: poetry install --all-extras --with dev

# Option 2: Pip editable mode (faster, best for testing local changes)
make install-editable-dev
# or: pip install -e ".[all]"
```

**When to use which option:**

- **Poetry**: Use for full development, managing dependencies, building releases
- **Pip editable**: Use when you're making frequent code changes and want tests to immediately reflect your changes

## Quick Test Commands

### Using Makefile (Recommended)

```bash
make help              # Show all available commands
make test              # Run tests (skip JVM tests) - FASTEST
make test-cov          # Run tests with coverage report
make test-ocrqa        # Run only OCRQA tests
make test-all          # Run all tests including JVM tests
make qa                # Run full QA (tests + lint + type check)
make clean             # Clean build artifacts
```

### Using Shell Script

```bash
./run_tests.sh              # Quick tests (skip JVM)
./run_tests.sh full         # All tests including JVM
./run_tests.sh ocrqa        # Only OCRQA tests
./run_tests.sh coverage     # With coverage report
./run_tests.sh qa           # Full QA suite
./run_tests.sh help         # Show help
```

### Using Poetry Directly

```bash
# Quick tests (skip JVM)
IMPRESSO_SKIP_JVM=1 poetry run pytest

# All tests
poetry run pytest

# Specific test file
poetry run pytest tests/ocrqa/test_ocrqa_pipeline.py

# Specific test function
poetry run pytest tests/ocrqa/test_ocrqa_pipeline.py::test_ocrqa_pipeline_basic

# Verbose output
poetry run pytest -v

# Stop on first failure
poetry run pytest -x

# Coverage
poetry run pytest --cov=impresso_pipelines --cov-report=html
```

## Running Individual Pipeline Tests

```bash
# OCRQA
make test-ocrqa
# or: IMPRESSO_SKIP_JVM=1 poetry run pytest tests/ocrqa/

# Language Identification
make test-langident
# or: IMPRESSO_SKIP_JVM=1 poetry run pytest tests/langident/

# LDA Topics
make test-ldatopics
# or: IMPRESSO_SKIP_JVM=1 poetry run pytest tests/ldatopics/

# News Agencies
make test-newsagencies
# or: IMPRESSO_SKIP_JVM=1 poetry run pytest tests/newsagencies/

# Solr Normalization (requires Lucene setup)
make setup-lucene && source lucene_env.sh
make test-solrnormalization
# or: poetry run pytest tests/solrnormalization/
```

## Code Quality Checks

```bash
# Linting
make lint
# or: poetry run pylint impresso_pipelines/

# Code formatting
make format              # Format code
make format-check        # Check without formatting

# Type checking
make type-check
# or: poetry run mypy impresso_pipelines/

# Run all QA checks
make qa
```

## Testing Specific Features

### Test Module-Level Functions

```bash
# Test OCRQA subtokens function
poetry run python -c "
from impresso_pipelines.ocrqa.ocrqa_pipeline import subtokens
tokens = subtokens('hello~world', version='2.0.0')
print(tokens)
"
```

### Test Pipeline Class

```bash
# Create a test script
cat > test_pipeline.py << 'EOF'
from impresso_pipelines.ocrqa import OCRQAPipeline

pipeline = OCRQAPipeline()
result = pipeline("Hello world, this is a test.", language="en")
print(f"Language: {result['language']}, Score: {result['score']}")
EOF

# Run it
poetry run python test_pipeline.py
```

## Environment Variables

```bash
# Skip JVM-dependent tests (default for quick testing)
IMPRESSO_SKIP_JVM=1 poetry run pytest

# Set Lucene classpath for Solr tests
export CLASSPATH=$PWD/lucene_jars/*
poetry run pytest tests/solrnormalization/

# Verbose pytest output
PYTEST_ADDOPTS="-v" poetry run pytest
```

## Common Workflows

### Before Committing

```bash
# 1. Run tests
make test

# 2. Check code quality
make qa

# 3. Format code
make format
```

### Before Creating PR

```bash
# Run full QA suite (mimics CI)
make qa

# Or run comprehensive checks
make test-cov          # Check test coverage
make format-check      # Verify formatting
make lint              # Check code quality
make type-check        # Verify type hints
```

### Debugging Test Failures

```bash
# Run with verbose output
poetry run pytest -v

# Run with print statements visible
poetry run pytest -s

# Run specific test with debugging
poetry run pytest tests/ocrqa/test_ocrqa_pipeline.py::test_ocrqa_pipeline_basic -v -s

# Stop on first failure
poetry run pytest -x

# Show local variables on failure
poetry run pytest --showlocals
```

## Troubleshooting


### Issue: "Poetry not found"

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: "ImportError: cannot import name 'subtokens'"

This means the test is importing from an old installed package instead of your local code.

**Quick fix:**

```bash
# Uninstall old package and reinstall from local source
pip uninstall impresso-pipelines -y
poetry install --all-extras

# Or use pip in editable mode
pip install -e ".[all]"
```

**Why this happens:**

- You have an older version installed system-wide (e.g., via `pip install impresso-pipelines`)
- Python imports from site-packages instead of your local development code
- Solution: Reinstall in editable/development mode

### Issue: "Poetry broken after Python upgrade"

```bash
# Reinstall poetry
curl -sSL https://install.python-poetry.org | python3 - --uninstall
curl -sSL https://install.python-poetry.org | python3 -
```

### Issue: "Module not found"

### Issue: "Module not found"

```bash
# Activate virtual environment
poetry shell

# Or reinstall
poetry install --all-extras --with dev
```

### Issue: "JVM tests failing"

```bash
# Skip JVM tests
IMPRESSO_SKIP_JVM=1 poetry run pytest

# Or set up Lucene
make setup-lucene
source lucene_env.sh
```

### Issue: "Coverage report not generated"

```bash
# Run with coverage
poetry run pytest --cov=impresso_pipelines --cov-report=html

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## CI/CD Comparison

| Action      | Local           | GitHub CI                    |
| ----------- | --------------- | ---------------------------- |
| Quick tests | `make test`     | `IMPRESSO_SKIP_JVM=1 pytest` |
| Full tests  | `make test-all` | `pytest` (with Lucene)       |
| QA checks   | `make qa`       | Automatic on PR              |
| Publish     | Manual          | Automatic on release         |

## Resources

- Full documentation: [CONTRIBUTING.md](CONTRIBUTING.md)
- Pipeline READMEs: `README_*.md`
- Example notebooks: [impresso-datalab-notebooks](https://github.com/impresso/impresso-datalab-notebooks)

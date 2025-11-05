# Contributing to impresso-pipelines

Thank you for your interest in contributing to impresso-pipelines! This document provides guidelines and instructions for local development.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests Locally](#running-tests-locally)
- [Code Quality Checks](#code-quality-checks)
- [Project Structure](#project-structure)
- [Working with Pipelines](#working-with-pipelines)
- [CI/CD Workflow](#cicd-workflow)
- [Troubleshooting](#troubleshooting)

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [Poetry](https://python-poetry.org/) 1.5 or higher
- Java Development Kit (JDK) 17+ (only for Solr normalization tests)
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/impresso/impresso-pipelines.git
   cd impresso-pipelines
   ```

2. **Install Poetry** (if not already installed):

   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies:**

   **Option 1: Using Poetry (recommended for full development)**

   ```bash
   make install-dev
   # or: poetry install --all-extras --with dev
   ```

   **Option 2: Using pip in editable mode (faster, good for quick iteration)**

   ```bash
   make install-editable-dev
   # or: pip install -e ".[all]"
   ```

4. **Activate the virtual environment (if using Poetry):**

   ```bash
   poetry shell
   ```

## Running Tests Locally

The project uses `pytest` for testing. Tests can be run with or without JVM-dependent components.

### Quick Start

```bash
# Show all available commands
make help

# Run all tests (skipping JVM-dependent tests)
make test

# Run tests with coverage report
make test-cov

# Run specific pipeline tests
make test-ocrqa
make test-langident
make test-ldatopics
make test-newsagencies
```

### Running Tests Without Make

```bash
# Quick tests (automatically skips JVM-dependent: ldatopics, solrnormalization)
poetry run pytest -m "not jvm"  # Or use: make test

# Run all tests including JVM-dependent ones (requires Java)
poetry run pytest  # Or use: make test-all

# Run specific test file
poetry run pytest tests/ocrqa/test_ocrqa_pipeline.py

# Run with verbose output
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=impresso_pipelines --cov-report=html
```

### Setting up Solr Normalization Tests

The Solr normalization pipeline requires Lucene JARs. To set this up locally:

```bash
# Download Lucene JARs and set up environment
make setup-lucene

# Source the environment file
source lucene_env.sh

# Now run Solr normalization tests
make test-solrnormalization

# Or run all tests including JVM-dependent ones
make test-all
```

**Manual setup:**

```bash
mkdir -p lucene_jars
curl -L -o lucene_jars/lucene-core-9.3.0.jar \
  https://repo1.maven.org/maven2/org/apache/lucene/lucene-core/9.3.0/lucene-core-9.3.0.jar
curl -L -o lucene_jars/lucene-analyzers-common-9.3.0.jar \
  https://repo1.maven.org/maven2/org/apache/lucene/lucene-analyzers-common/9.3.0/lucene-analyzers-common-9.3.0.jar
curl -L -o lucene_jars/lucene-analysis-custom-9.3.0.jar \
  https://repo1.maven.org/maven2/org/apache/lucene/lucene-analysis-custom/9.3.0/lucene-analysis-custom-9.3.0.jar

export CLASSPATH=$PWD/lucene_jars/*
```

## Code Quality Checks

### Linting

```bash
# Run linting checks
make lint

# Or manually
poetry run pylint impresso_pipelines/
poetry run flake8 impresso_pipelines/
```

### Code Formatting

```bash
# Format code with black
make format

# Check formatting without changes
make format-check
```

### Type Checking

```bash
# Run mypy type checking
make type-check

# Or manually
poetry run mypy impresso_pipelines/
```

### Run All QA Checks

```bash
# Run tests, linting, and type checking (mimics CI)
make qa
```

## Project Structure

```
impresso-pipelines/
├── impresso_pipelines/          # Main package
│   ├── langident/               # Language identification pipeline
│   ├── ocrqa/                   # OCR quality assessment pipeline
│   ├── ldatopics/               # LDA topic modeling pipeline
│   ├── newsagencies/            # News agencies extraction pipeline
│   └── solrnormalization/       # Solr text normalization pipeline
├── tests/                       # Test files (mirrors package structure)
│   ├── langident/
│   ├── ocrqa/
│   ├── ldatopics/
│   ├── newsagencies/
│   └── solrnormalization/
├── .github/workflows/           # CI/CD workflows
├── pyproject.toml               # Poetry configuration
├── Makefile                     # Local development commands
└── README*.md                   # Documentation files
```

## Working with Pipelines

### Adding a New Pipeline

1. Create a new directory under `impresso_pipelines/`
2. Implement the pipeline class with a `__call__` method
3. Add corresponding tests in `tests/`
4. Update `pyproject.toml` with optional dependencies
5. Create a `README_<pipeline>.md` file
6. Update main `README.md`

### Testing Your Pipeline

```bash
# Create a simple test script
cat > test_my_pipeline.py << 'EOF'
from impresso_pipelines.ocrqa import OCRQAPipeline

pipeline = OCRQAPipeline()
result = pipeline("Hello world, this is a test.")
print(result)
EOF

# Run it
poetry run python test_my_pipeline.py
```

### Module-Level Functions

Some pipelines expose module-level functions for direct use:

```python
# OCRQA pipeline
from impresso_pipelines.ocrqa.ocrqa_pipeline import subtokens, normalize_text

# Use functions directly without pipeline instance
tokens = subtokens("hello~world", version="2.0.0")
print(tokens)  # ['hello', '~', 'world']
```

## CI/CD Workflow

### GitHub Actions

The project uses GitHub Actions for continuous integration:

- **QA Workflow** (`.github/workflows/qa.yml`):

  - Runs on pull requests
  - Python 3.12 with Poetry 1.5
  - Sets up Java 17 and Lucene JARs
  - Runs all tests including JVM-dependent ones
  - Command: `IMPRESSO_SKIP_JVM=1 poetry run pytest`

- **Publish Workflows**:
  - `python-publish.yml`: Publishes to PyPI on releases
  - `python-publish-test.yml`: Publishes to Test PyPI on pre-releases

### Replicating CI Locally

To replicate the CI environment locally:

```bash
# 1. Install all dependencies
make install-dev

# 2. Set up Lucene (for complete CI simulation)
make setup-lucene
source lucene_env.sh

# 3. Run QA checks
make qa

# 4. Run all tests (including JVM)
make test-all
```

### Environment Variables

- `JAVA_HOME`: Path to Java installation (required for LDA Topics and Solr Normalization pipelines)
- `CLASSPATH`: Path to Lucene JAR files (for Solr normalization, if using external JARs)

## Troubleshooting

### Common Issues

**1. Poetry not found:**

```bash
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
```

**2. Python version mismatch:**

```bash
# Use pyenv to install correct Python version
pyenv install 3.11
pyenv local 3.11
```

**3. ImportError: cannot import name from installed package:**
This happens when tests import from an older installed version instead of your local changes.

```bash
# Solution 1: Reinstall in editable mode
poetry install --all-extras

# Solution 2: Uninstall the system package first
pip uninstall impresso-pipelines -y
poetry install --all-extras

# Solution 3: Use pip in editable mode
pip install -e ".[all]"
```

**4. Poetry broken after Python upgrade:**

```bash
# Reinstall poetry
curl -sSL https://install.python-poetry.org | python3 - --uninstall
curl -sSL https://install.python-poetry.org | python3 -

# Or use pipx (recommended)
pipx install poetry
```

**5. JVM tests failing:**

**4. HuggingFace Hub authentication:**
Some pipelines download models from HuggingFace. If you encounter authentication issues:

```bash
# Login to HuggingFace CLI
huggingface-cli login
```

**5. Module import errors:**

```bash
# Make sure you're in the poetry shell
poetry shell

# Or run with poetry prefix
poetry run python your_script.py
```

### Getting Help

- Check the [README files](README.md) for pipeline-specific documentation
- Look at example notebooks in [impresso-datalab-notebooks](https://github.com/impresso/impresso-datalab-notebooks/tree/main/annotate)
- Open an issue on GitHub for bugs or feature requests

## Development Workflow

1. **Create a feature branch:**

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes and test:**

   ```bash
   # Run tests
   make test

   # Check code quality
   make qa
   ```

3. **Commit your changes:**

   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

4. **Push and create a pull request:**
   ```bash
   git push origin feature/my-new-feature
   ```

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public APIs
- Keep functions focused and modular
- Add tests for new functionality

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG (if exists)
3. Create a git tag: `git tag v0.x.x`
4. Push tag: `git push --tags`
5. GitHub Actions will automatically publish to PyPI

---

**Thank you for contributing to impresso-pipelines!** 🎉

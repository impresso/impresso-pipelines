# AGENTS.md

## Repository Overview

**impresso-pipelines** is a Python package (v0.6.x) providing modular NLP/text processing pipelines for the [impresso](https://impresso-project.ch/) digital humanities project. It enables reproducible processing of historical newspaper content.

- **PyPI**: `impresso-pipelines`
- **Python**: 3.11–3.12
- **Package manager**: `uv` (preferred) or `poetry`

---

## Repository Structure

```
impresso_pipelines/       # Main package
  adclassifier/           # Advertisement classifier pipeline
  langident/              # Language identification pipeline
  ldatopics/              # LDA topic modeling pipeline
  newsagencies/           # News agency extraction pipeline
  ocrqa/                  # OCR quality assessment pipeline
  solrnormalization/      # Lucene/Solr text normalization pipeline
tests/                    # Pytest test suite mirroring package structure
lucene_jars/              # Lucene JARs for Solr normalization (downloaded via make)
```

---

## Pipelines

| Pipeline                | Module              | Description                                                          |
| ----------------------- | ------------------- | -------------------------------------------------------------------- |
| Language Identification | `langident`         | Detects language of input text with confidence score                 |
| OCR QA                  | `ocrqa`             | Estimates OCR quality (0–1) using Bloom filters                      |
| LDA Topics              | `ldatopics`         | Soft clustering via LDA-based topic modeling (requires Java)         |
| News Agencies           | `newsagencies`      | Extracts and ranks news agency entities; links to Wikidata           |
| Ad Classifier           | `adclassifier`      | Detects advertisements using fine-tuned XLM-RoBERTa                  |
| Solr Normalization      | `solrnormalization` | Replicates Solr language-specific text normalization (requires Java) |

---

## Development Setup

```bash
# Clone
git clone https://github.com/impresso/impresso-pipelines.git
cd impresso-pipelines

# Install all extras + dev deps (uv preferred)
make install-dev
# or: uv sync --extra all --extra dev
# or: poetry install --all-extras --with dev

# Download Lucene JARs (required for ldatopics and solrnormalization)
make setup-lucene
```

---

## Running Tests

```bash
make test           # All tests (JVM tests run in separate sessions to avoid conflicts)
make test-ocrqa
make test-langident
make test-ldatopics         # Requires Java
make test-newsagencies
make test-solrnormalization # Requires Java
make test-cov       # With coverage report
```

Individual modules can be installed independently:

```bash
pip install "impresso-pipelines[langident]"
pip install "impresso-pipelines[ocrqa]"
pip install "impresso-pipelines[ldatopics]"
pip install "impresso-pipelines[newsagencies]"
pip install "impresso-pipelines[adclassifier]"
pip install "impresso-pipelines[solrnormalization]"
pip install "impresso-pipelines[all]"
```

---

## Code Quality

```bash
make lint        # flake8
make format      # black
make type-check  # mypy
make qa          # lint + type-check + tests
```

---

## Key Notes for Agents

- **Java (JDK 17+)** is required for `ldatopics` and `solrnormalization` pipelines.
- Models are fetched from **Hugging Face Hub** at runtime; tests may require internet access unless running in offline mode (see `tests/test_offline_mode.py`).
- JVM conflicts can occur when running all tests together; use `make test` (not `make test-all-together`) to run JVM-dependent tests in separate sessions.
- The package uses `pyproject.toml` with both `[project]` (PEP 517) and `[tool.poetry]` sections for dual `uv`/`poetry` support.

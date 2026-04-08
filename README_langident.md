### Language Identification Example

Make sure the package is installed as shown in the main [README](README.md).

> **Note:** For more documentation and usage details, see the inline docstrings and comments in the code.

```python
from impresso_pipelines.langident import LangIdentPipeline
# Initialize the pipeline
lang_pipeline = LangIdentPipeline()

# Example text in German
text = """Vieles Seltsame geschieht auf Erden :
Nichts Seltsameres sieht der Mond
Als das Glück, das im Knopfloch wohnt.
Zaubrisch faßt es den ernsten Mann.
Ohne nach Weib u. Kind zu fragen
Reitet er aus, nach dem Glück zu jagen,
Nur nacb ihm war stets sein Vegehr.
Aber neben ihm reitet der Dämon her
Des Ehrgeizes mit finsterer Tücke,
Und so jagt er zuletzt auf die Brücke,
Die über dem Abgrund, d:m nächtlich schwarzen
Jählings abbricht."""


# Detect language
result = lang_pipeline(text)
print(result)
```

**Expected Output:**

```
{'language': 'de', 'score': 1.0}
```

The score represents the model’s confidence (as a probability) in the detected language.

## Configuration

### Initialization Parameters

```python
LangIdentPipeline(
    model_id: Optional[str] = None,       # Specific model file (auto-detects latest if None)
    repo_id: str = "impresso-project/impresso-floret-langident",
    revision: str = "main",               # Repository revision/branch/tag
    local_files_only: bool = False        # Use only cached files (offline mode)
)
```

**Examples:**

```python
# Use default settings (auto-detect latest model)
pipeline = LangIdentPipeline()

# Use specific model version
pipeline = LangIdentPipeline(model_id="langident-v1.2.3.bin")

# Offline mode - use only cached files (no network access)
pipeline = LangIdentPipeline(local_files_only=True)
```

### Offline / Cache-First Mode

For production environments, HPC clusters, or situations where you want to avoid network dependencies, you can use `local_files_only=True`:

```python
# Initialize in offline mode
pipeline = LangIdentPipeline(local_files_only=True)

# The pipeline will:
# 1. Scan the local HuggingFace cache for available model files
# 2. Use only cached models
# 3. Not make any network requests to HuggingFace Hub
# 4. Fail cleanly if required files are not cached

result = pipeline("Text to identify")
```

**Benefits:**

- **Reliable parallel execution**: Many workers can initialize simultaneously without Hub rate limits
- **Offline operation**: Works in environments with restricted network access
- **Deterministic behavior**: Uses specific cached versions without checking for updates
- **Faster initialization**: No network latency from repository listing calls

**Requirements:**

- Cache must be pre-populated (run once with `local_files_only=False` to download models)
- For auto-detection of the latest model, the model files must be cached
- Alternatively, specify an explicit `model_id` parameter to use a specific cached model

---

For more details about usage and available features, see the demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/langident_pipeline_demo.ipynb).

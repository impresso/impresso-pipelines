# Implementation Summary: Cache-First / Offline-Friendly Initialization

## Changes Made

### 1. OCRQAPipeline (`impresso_pipelines/ocrqa/ocrqa_pipeline.py`)

**New parameter:** `local_files_only: bool = False`

**Changes:**

- Added `scan_cache_dir` import from `huggingface_hub`
- Added `local_files_only` parameter to `__init__()`
- Implemented `_scan_local_cache()` method to discover cached files without network access
- Modified initialization to use cache scanning when `local_files_only=True`
- Updated `get_bloomfilter()` function to accept and pass through `local_files_only` parameter
- All `hf_hub_download()` calls now respect the `local_files_only` flag
- Passes `local_files_only` to nested `LangIdentPipeline` initialization

**Key behavior:**

- When `local_files_only=True`, scans HF cache instead of calling `list_repo_files()`
- Falls back gracefully if cache scan fails
- Logs detailed information about offline mode and discovered files

### 2. LangIdentPipeline (`impresso_pipelines/langident/langident_pipeline.py`)

**New parameter:** `local_files_only: bool = False`

**Changes:**

- Added `scan_cache_dir` import from `huggingface_hub`
- Added `local_files_only` parameter to `__init__()`
- Implemented `_scan_local_cache()` method to discover cached model files
- Modified initialization to use cache scanning when `local_files_only=True`
- All `hf_hub_download()` calls now respect the `local_files_only` flag

**Key behavior:**

- When `local_files_only=True`, scans HF cache for model files instead of calling `list_repo_files()`
- Still performs automatic "latest version" selection from cached files
- Logs detailed information about offline mode

### 3. Documentation

**Updated READMEs:**

- `README_ocrqa.md`: Added documentation for `local_files_only` parameter and offline mode
- `README_langident.md`: Added configuration section with offline mode documentation

**Test script:**

- Created `test_offline_mode.py` to validate offline initialization works correctly

## API Usage

### OCRQAPipeline

```python
# Offline mode - no network access required
pipeline = OCRQAPipeline(local_files_only=True)
result = pipeline("Text to assess", language="en")
```

### LangIdentPipeline

```python
# Offline mode - no network access required
pipeline = LangIdentPipeline(local_files_only=True)
result = pipeline("Text to identify")
```

## Benefits

1. **Reliable parallel execution**: Many workers can initialize simultaneously without Hub rate limits or timeouts
2. **Offline operation**: Works in environments with restricted network access
3. **Deterministic behavior**: Uses specific cached versions without checking for updates
4. **Faster initialization**: No network latency from repository listing calls
5. **Production-ready**: Makes pipelines suitable for HPC, batch processing, and CI/CD environments

## Backward Compatibility

✅ Fully backward compatible:

- Default behavior unchanged (`local_files_only=False`)
- Existing code continues to work without modification
- Opt-in feature for users who need it

## Testing

All tests passed successfully:

- ✓ LangIdentPipeline offline mode
- ✓ OCRQAPipeline offline mode
- Cache warming works correctly
- Offline initialization succeeds with `HF_HUB_OFFLINE=1` environment variable

## Implementation Notes

The implementation uses `huggingface_hub.scan_cache_dir()` to discover cached files without network access. This allows:

- Discovery of available languages/versions from cache
- Graceful degradation if cache metadata is unavailable
- Automatic selection of latest cached version (same logic as online mode)

The cache scanning is defensive - if it fails, the pipeline logs a warning and continues, allowing explicit file resolution to still work from cache.

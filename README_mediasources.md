# Media Sources Pipeline

This pipeline identifies media-source mentions in historical newspaper text. It currently recognizes press agencies and radio stations, returns exact character offsets, and links known entities to Wikidata.

The pipeline is separate from the legacy `NewsAgenciesPipeline`. It uses a joint named entity recognition model and supports additional media-source classes, long-document chunking, batched inference, and optional date-aware filtering.

## Installation

```bash
pip install "impresso-pipelines[mediasources]"
```

The model is downloaded from Hugging Face on first use and cached locally.

## Quick Start

```python
from impresso_pipelines.mediasources import MediaSourcesPipeline

pipeline = MediaSourcesPipeline()

text = "Reuters reported the decision, while the BBC World Service provided updates."
result = pipeline(text)

print(result["entities"])
print(result["summary"])
```

Example output:

```python
{
    "entities": [
        {
            "surface": "Reuters",
            "label": "org.ent.pressagency.reuters",
            "wkdata_qid": "Q130879",
            "start_year": 1851,
            "start": 0,
            "stop": 7,
            "score": 0.9259085746581319,
        },
        {
            "surface": "BBC World",
            "label": "org.ent.radiostation.bbc",
            "wkdata_qid": "Q9531",
            "start_year": 1922,
            "start": 41,
            "stop": 50,
            "score": 0.7754964914062266,
        },
    ],
    "summary": [
        {
            "uid": "org.ent.pressagency.reuters",
            "wkdata_qid": "Q130879",
            "score": 0.9259085746581319,
        },
        {
            "uid": "org.ent.radiostation.bbc",
            "wkdata_qid": "Q9531",
            "score": 0.7754964914062266,
        },
    ],
}
```

## Input

Pass either one string or a sequence of strings:

```python
# One document returns one dictionary
result = pipeline("Reuters reported from London.")

# Multiple documents return a list in the same order
results = pipeline([
    "Reuters reported from London.",
    "The BBC World Service provided an update.",
    "No media source is mentioned here.",
])
```

For explicit control over per-document dates, use `predict_many`:

```python
results = pipeline.predict_many(
    ["Reuters reported the news.", "The BBC reported the news."],
    publication_dates=["1920-01-01", "1920-01-01"],
    filter_anachronistic=True,
)
```

## Output

The standard result contains:

- `entities`: every detected mention, including its surface text, normalized label, Wikidata QID when available, organization start year, character offsets, and confidence score.
- `summary`: the highest-scoring mention for each normalized label, sorted by score.

Some known labels do not yet have a Wikidata mapping. Their `wkdata_qid` is `None`.

## Confidence Filtering

Use `min_score` to remove low-confidence entities after sequence decoding:

```python
pipeline = MediaSourcesPipeline(min_score=0.80)

# Override the default for one call
result = pipeline(text, min_score=0.90)
```

## Date-Aware Filtering

Each known media source has a start year. When `filter_anachronistic=True` and a publication date is supplied, entities founded after the document year are removed:

```python
result = pipeline(
    "The BBC reported the news.",
    publication_date="1920-05-10",
    filter_anachronistic=True,
)
```

The filter is inactive when no publication date is provided. Dates may be integer years or strings containing a four-digit year.

## Diagnostics

Enable diagnostics to inspect token-level decoding and long-document processing:

```python
result = pipeline(text, diagnostics=True)

print(result["tokens"])
print(result["token_labels"])
print(result["token_scores"])
print(result["inference_diagnostics"])
```

Diagnostic output adds the original text, tokens, character offsets, BIO labels, token scores, and these windowing fields:

- `annotation_tokens`
- `model_chunks`
- `overlapping_annotation_tokens`
- `overlap_replacements`
- `rescued_tokens`

Aggregate timing and batching statistics from the most recent call are available through `pipeline.last_inference_stats`.

## Configuration

```python
pipeline = MediaSourcesPipeline(
    model="impresso-project/mmbert-impresso-mediasources-ner",
    revision="v2.0.0",
    min_score=None,
    batch_size=8,
    dtype="float32",             # float32, float16, or bfloat16
    device=None,                 # auto-detect, or "cpu", "mps", "cuda:0"
    max_sequence_len=None,       # use the model protocol by default
    max_annotation_tokens=None,  # use the model protocol by default
    stride=None,                 # use the model protocol by default
    filter_anachronistic=False,
    local_files_only=False,
    trust_remote_code=True,
)
```

Model protocol settings are read from the model metadata. Override tokenization or decoding settings only when using a compatible custom model.

## Batch Processing

Batching combines model windows from multiple documents, which is more efficient on accelerators:

```python
pipeline = MediaSourcesPipeline(batch_size=16, device="cuda:0")
results = pipeline(list_of_texts)

print(pipeline.last_inference_stats)
```

Long texts are split into smaller overlapping sections for processing. The results are then combined, while keeping entity positions matched to the original text.

## Offline Mode

After the model has been cached, prevent network access with:

```python
pipeline = MediaSourcesPipeline(local_files_only=True)
```

By default, Hugging Face stores downloaded model files under `~/.cache/huggingface/hub/`.

## Notes

- The default model recognizes press agencies and radio stations in multilingual historical newspaper text.
- CPU, Apple Silicon MPS, and NVIDIA CUDA devices are supported through PyTorch.
- `float32` is the safest default. Use `float16` or `bfloat16` only on compatible hardware.
- Empty input produces empty `entities` and `summary` lists.

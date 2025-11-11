# Ad Classification Pipeline

This pipeline identifies advertisements in historical newspaper content using a fine-tuned XLM-RoBERTa model combined with rule-based features and adaptive thresholding.

## Installation

### Using pip with extras (recommended)

```bash
pip install "impresso-pipelines[adclassifier]"
```

This installs the pipeline with all required dependencies (torch, transformers, numpy).

### Manual installation

```bash
pip install torch transformers numpy
```

For Apple Silicon (M1/M2/M3) Macs, PyTorch will automatically use MPS acceleration if available.

## Quick Start

```python
from impresso_pipelines.adclassifier import AdClassifierPipeline

# Initialize pipeline (downloads model on first use)
pipeline = AdClassifierPipeline()

# Classify a single text
result = pipeline("À vendre: Belle villa 5 pièces, CHF 850'000. Tél. 021 123 45 67")
print(result['type'])  # 'ad' or 'non-ad'
```

## Usage

The pipeline accepts multiple input formats:

### 1. Single text string
```python
result = pipeline("Your text here")
```

### 2. List of text strings
```python
results = pipeline([
    "Text 1",
    "Text 2",
    "Text 3"
])
```

### 3. Single dictionary with 'ft' field
```python
result = pipeline({
    "id": "doc1",
    "lg": "fr",  # language (optional)
    "ft": "Your text here"
})
```

### 4. List of dictionaries
```python
results = pipeline([
    {"id": "doc1", "ft": "Text 1", "lg": "fr"},
    {"id": "doc2", "ft": "Text 2", "lg": "de"}
])
```

## Output Format

Each result is a dictionary containing:

```python
{
    "id": "doc1",                          # Document ID (if provided in input)
    "type": "ad",                          # Classification: "ad" or "non-ad"
    "promotion_prob": 0.876543,           # Raw promotion probability
    "promotion_prob_final": 0.891234,     # Final probability after adjustments
    "ensemble_ad_signal": 0.823456,       # Ensemble signal from all genre predictions
    "xgenre_top_label": "Promotion",      # Top predicted genre label
    "xgenre_top_prob": 0.876543,          # Probability of top genre
    "threshold_used": 0.075500,           # Adaptive threshold used
    "rule_score": 5.5,                    # Rule-based feature score
    "rule_confidence": 0.8,               # Confidence in rule-based features
    "model_confidence": 0.753086          # Model prediction confidence
}
```

## Configuration

You can customize the pipeline parameters:

```python
pipeline = AdClassifierPipeline(
    model_name="impresso-project/impresso-ad-classification-xlm-one-class",
    batch_size=16,              # Batch size for GPU processing
    max_length=512,             # Maximum tokens per chunk
    chunk_words=0,              # Words per chunk (0 = no chunking)
    pool="logits_max",          # Pooling strategy: logits_max, logits_mean, logits_weighted
    ad_threshold=0.9991,        # Default threshold for ad classification
    lang_thresholds="other:0.9991,fr:0.0755",  # Language-specific thresholds
    short_len=30,               # Word count considered 'short'
    short_bonus=0.2,            # Threshold reduction for short texts
    temperature=0.8,            # Calibration temperature
    device=None                 # 'cuda', 'mps', 'cpu', or None for auto-detect
)
```

## Device Support

The pipeline automatically detects and uses:
- **CUDA** (NVIDIA GPUs) if available
- **MPS** (Apple Silicon) if available
- **CPU** as fallback

You can manually specify the device:
```python
pipeline = AdClassifierPipeline(device="cuda")  # Force CUDA
pipeline = AdClassifierPipeline(device="mps")   # Force MPS (Apple Silicon)
pipeline = AdClassifierPipeline(device="cpu")   # Force CPU
```

## Model Details

The pipeline uses the fine-tuned model from HuggingFace:
- **Model**: `impresso-project/impresso-ad-classification-xlm-one-class`
- **Base**: XLM-RoBERTa
- **Task**: Multi-genre text classification with ad detection
- **Languages**: Multilingual (optimized for French, German, Luxembourgish)

### Best Parameters

The default parameters are optimized based on cross-validation:
- **Pool**: `logits_max`
- **Temperature**: `0.8`
- **Ad threshold**: `0.9991` (for languages other than French)
- **French threshold**: `0.0755`
- **Macro F1**: 0.9450
- **Balanced accuracy**: 0.9450

## Rule-Based Features

The pipeline combines neural predictions with rule-based features for improved accuracy:

- **Phone numbers** (Swiss and international formats)
- **Prices** (CHF, EUR, USD, etc.)
- **Real estate features** (area in m², number of rooms)
- **Ad cue words** (à vendre, zu verkaufen, louer, etc.)
- **Address patterns** (Rue, Avenue, Strasse, etc.)
- **Swiss postal codes** (4-digit)

## Examples

See [`impresso_pipelines/adclassifier/old/example_usage.py`](impresso_pipelines/adclassifier/old/example_usage.py) for complete examples.

## Performance

- **Macro F1**: 0.9450
- **Balanced Accuracy**: 0.9450
- **Matthews Correlation Coefficient**: 0.8907

Tested on 400 samples (200 ads, 200 non-ads) from historical newspapers.

## Notes

- The model downloads automatically on first use and is cached locally
- Default cache location: `~/.cache/huggingface/hub/`
- Supports batch processing for efficient GPU utilization
- Adaptive thresholding based on language and text length

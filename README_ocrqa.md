# OCR Quality Assessment Pipeline

Make sure you have installed the package as demonstrated in the main [README](README.md).

> **Note:** For more documentation and usage details, see the inline docstrings and comments in the code.

## Overview

The OCR Quality Assessment (OCRQA) pipeline evaluates the quality of OCR-processed text by comparing tokens against language-specific Bloom filter lexicons. The quality score represents the ratio of known words to total words, helping identify poorly OCR'd documents.

## Quick Start

```python
from impresso_pipelines.ocrqa import OCRQAPipeline

# Initialize the pipeline
ocrqa_pipeline = OCRQAPipeline()

# Example text extracted from OCR
de_text = """Vieles Seltsame geschieht auf Erden :
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

# Get quality assessment
result = ocrqa_pipeline(de_text)
print(result)
```

**Expected Output:**

```python
{'language': 'de', 'score': 1.0}
```

## Configuration Options

### Initialization Parameters

```python
OCRQAPipeline(
    repo_id: Optional[str] = None,        # HuggingFace repo (default: "impresso-project/OCR-quality-assessment-unigram")
    revision: str = "main",                # Repository revision/branch/tag
    score_precision: int = 2               # Number of decimal places for score (default: 2)
)
```

**Examples:**

```python
# Use default repository
pipeline = OCRQAPipeline()

# Use custom repository and branch
pipeline = OCRQAPipeline(
    repo_id="my-org/my-ocrqa-models",
    revision="v2.0"
)

# Customize score precision
pipeline = OCRQAPipeline(score_precision=3)  # 3 decimal places
```

### Processing Parameters

```python
pipeline(
    text: str,                             # Input text to assess
    language: Optional[str] = None,        # Language code (auto-detected if None)
    version: Optional[str] = None,         # BloomFilter version (latest if None)
    diagnostics: bool = False,             # Include detailed token analysis
    model_id: bool = False,                # Include model identifier
    supported_languages: bool = False      # Include list of supported languages
)
```

## Advanced Usage

### Diagnostics Mode

Get detailed information about known and unknown tokens:

```python
result = ocrqa_pipeline(de_text, diagnostics=True)
print(result)
```

**Output:**

```python
{
    'language': 'de',
    'score': 0.95,
    'diagnostics': {
        'known_tokens': ['vieles', 'seltsame', 'geschieht', ...],
        'unknown_tokens': ['vegehr', 'zaubrisch'],
        'model_id': 'ocrqa-wp_v2.0.0-de'
    }
}
```

### Explicit Language and Version

```python
# Specify language explicitly (skip auto-detection)
result = ocrqa_pipeline(de_text, language="de")

# Use specific BloomFilter version
result = ocrqa_pipeline(de_text, language="de", version="2.0.0")

# Get model information
result = ocrqa_pipeline(de_text, model_id=True)
```

### Check Supported Languages

```python
result = ocrqa_pipeline(de_text, supported_languages=True)
print(result['supported_languages'])
# Output: ['de', 'en', 'fr', 'lb', ...]
```

## BloomFilter Versions

The pipeline supports multiple BloomFilter versions with different normalization strategies:

### Version 1.x.x (Legacy)

- Simple character normalization
- Replaces punctuation and special characters with spaces
- Normalizes all digits to `0`
- Lowercasing during Unicode normalization

### Version 2.x.x (Current)

- **Enhanced OCR artifact detection**: Preserves special characters (`~`, `|`, `§`, `£`, `#`, `•`, `■`, `+`, `-`) as separate tokens
- **Luxembourgish support**: Preserves word-internal apostrophes after 'e' or 'o' (e.g., `ge'nt`, `kre'en`)
- **Improved tokenization**: Better handling of OCR errors and artifacts
- **Lowercasing after Unicode normalization**: More consistent text processing

**The pipeline automatically selects the appropriate normalization based on the BloomFilter version.**

## Language-Specific Features

### Luxembourgish (lb)

Version 2.x.x includes special handling for Luxembourgish historical texts:

```python
lb_text = "De Mann huet ge'nt an der Stad."
result = ocrqa_pipeline(lb_text, language="lb", version="2.0.0")
# Word-internal apostrophes are preserved: ge'nt stays as "ge'nt"
```

## Score Interpretation

The quality score ranges from 0.0 to 1.0:

- **1.0**: All tokens are known (perfect quality)
- **0.9 - 0.99**: Excellent quality (few unknown tokens)
- **0.8 - 0.89**: Good quality (some OCR errors)
- **0.7 - 0.79**: Fair quality (noticeable errors)
- **< 0.7**: Poor quality (significant OCR problems)

**Note:** The score represents the ratio of known tokens to total tokens. Unknown tokens may indicate:

- OCR errors (e.g., `vegehr` instead of `Begehr`)
- Rare or archaic words
- Proper nouns not in the lexicon
- OCR artifacts (v2 treats these as separate tokens)

## Examples

### Basic Quality Assessment

```python
pipeline = OCRQAPipeline()

good_text = "The quick brown fox jumps over the lazy dog."
result = pipeline(good_text)
print(result)  # {'language': 'en', 'score': 1.0}

poor_text = "Th3 qu!ck br0wn f0x jump$ 0v3r th3 |@zy d0g."
result = pipeline(poor_text)
print(result)  # {'language': 'en', 'score': 0.6}
```

### Custom Precision

```python
# Default: 2 decimal places
pipeline_default = OCRQAPipeline()
result = pipeline_default(text)
print(result['score'])  # 0.95

# Custom: 3 decimal places
pipeline_precise = OCRQAPipeline(score_precision=3)
result = pipeline_precise(text)
print(result['score'])  # 0.954

# Integer scores only
pipeline_int = OCRQAPipeline(score_precision=0)
result = pipeline_int(text)
print(result['score'])  # 1
```

### Multi-language Processing

```python
pipeline = OCRQAPipeline()

texts = {
    "en": "Hello world, this is a test.",
    "de": "Hallo Welt, dies ist ein Test.",
    "fr": "Bonjour le monde, ceci est un test.",
}

for lang, text in texts.items():
    result = pipeline(text, language=lang, model_id=True)
    print(f"{lang}: {result['score']} (model: {result['model_id']})")
```

## Resources

For more detailed examples and usage scenarios, check out our demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/ocrqa_pipeline_demo.ipynb).

## Supported Languages

The pipeline currently supports the following languages (with v2.x.x BloomFilters):

- German (de)
- English (en)
- French (fr)
- Luxembourgish (lb)
- And more...

To check the current list of supported languages:

```python
pipeline = OCRQAPipeline()
result = pipeline("test", supported_languages=True)
print(result['supported_languages'])
```

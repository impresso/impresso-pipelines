# NERLongPipeline

A simple Named Entity Recognition (NER) pipeline for processing long texts in manageable chunks, built on Hugging Face Transformers and PyTorch.

## Features

- **Chunk-aware** token classification to handle texts longer than the model’s max length
- Stride-based overlapping windows to avoid missing entities at chunk boundaries
- Configurable minimum confidence (`min_score`) to filter low-confidence entities
- Suppression list to ignore unwanted entity types (e.g. unknown press agencies, authors)


## NERLong usage example


```python
from impresso_pipelines.nerlong import NERLongPipeline

# Initialize the pipeline
ner = NERLongPipeline()

# Run on your text
text = "Your long document text goes here…"
entities, summary = ner(
    input_text=text,
    min_score=0.30
)

print("Extracted Entities:")
for ent in entities:
    print(ent)

print("\nSummary (max confidence per entity type):")
print(summary)

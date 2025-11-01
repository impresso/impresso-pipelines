### OCR QA Example
Make sure you have installed the package as demonstrated in the main [README](README.md). 

> **Note:** For more documentation and usage details, see the inline docstrings and comments in the code.

```python
from impresso_pipelines.ocrqa import OCRQAPipeline()
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
     

# Get an answer
result = ocrqa_pipeline(de_text)
print(result)
```
**Expected Output:**
```
{'language': 'de', 'score': 1.0}
```
Score roughly represents the ratio between known and unknown words in the text in comparison to the language-specific Bloom filter database.

For a more details about the usage and the possibilities that this pipeline provides, please check out our demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/ocrqa_pipeline_demo.ipynb).
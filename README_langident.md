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


For more details about usage and available features, see the demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/langident_pipeline_demo.ipynb).

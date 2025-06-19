### Solr Normalization Example
Make sure you have installed the package as demonstrated in the main README.

```python
# Initialize the pipeline
pipeline = SolrNormalizationPipeline()

# Example text for normalization
de_text = "Der Hund läuft schnell durch den Wald und über die Wiese."

# Normalize the text
result = pipeline(de_text)
print(result)
```

**Expected Output:**
```
{'language': 'de', 'tokens': ['hund', 'läuft', 'schnell', 'wald', 'wiese']}
```

The pipeline detects the language of the input text (if not explicitly provided) and returns normalized tokens based on language-specific rules.

For more details about usage and customization, please check out our demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/solrnormalization_pipeline_demo.ipynb).

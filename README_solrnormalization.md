### Solr Normalization Example
Make sure you have installed the package as demonstrated in the main [README](README.md).

> **Note:** For more documentation and usage details, see the inline docstrings and comments in the code.

```python
from impresso_pipelines.solrnormalization import SolrNormalizationPipeline
# Initialize the pipeline
pipeline = SolrNormalizationPipeline()

# Example text in German
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

# Normalize the text
result = pipeline(de_text)
print(result)
```

**Expected Output:**
```
{'language': 'de',
 'tokens': 
    ['viel',
    'seltsam',
    'geschieht',
    'erde',
    'seltsamer',
    'sieht',
    'mond',
    'gluck',
    'knopfloch',
    'wohnt',
    'zaubrisch',
    'fasst',
    'ernst',
    'mann',
    'weib',
    'u',
    'kind',
    'frag',
    'reitet',
    'gluck',
    'jage',
    'nacb',
    'stet',
    'vegeh',
    'nebe',
    'reitet',
    'damo',
    'her',
    'ehrgeiz',
    'finster',
    'tuck',
    'jagt',
    'zuletzt',
    'bruck',
    'abgrund',
    'd:m',
    'nachtlich',
    'schwarz',
    'jahling',
    'abbricht']}
```

The pipeline detects the language of the input text (if not explicitly provided) and returns normalized tokens based on language-specific rules.

For more details about usage and customization, please check out our demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/solrnormalization_pipeline_demo.ipynb).

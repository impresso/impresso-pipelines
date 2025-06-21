### Language Identification Example
Make sure you have installed the package as demostrated in the main README. 
```python
from impresso_pipelines.langident import LangIdentPipeline
# Initialize the pipeline
lang_pipeline = LangIdentPipeline()

# Example text in German
de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."
     

# Detect language
result = lang_pipeline(de_text)
print(result)
```
**Expected Output:**
```
{'language': 'de', 'score': 1.0}
```
Score represents the probability of the detected language based on the input text.


For a more details about the usage and the possibilities that this pipeline provides, please check out our demo [notebook](https://github.com/impresso/impresso-datalab-notebooks/blob/main/annotate/langident_pipeline_demo.ipynb). 
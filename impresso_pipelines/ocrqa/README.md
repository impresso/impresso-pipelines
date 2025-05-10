### OCR QA Example
Make sure you have installed the package as demostrated in the main README. 
```python
# Initialize the pipeline
ocrqa_pipeline = OCRQAPipeline()

# Example text extracted from OCR
de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."
     

# Get an answer
result = ocrqa_pipeline(de_text)
print(result)
```
**Expected Output:**
```
{'language': 'de', 'score': 1.0}
```
Score roughly represents the ratio between known and unknown words in the text in comparison to the language-specific Bloom filter database.
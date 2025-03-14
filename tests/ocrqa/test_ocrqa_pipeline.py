import pytest
from impresso_pipelines.ocrqa.ocrqa_pipeline import OCRQAPipeline

def test_ocrqa_pipeline_basic():
    ocrqa_pipeline = OCRQAPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."

    result = ocrqa_pipeline(de_text)

    # assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # assert that result has keys 'question', 'answer'
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_ocrqa_pipeline_with_diagnostics():
    ocrqa_pipeline = OCRQAPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."

    result = ocrqa_pipeline(de_text, diagnostics=True)

    # assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # assert that result has keys 'question', 'answer', 'diagnostics'
    assert 'diagnostics' in result.keys()
    assert 'language' in result.keys()
    assert 'score' in result.keys()
    # assert that diagnostics is a dictionary
    assert isinstance(result['diagnostics'], dict)
    # assert model_id is in diagnostics
    assert 'model_id' in result['diagnostics'].keys()
    # assert that model_id is a string
    assert isinstance(result['diagnostics']['model_id'], str)
    # assert that model_id is not empty
    assert result['diagnostics']['model_id'] != ''

def test_ocrqa_pipeline_version():
    ocrqa_pipeline = OCRQAPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."

    result = ocrqa_pipeline(de_text, version="1.0.5", model_id=True)

    expected = 'ocrqa-wp_v1.0.5-de'
    
    #check that model_id is as expected
    assert result['diagnostics']['model_id'] == expected
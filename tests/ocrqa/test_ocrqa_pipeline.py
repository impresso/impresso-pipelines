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

    result = ocrqa_pipeline(de_text, version="1.0.5", diagnostics=True)

    expected = 'ocrqa-wp_v1.0.5-de'
    
    #check that model_id is as expected
    assert result['diagnostics']['model_id'] == expected

def test_ocrqa_pipeline_basic_fr():
    ocrqa_pipeline = OCRQAPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants. Un jour, il trouva un jardin secret que personne ne connaissait. Max décida d'explorer le jardin et découvrit de nombreuses belles fleurs et animaux. Dès ce jour, il visita le jardin chaque après-midi."
    result = ocrqa_pipeline(fr_text)
    assert isinstance(result, dict)
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_ocrqa_pipeline_with_diagnostics_fr():
    ocrqa_pipeline = OCRQAPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants. Un jour, il trouva un jardin secret que personne ne connaissait. Max décida d'explorer le jardin et découvrit de nombreuses belles fleurs et animaux. Dès ce jour, il visita le jardin chaque après-midi."
    result = ocrqa_pipeline(fr_text, diagnostics=True)
    assert isinstance(result, dict)
    assert 'diagnostics' in result.keys()
    assert 'language' in result.keys()
    assert 'score' in result.keys()
    assert isinstance(result['diagnostics'], dict)
    assert 'model_id' in result['diagnostics'].keys()
    assert isinstance(result['diagnostics']['model_id'], str)
    assert result['diagnostics']['model_id'] != ''

def test_ocrqa_pipeline_version_fr():
    ocrqa_pipeline = OCRQAPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants. Un jour, il trouva un jardin secret que personne ne connaissait. Max décida d'explorer le jardin et découvrit de nombreuses belles fleurs et animaux. Dès ce jour, il visita le jardin chaque après-midi."
    result = ocrqa_pipeline(fr_text, version="1.0.5", diagnostics=True)
    expected = 'ocrqa-wp_v1.0.5-fr'
    assert result['diagnostics']['model_id'] == expected

def test_ocrqa_pipeline_basic_lb():
    ocrqa_pipeline = OCRQAPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result = ocrqa_pipeline(lb_text, language="lb")
    assert isinstance(result, dict)
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_ocrqa_pipeline_with_diagnostics_lb():
    ocrqa_pipeline = OCRQAPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result = ocrqa_pipeline(lb_text, language="lb", diagnostics=True)
    assert isinstance(result, dict)
    assert 'diagnostics' in result.keys()
    assert 'language' in result.keys()
    assert 'score' in result.keys()
    assert isinstance(result['diagnostics'], dict)
    assert 'model_id' in result['diagnostics'].keys()
    assert isinstance(result['diagnostics']['model_id'], str)
    assert result['diagnostics']['model_id'] != ''

def test_ocrqa_pipeline_version_lb():
    ocrqa_pipeline = OCRQAPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result = ocrqa_pipeline(lb_text, language="lb", version="1.0.5", diagnostics=True)
    expected = 'ocrqa-wp_v1.0.5-lb'
    assert result['diagnostics']['model_id'] == expected
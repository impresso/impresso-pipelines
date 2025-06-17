import pytest
from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline

def test_basics():
    lang_pipeline = LangIdentPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."
    result = lang_pipeline(de_text)

    # assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # assert that first value is 'de'
    assert list(result.values())[0] == 'de'
    # assert that result has keys 'language', 'score'
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_diagnostics():
    lang_pipeline = LangIdentPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."
    result_2 = lang_pipeline(de_text, diagnostics=True)
    # assert that the pipeline returns a dictionary
    assert isinstance(result_2, dict)
    # assert that result has keys 'language', 'score', 'diagnostics'
    assert 'diagnostics' in result_2.keys()
    assert 'language' in result_2.keys()
    assert 'score' in result_2.keys()
    # assert that diagnostics is a dictionary
    assert isinstance(result_2['diagnostics'], dict)

def test_model_id():
    lang_pipeline = LangIdentPipeline()
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. Eines Tages fand er einen geheimen Garten, den niemand kannte. Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. Von diesem Tag an besuchte er den Garten jeden Nachmittag."
    result_3 = lang_pipeline(de_text, diagnostics=True, model_id=True)
    # assert that the pipeline returns a dictionary
    assert isinstance(result_3, dict)
    # assert that result has keys 'language', 'score', 'diagnostics', 'model_id'
    assert 'diagnostics' in result_3.keys()
    assert 'language' in result_3.keys()
    assert 'score' in result_3.keys()
    assert 'model_id' in result_3.keys()
    # assert that diagnostics is a dictionary
    assert isinstance(result_3['diagnostics'], dict)
    # assert that model_id is a string
    assert isinstance(result_3['model_id'], str)
    # assert that model_id is not empty
    assert result_3['model_id'] != ''

def test_basics_fr():
    lang_pipeline = LangIdentPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants."
    result = lang_pipeline(fr_text)
    assert isinstance(result, dict)
    assert list(result.values())[0] == 'fr'
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_diagnostics_fr():
    lang_pipeline = LangIdentPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants."
    result_2 = lang_pipeline(fr_text, diagnostics=True)
    assert isinstance(result_2, dict)
    assert 'diagnostics' in result_2.keys()
    assert 'language' in result_2.keys()
    assert 'score' in result_2.keys()
    assert isinstance(result_2['diagnostics'], dict)

def test_model_id_fr():
    lang_pipeline = LangIdentPipeline()
    fr_text = "Un petit chien nommé Max vivait dans un village tranquille. Chaque jour, il courait dans les rues et jouait avec les enfants."
    result_3 = lang_pipeline(fr_text, diagnostics=True, model_id=True)
    assert isinstance(result_3, dict)
    assert 'diagnostics' in result_3.keys()
    assert 'language' in result_3.keys()
    assert 'score' in result_3.keys()
    assert 'model_id' in result_3.keys()
    assert isinstance(result_3['diagnostics'], dict)
    assert isinstance(result_3['model_id'], str)
    assert result_3['model_id'] != ''

def test_basics_lb():
    lang_pipeline = LangIdentPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result = lang_pipeline(lb_text)
    assert isinstance(result, dict)
    assert list(result.values())[0] == 'lb'
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_diagnostics_lb():
    lang_pipeline = LangIdentPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result_2 = lang_pipeline(lb_text, diagnostics=True)
    assert isinstance(result_2, dict)
    assert 'diagnostics' in result_2.keys()
    assert 'language' in result_2.keys()
    assert 'score' in result_2.keys()
    assert isinstance(result_2['diagnostics'], dict)

def test_model_id_lb():
    lang_pipeline = LangIdentPipeline()
    lb_text = "De Max huet gär am Gaart mat senge Frënn gespillt. All Moien huet hien haart gebellt fir jiddereen ze begréissen."
    result_3 = lang_pipeline(lb_text, diagnostics=True, model_id=True)
    assert isinstance(result_3, dict)
    assert 'diagnostics' in result_3.keys()
    assert 'language' in result_3.keys()
    assert 'score' in result_3.keys()
    assert 'model_id' in result_3.keys()
    assert isinstance(result_3['diagnostics'], dict)
    assert isinstance(result_3['model_id'], str)
    assert result_3['model_id'] != ''

def test_basics_en():
    lang_pipeline = LangIdentPipeline()
    en_text = "A small dog named Max lived in a quiet village. Every day he ran through the streets and played with the children."
    result = lang_pipeline(en_text)
    assert isinstance(result, dict)
    assert list(result.values())[0] == 'en'
    assert 'language' in result.keys()
    assert 'score' in result.keys()

def test_diagnostics_en():
    lang_pipeline = LangIdentPipeline()
    en_text = "A small dog named Max lived in a quiet village. Every day he ran through the streets and played with the children."
    result_2 = lang_pipeline(en_text, diagnostics=True)
    assert isinstance(result_2, dict)
    assert 'diagnostics' in result_2.keys()
    assert 'language' in result_2.keys()
    assert 'score' in result_2.keys()
    assert isinstance(result_2['diagnostics'], dict)

def test_model_id_en():
    lang_pipeline = LangIdentPipeline()
    en_text = "A small dog named Max lived in a quiet village. Every day he ran through the streets and played with the children."
    result_3 = lang_pipeline(en_text, diagnostics=True, model_id=True)
    assert isinstance(result_3, dict)
    assert 'diagnostics' in result_3.keys()
    assert 'language' in result_3.keys()
    assert 'score' in result_3.keys()
    assert 'model_id' in result_3.keys()
    assert isinstance(result_3['diagnostics'], dict)
    assert isinstance(result_3['model_id'], str)
    assert result_3['model_id'] != ''




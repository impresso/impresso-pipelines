import pytest
from impresso_pipelines.langident.langident_pipeline import LangIdentPipeline

def test_some_function():
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

    result_2 = lang_pipeline(de_text, diagnostics=True)
    # assert that the pipeline returns a dictionary
    assert isinstance(result_2, dict)
    # assert that result has keys 'language', 'score', 'diagnostics'
    assert 'diagnostics' in result_2.keys()
    assert 'language' in result_2.keys()
    assert 'score' in result_2.keys()
    # assert that diagnostics is a dictionary
    assert isinstance(result_2['diagnostics'], dict)

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
    



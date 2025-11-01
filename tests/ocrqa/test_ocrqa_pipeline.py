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

def test_ocrqa_pipeline_score_precision():
    """Test that score_precision parameter works correctly."""
    de_text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. Jeden Tag rannte er durch die Straßen und spielte mit den Kindern."
    
    # Test default precision (2 decimal places)
    pipeline_default = OCRQAPipeline()
    result_default = pipeline_default(de_text, language="de")
    assert isinstance(result_default, dict)
    assert 'score' in result_default
    score_str = str(result_default['score'])
    # Check that score has at most 2 decimal places
    if '.' in score_str:
        decimals = len(score_str.split('.')[1])
        assert decimals <= 2, f"Default precision should be 2, but got {decimals} decimal places"
    
    # Test custom precision (3 decimal places)
    pipeline_custom = OCRQAPipeline(score_precision=3)
    result_custom = pipeline_custom(de_text, language="de")
    assert isinstance(result_custom, dict)
    assert 'score' in result_custom
    score_str_custom = str(result_custom['score'])
    if '.' in score_str_custom:
        decimals_custom = len(score_str_custom.split('.')[1])
        assert decimals_custom <= 3, f"Custom precision should be 3, but got {decimals_custom} decimal places"
    
    # Test precision 1 (1 decimal place)
    pipeline_one = OCRQAPipeline(score_precision=1)
    result_one = pipeline_one(de_text, language="de")
    assert isinstance(result_one, dict)
    assert 'score' in result_one
    score_str_one = str(result_one['score'])
    if '.' in score_str_one:
        decimals_one = len(score_str_one.split('.')[1])
        assert decimals_one <= 1, f"Precision 1 should have 1 decimal place, but got {decimals_one} decimal places"
    
    # Test precision 0 (no decimal places)
    pipeline_zero = OCRQAPipeline(score_precision=0)
    result_zero = pipeline_zero(de_text, language="de")
    assert isinstance(result_zero, dict)
    assert 'score' in result_zero
    # With precision 0, score should be an integer (0 or 1)
    assert result_zero['score'] in [0, 1] or isinstance(result_zero['score'], int)

def test_ocrqa_pipeline_v2_normalization():
    """Test v2 normalization features including OCR artifacts and Luxembourgish support."""
    ocrqa_pipeline = OCRQAPipeline()
    
    # Test OCR artifacts preservation (v2 feature) using subtokens
    text_with_artifacts = "hello~world test|here"
    tokens = ocrqa_pipeline.subtokens(text_with_artifacts, version="2.0.0")
    # In v2, artifacts are preserved as separate tokens
    assert "~" in tokens, "OCR artifact ~ should be a separate token"
    assert "|" in tokens, "OCR artifact | should be a separate token"
    assert "hello" in tokens
    assert "world" in tokens
    
    # Test Luxembourgish apostrophe handling (v2 feature)
    lb_text = "ge'nt kre'en"  # Luxembourgish words with apostrophes after 'e' or 'o'
    lb_tokens = ocrqa_pipeline.subtokens(lb_text, version="2.0.0", language="lb")
    # In v2 with lb language, word-internal apostrophes should be preserved
    assert "ge'nt" in lb_tokens, "Luxembourgish apostrophes should be preserved in v2"
    assert "kre'en" in lb_tokens, "Luxembourgish apostrophes should be preserved in v2"
    
    # Test that v1 normalization still works
    v1_text = "hello~world"
    v1_tokens = ocrqa_pipeline.subtokens(v1_text, version="1.0.5")
    # In v1, ~ is replaced with space, not a separate token
    assert "~" not in v1_tokens, "v1 should not preserve ~ as separate token"
    assert "hello" in v1_tokens
    assert "world" in v1_tokens
    
    # Test digit normalization (both versions)
    text_with_digits = "Price: 100 dollars"
    v1_tokens_digits = ocrqa_pipeline.subtokens(text_with_digits, version="1.0.5")
    v2_tokens_digits = ocrqa_pipeline.subtokens(text_with_digits, version="2.0.0")
    assert "000" in v1_tokens_digits, "v1 should normalize digits to 0"
    assert "000" in v2_tokens_digits, "v2 should normalize digits to 0"
    
    # Test min_length filtering
    text = "a bc def ghij"
    tokens_min1 = ocrqa_pipeline.subtokens(text, version="2.0.0", min_length=1)
    tokens_min3 = ocrqa_pipeline.subtokens(text, version="2.0.0", min_length=3)
    assert len(tokens_min1) == 4, "min_length=1 should include all tokens"
    assert len(tokens_min3) == 2, "min_length=3 should only include 'def' and 'ghij'"
    assert "def" in tokens_min3 and "ghij" in tokens_min3
    
    # Test lowercase parameter
    text_upper = "HELLO WORLD"
    tokens_lower = ocrqa_pipeline.subtokens(text_upper, version="2.0.0", lowercase=True)
    tokens_no_lower = ocrqa_pipeline.subtokens(text_upper, version="2.0.0", lowercase=False)
    assert "hello" in tokens_lower, "lowercase=True should lowercase text"
    assert "HELLO" in tokens_no_lower, "lowercase=False should preserve case"

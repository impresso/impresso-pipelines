"""
Tests for the OCR Quality Assessment pipeline.

This module contains comprehensive tests for the OCRQAPipeline class,
including basic functionality, diagnostics, version-specific behavior,
and v2 normalization features.
"""

import pytest
from impresso_pipelines.ocrqa.ocrqa_pipeline import OCRQAPipeline, subtokens


# Test fixtures and constants
SAMPLE_TEXT_DE = (
    "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. "
    "Jeden Tag rannte er durch die Straßen und spielte mit den Kindern. "
    "Eines Tages fand er einen geheimen Garten, den niemand kannte. "
    "Max entschied sich, den Garten zu erkunden und entdeckte viele schöne Blumen und Tiere. "
    "Von diesem Tag an besuchte er den Garten jeden Nachmittag."
)

SAMPLE_TEXT_FR = (
    "Un petit chien nommé Max vivait dans un village tranquille. "
    "Chaque jour, il courait dans les rues et jouait avec les enfants. "
    "Un jour, il trouva un jardin secret que personne ne connaissait. "
    "Max décida d'explorer le jardin et découvrit de nombreuses belles fleurs et animaux. "
    "Dès ce jour, il visita le jardin chaque après-midi."
)

SAMPLE_TEXT_LB = (
    "De Max huet gär am Gaart mat senge Frënn gespillt. "
    "All Moien huet hien haart gebellt fir jiddereen ze begréissen."
)


@pytest.fixture
def pipeline():
    """Create a fresh OCRQAPipeline instance for each test."""
    return OCRQAPipeline()


# German language tests
def test_ocrqa_pipeline_basic(pipeline):
    """Test basic pipeline functionality with German text."""
    result = pipeline(SAMPLE_TEXT_DE)

    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    assert isinstance(result['score'], (int, float)), "Score should be numeric"
    assert 0 <= result['score'] <= 1, "Score should be between 0 and 1"


def test_ocrqa_pipeline_with_diagnostics(pipeline):
    """Test pipeline with diagnostics enabled for German text."""
    result = pipeline(SAMPLE_TEXT_DE, diagnostics=True)

    # Basic structure assertions
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'diagnostics' in result, "Result should contain 'diagnostics' key when enabled"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    
    # Diagnostics structure assertions
    assert isinstance(result['diagnostics'], dict), "Diagnostics should be a dictionary"
    assert 'model_id' in result['diagnostics'], "Diagnostics should contain 'model_id'"
    assert isinstance(result['diagnostics']['model_id'], str), "Model ID should be a string"
    assert result['diagnostics']['model_id'], "Model ID should not be empty"
    assert 'known_tokens' in result['diagnostics'], "Diagnostics should contain 'known_tokens'"
    assert 'unknown_tokens' in result['diagnostics'], "Diagnostics should contain 'unknown_tokens'"


def test_ocrqa_pipeline_version(pipeline):
    """Test pipeline with specific BloomFilter version for German text."""
    result = pipeline(SAMPLE_TEXT_DE, version="1.0.5", diagnostics=True)
    expected_model_id = 'ocrqa-wp_v1.0.5-de'
    
    assert result['diagnostics']['model_id'] == expected_model_id, \
        f"Model ID should be '{expected_model_id}' for version 1.0.5 and German"


# French language tests
def test_ocrqa_pipeline_basic_fr(pipeline):
    """Test basic pipeline functionality with French text."""
    result = pipeline(SAMPLE_TEXT_FR)
    
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    assert result['language'] == 'fr', "Should detect French language"


def test_ocrqa_pipeline_with_diagnostics_fr(pipeline):
    """Test pipeline with diagnostics enabled for French text."""
    result = pipeline(SAMPLE_TEXT_FR, diagnostics=True)
    
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'diagnostics' in result, "Result should contain 'diagnostics' key when enabled"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    assert isinstance(result['diagnostics'], dict), "Diagnostics should be a dictionary"
    assert 'model_id' in result['diagnostics'], "Diagnostics should contain 'model_id'"
    assert isinstance(result['diagnostics']['model_id'], str), "Model ID should be a string"
    assert result['diagnostics']['model_id'], "Model ID should not be empty"


def test_ocrqa_pipeline_version_fr(pipeline):
    """Test pipeline with specific BloomFilter version for French text."""
    result = pipeline(SAMPLE_TEXT_FR, version="1.0.5", diagnostics=True)
    expected_model_id = 'ocrqa-wp_v1.0.5-fr'
    
    assert result['diagnostics']['model_id'] == expected_model_id, \
        f"Model ID should be '{expected_model_id}' for version 1.0.5 and French"


# Luxembourgish language tests
def test_ocrqa_pipeline_basic_lb(pipeline):
    """Test basic pipeline functionality with Luxembourgish text."""
    result = pipeline(SAMPLE_TEXT_LB, language="lb")
    
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    assert result['language'] == 'lb', "Should use specified Luxembourgish language"


def test_ocrqa_pipeline_with_diagnostics_lb(pipeline):
    """Test pipeline with diagnostics enabled for Luxembourgish text."""
    result = pipeline(SAMPLE_TEXT_LB, language="lb", diagnostics=True)
    
    assert isinstance(result, dict), "Pipeline should return a dictionary"
    assert 'diagnostics' in result, "Result should contain 'diagnostics' key when enabled"
    assert 'language' in result, "Result should contain 'language' key"
    assert 'score' in result, "Result should contain 'score' key"
    assert isinstance(result['diagnostics'], dict), "Diagnostics should be a dictionary"
    assert 'model_id' in result['diagnostics'], "Diagnostics should contain 'model_id'"
    assert isinstance(result['diagnostics']['model_id'], str), "Model ID should be a string"
    assert result['diagnostics']['model_id'], "Model ID should not be empty"


def test_ocrqa_pipeline_version_lb(pipeline):
    """Test pipeline with specific BloomFilter version for Luxembourgish text."""
    result = pipeline(SAMPLE_TEXT_LB, language="lb", version="1.0.5", diagnostics=True)
    expected_model_id = 'ocrqa-wp_v1.0.5-lb'
    
    assert result['diagnostics']['model_id'] == expected_model_id, \
        f"Model ID should be '{expected_model_id}' for version 1.0.5 and Luxembourgish"


# Score precision tests
def test_ocrqa_pipeline_score_precision():
    """Test that score_precision parameter controls decimal places correctly."""
    test_text = (
        "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf. "
        "Jeden Tag rannte er durch die Straßen und spielte mit den Kindern."
    )
    
    # Test default precision (2 decimal places)
    pipeline_default = OCRQAPipeline()
    result_default = pipeline_default(test_text, language="de")
    assert 'score' in result_default, "Result should contain 'score' key"
    score = result_default['score']
    assert isinstance(score, (int, float)), "Score should be numeric"
    _assert_decimal_precision(score, max_precision=2, description="default")
    
    # Test custom precision (3 decimal places)
    pipeline_custom = OCRQAPipeline(score_precision=3)
    result_custom = pipeline_custom(test_text, language="de")
    assert 'score' in result_custom, "Result should contain 'score' key"
    score = result_custom['score']
    assert isinstance(score, (int, float)), "Score should be numeric"
    _assert_decimal_precision(score, max_precision=3, description="custom (3)")
    
    # Test precision 1 (1 decimal place)
    pipeline_one = OCRQAPipeline(score_precision=1)
    result_one = pipeline_one(test_text, language="de")
    assert 'score' in result_one, "Result should contain 'score' key"
    score = result_one['score']
    assert isinstance(score, (int, float)), "Score should be numeric"
    _assert_decimal_precision(score, max_precision=1, description="precision 1")
    
    # Test precision 0 (no decimal places - integer)
    pipeline_zero = OCRQAPipeline(score_precision=0)
    result_zero = pipeline_zero(test_text, language="de")
    assert 'score' in result_zero, "Result should contain 'score' key"
    assert result_zero['score'] in [0, 1] or isinstance(result_zero['score'], int), \
        "With precision 0, score should be 0 or 1 (integer)"


# V2 normalization feature tests
def test_ocrqa_pipeline_v2_ocr_artifacts():
    """Test v2 normalization preserves OCR artifacts as separate tokens."""
    text_with_artifacts = "hello~world test|here"
    tokens = subtokens(text_with_artifacts, version="2.0.0")
    
    # OCR artifacts should be separate tokens in v2
    assert "~" in tokens, "OCR artifact '~' should be preserved as separate token in v2"
    assert "|" in tokens, "OCR artifact '|' should be preserved as separate token in v2"
    assert "hello" in tokens, "Word tokens should be preserved"
    assert "world" in tokens, "Word tokens should be preserved"
    assert "test" in tokens, "Word tokens should be preserved"
    assert "here" in tokens, "Word tokens should be preserved"


def test_ocrqa_pipeline_v2_luxembourgish_apostrophes():
    """Test v2 normalization preserves Luxembourgish word-internal apostrophes."""
    lb_text = "ge'nt kre'en"  # Luxembourgish words with apostrophes after 'e' or 'o'
    lb_tokens = subtokens(lb_text, version="2.0.0", language="lb")
    
    # Luxembourgish apostrophes should be preserved within words in v2
    assert "ge'nt" in lb_tokens, \
        "Luxembourgish word-internal apostrophes should be preserved in v2"
    assert "kre'en" in lb_tokens, \
        "Luxembourgish word-internal apostrophes should be preserved in v2"


def test_ocrqa_pipeline_v1_normalization_compatibility():
    """Test that v1 normalization still works correctly (backward compatibility)."""
    v1_text = "hello~world"
    v1_tokens = subtokens(v1_text, version="1.0.5")
    
    # In v1, ~ is replaced with space, not preserved as separate token
    assert "~" not in v1_tokens, \
        "v1 should not preserve '~' as separate token (replaced with space)"
    assert "hello" in v1_tokens, "Word tokens should be preserved in v1"
    assert "world" in v1_tokens, "Word tokens should be preserved in v1"


def test_ocrqa_pipeline_digit_normalization():
    """Test digit normalization works in both v1 and v2."""
    text_with_digits = "Price: 100 dollars"
    v1_tokens_digits = subtokens(text_with_digits, version="1.0.5")
    v2_tokens_digits = subtokens(text_with_digits, version="2.0.0")
    
    assert "000" in v1_tokens_digits, "v1 should normalize all digits to '0'"
    assert "000" in v2_tokens_digits, "v2 should normalize all digits to '0'"
    assert "price" in v1_tokens_digits, "Word tokens should be preserved and lowercased"
    assert "dollars" in v1_tokens_digits, "Word tokens should be preserved and lowercased"


def test_ocrqa_pipeline_min_length_filtering():
    """Test min_length parameter filters tokens correctly."""
    text = "a bc def ghij"
    tokens_min1 = subtokens(text, version="2.0.0", min_length=1)
    tokens_min3 = subtokens(text, version="2.0.0", min_length=3)
    
    assert len(tokens_min1) == 4, \
        "min_length=1 should include all tokens: 'a', 'bc', 'def', 'ghij'"
    assert len(tokens_min3) == 2, \
        "min_length=3 should only include 'def' and 'ghij'"
    assert set(tokens_min3) == {"def", "ghij"}, \
        "min_length=3 should filter out tokens shorter than 3 characters"


def test_ocrqa_pipeline_lowercase_parameter():
    """Test lowercase parameter controls text casing correctly."""
    text_upper = "HELLO WORLD"
    tokens_lower = subtokens(text_upper, version="2.0.0", lowercase=True)
    tokens_no_lower = subtokens(text_upper, version="2.0.0", lowercase=False)
    
    assert "hello" in tokens_lower, "lowercase=True should convert text to lowercase"
    assert "world" in tokens_lower, "lowercase=True should convert text to lowercase"
    assert "HELLO" in tokens_no_lower, "lowercase=False should preserve original case"
    assert "WORLD" in tokens_no_lower, "lowercase=False should preserve original case"


# Helper functions
def _assert_decimal_precision(value: int | float, max_precision: int, description: str) -> None:
    """
    Assert that a numeric value has at most the specified number of decimal places.
    
    Args:
        value: The numeric value to check
        max_precision: Maximum allowed decimal places
        description: Description for error messages
    """
    value_str = str(value)
    if '.' in value_str:
        decimals = len(value_str.split('.')[1])
        assert decimals <= max_precision, \
            f"{description} precision should be {max_precision}, but got {decimals} decimal places"

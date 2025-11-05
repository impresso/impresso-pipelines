import pytest
import glob
import os

from impresso_pipelines.solrnormalization.solrnormalization_pipeline import SolrNormalizationPipeline

@pytest.fixture(scope="session")
def shared_pipeline():
    """Create a single pipeline instance for all tests to share (JVM can only start once)."""
    pipeline = SolrNormalizationPipeline()
    yield pipeline
    pipeline.cleanup()

def test_solrnormalization_pipeline_importable():
    """Test that SolrNormalizationPipeline can be imported and instantiated."""
    pipeline = SolrNormalizationPipeline()
    assert pipeline is not None
    # Don't cleanup - leave for session cleanup

def test_solrnormalization_pipeline_error_on_unsupported_language(shared_pipeline):
    """Test that unsupported language raises ValueError (no JVM/Lucene needed)."""
    with pytest.raises(ValueError):
        shared_pipeline("This is English text.", lang="ru")

def test_solrnormalization_pipeline_basic(shared_pipeline):
    """Test basic pipeline functionality with automatic JAR download."""
    de_text = "Der Hund läuft schnell durch den Wald und über die Wiese."

    result = shared_pipeline(de_text)

    # Assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # Assert that the result contains 'language' and 'tokens' keys
    assert 'language' in result.keys()
    assert 'tokens' in result.keys()
    # Assert that 'tokens' is a list
    assert isinstance(result['tokens'], list)
    # Assert that the detected language is 'de'
    assert result['language'] == 'de'

def test_solrnormalization_pipeline_with_language(shared_pipeline):
    """Test pipeline with explicit language specification."""
    fr_text = "Le chien court rapidement à travers la forêt et sur la prairie."

    result = shared_pipeline(fr_text, lang="fr")

    # Assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # Assert that the result contains 'language' and 'tokens' keys
    assert 'language' in result.keys()
    assert 'tokens' in result.keys()
    # Assert that 'tokens' is a list
    assert isinstance(result['tokens'], list)
    # Assert that the specified language is 'fr'
    assert result['language'] == 'fr'

def test_solrnormalization_pipeline_detect_language(shared_pipeline):
    """Test automatic language detection."""
    de_text = "Der Hund läuft schnell durch den Wald und über die Wiese."

    result = shared_pipeline(de_text)

    # Assert that the detected language is 'de'
    assert result['language'] == 'de'
    # Assert that tokens are normalized correctly (stemmed)
    assert "hund" in result['tokens']
    assert "wald" in result['tokens']
    assert "wies" in result['tokens']  # "wiese" gets stemmed to "wies"

def test_solrnormalization_pipeline_detect_language_fr(shared_pipeline):
    """Test automatic language detection for French."""
    fr_text = "Le chien court rapidement à travers la forêt et sur la prairie."

    result = shared_pipeline(fr_text)

    # Assert that the detected language is 'fr'
    assert result['language'] == 'fr'
    # Assert that tokens are normalized correctly (stemmed, accents removed)
    assert "chien" in result['tokens']
    assert "foret" in result['tokens']  # "forêt" gets normalized to "foret" (accent removed)
    assert "prairi" in result['tokens']  # "prairie" gets stemmed to "prairi"

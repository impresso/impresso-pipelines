import pytest

@pytest.fixture(scope="session", autouse=True)
def ensure_jvm():
    import jpype
    if not jpype.isJVMStarted():
        try:
            jpype.startJVM(classpath=["lucene_jars/*"])
        except RuntimeError:
            pytest.skip("Lucene JARs not available or JVM failed to start", allow_module_level=True)
    try:
        from org.apache.lucene.analysis.standard import StandardAnalyzer
    except ImportError:
        pytest.skip("Lucene JARs not available or JVM failed to start", allow_module_level=True)

from impresso_pipelines.solrnormalization.solrnormalization_pipeline import SolrNormalizationPipeline

def test_solrnormalization_pipeline_basic():
    pipeline = SolrNormalizationPipeline()
    de_text = "Der Hund läuft schnell durch den Wald und über die Wiese."

    result = pipeline(de_text)

    # Assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # Assert that the result contains 'language' and 'tokens' keys
    assert 'language' in result.keys()
    assert 'tokens' in result.keys()
    # Assert that 'tokens' is a list
    assert isinstance(result['tokens'], list)
    # Assert that the detected language is 'de'
    assert result['language'] == 'de'

def test_solrnormalization_pipeline_with_language():
    pipeline = SolrNormalizationPipeline()
    fr_text = "Le chien court rapidement à travers la forêt et sur la prairie."

    result = pipeline(fr_text, lang="fr")

    # Assert that the pipeline returns a dictionary
    assert isinstance(result, dict)
    # Assert that the result contains 'language' and 'tokens' keys
    assert 'language' in result.keys()
    assert 'tokens' in result.keys()
    # Assert that 'tokens' is a list
    assert isinstance(result['tokens'], list)
    # Assert that the specified language is 'fr'
    assert result['language'] == 'fr'

def test_solrnormalization_pipeline_detect_language():
    pipeline = SolrNormalizationPipeline()
    de_text = "Der Hund läuft schnell durch den Wald und über die Wiese."

    result = pipeline(de_text)

    # Assert that the detected language is 'de'
    assert result['language'] == 'de'
    # Assert that tokens are normalized correctly
    assert "hund" in result['tokens']
    assert "wald" in result['tokens']
    assert "wiese" in result['tokens']

def test_solrnormalization_pipeline_detect_language_fr():
    pipeline = SolrNormalizationPipeline()
    fr_text = "Le chien court rapidement à travers la forêt et sur la prairie."

    result = pipeline(fr_text)

    # Assert that the detected language is 'fr'
    assert result['language'] == 'fr'
    # Assert that tokens are normalized correctly
    assert "chien" in result['tokens']
    assert "forêt" in result['tokens']
    assert "prairie" in result['tokens']

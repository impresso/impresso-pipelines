import pytest
from impresso_pipelines.ldatopics.mallet_pipeline import LDATopicsPipeline

def test_pipeline_with_language():
    pipeline = LDATopicsPipeline()
    text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf."
    result = pipeline(text, language="de")

    # Assert that the result is a list
    assert isinstance(result, list)
    # Assert that each entry in the result contains expected keys
    for entry in result:
        assert "ci_id" in entry
        assert "topics" in entry
        assert "topic_model_description" in entry

def test_pipeline_without_language():
    pipeline = LDATopicsPipeline()
    text = "Un petit chien nommé Max vivait dans un village calme."
    result = pipeline(text)

    # Assert that the result is a list
    assert isinstance(result, list)
    # Assert that the detected language is 'fr'
    assert result[0]["lg"] == "fr"
    # Assert that each entry in the result contains expected keys
    for entry in result:
        assert "ci_id" in entry
        assert "topics" in entry
        assert "topic_model_description" in entry

def test_pipeline_with_unsupported_language():
    pipeline = LDATopicsPipeline()
    text = "This is a test sentence in English."

    with pytest.raises(ValueError, match="Unsupported language: en"):
        pipeline(text, language="en")

def test_pipeline_with_custom_doc_name():
    pipeline = LDATopicsPipeline()
    text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf."
    result = pipeline(text, language="de", doc_name="custom_doc")

    assert isinstance(result, list)
    assert len(result) > 0
    assert result[0]["ci_id"] == "custom_doc"

def test_pipeline_automatic_doc_naming():
    pipeline = LDATopicsPipeline()
    text = "Ein kleiner Hund namens Max lebte in einem ruhigen Dorf."
    
    # First document should be doc0
    result1 = pipeline(text, language="de")
    assert result1[0]["ci_id"] == "doc0"
    
    # Second document should be doc1
    result2 = pipeline(text, language="de")
    assert result2[0]["ci_id"] == "doc1"
    
    # Third document should be doc2
    result3 = pipeline(text, language="de")
    assert result3[0]["ci_id"] == "doc2"

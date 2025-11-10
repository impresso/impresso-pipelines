"""
Tests for the Ad Classification Pipeline.
"""

import pytest
from impresso_pipelines.oneclassadclassifier import AdClassificationPipeline


@pytest.fixture
def pipeline():
    """Create an AdClassificationPipeline instance."""
    return AdClassificationPipeline()


class TestAdClassificationPipeline:
    """Test suite for AdClassificationPipeline."""

    def test_pipeline_initialization(self, pipeline):
        """Test that the pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.model is not None
        assert pipeline.tokenizer is not None
        assert pipeline.device in ["cuda", "mps", "cpu"]

    def test_single_text_classification(self, pipeline):
        """Test classification of a single text string."""
        text = "À vendre: Belle villa 5 pièces, CHF 850'000. Tél. 021 123 45 67"
        result = pipeline(text)
        
        assert isinstance(result, dict)
        assert "type" in result
        assert result["type"] in ["ad", "non-ad"]
        assert "promotion_prob" in result
        assert "promotion_prob_final" in result
        assert 0 <= result["promotion_prob"] <= 1
        assert 0 <= result["promotion_prob_final"] <= 1

    def test_list_of_texts_classification(self, pipeline):
        """Test classification of a list of text strings."""
        texts = [
            "À vendre: Belle villa 5 pièces, CHF 850'000. Tél. 021 123 45 67",
            "Le conseil municipal s'est réuni hier pour discuter du budget."
        ]
        results = pipeline(texts)
        
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)
            assert "type" in result
            assert result["type"] in ["ad", "non-ad"]

    def test_single_dict_classification(self, pipeline):
        """Test classification of a single dictionary."""
        doc = {
            "id": "test-doc-1",
            "lg": "fr",
            "ft": "À louer appartement 3 pièces, loyer CHF 1200.-/mois. Contact: 078 123 45 67"
        }
        result = pipeline(doc)
        
        assert isinstance(result, dict)
        assert result["id"] == "test-doc-1"
        assert result["type"] in ["ad", "non-ad"]

    def test_list_of_dicts_classification(self, pipeline):
        """Test classification of a list of dictionaries."""
        docs = [
            {
                "id": "doc1",
                "lg": "fr",
                "ft": "Villa à vendre, 4 pièces, jardin, CHF 650'000. Tél. 021 555 1234"
            },
            {
                "id": "doc2",
                "lg": "de",
                "ft": "Der Bundesrat hat heute neue Massnahmen beschlossen."
            }
        ]
        results = pipeline(docs)
        
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[1]["id"] == "doc2"
        for result in results:
            assert result["type"] in ["ad", "non-ad"]

    def test_empty_text(self, pipeline):
        """Test classification of empty text."""
        result = pipeline("")
        
        assert isinstance(result, dict)
        assert "type" in result

    def test_short_text(self, pipeline):
        """Test classification of short text."""
        result = pipeline("Short text")
        
        assert isinstance(result, dict)
        assert "type" in result

    def test_output_fields(self, pipeline):
        """Test that all expected output fields are present."""
        text = "À vendre: Belle villa"
        result = pipeline(text)
        
        expected_fields = [
            "type",
            "promotion_prob",
            "promotion_prob_final",
            "ensemble_ad_signal",
            "xgenre_top_label",
            "xgenre_top_prob",
            "threshold_used",
            "rule_score",
            "rule_confidence",
            "model_confidence"
        ]
        
        for field in expected_fields:
            assert field in result, f"Missing field: {field}"

    def test_language_handling(self, pipeline):
        """Test that different languages are handled correctly."""
        docs = [
            {"id": "fr", "lg": "fr", "ft": "À vendre maison"},
            {"id": "de", "lg": "de", "ft": "Zu verkaufen Haus"},
            {"id": "none", "lg": None, "ft": "For sale house"}
        ]
        results = pipeline(docs)
        
        assert len(results) == 3
        for result in results:
            assert "type" in result

    def test_rule_based_features(self, pipeline):
        """Test that rule-based features are detected."""
        # Text with clear ad indicators
        text = "À vendre: Villa 5 pièces, 150m², CHF 850'000. Tél. 021 123 45 67, Rue de la Paix 12, 1000 Lausanne"
        result = pipeline(text)
        
        # Should have high rule score due to multiple indicators
        assert "rule_score" in result
        assert "rule_confidence" in result

    def test_batch_processing(self, pipeline):
        """Test efficient batch processing of multiple documents."""
        # Create a larger batch to test batching
        docs = [
            {"id": f"doc{i}", "ft": f"Sample text {i}"} 
            for i in range(20)
        ]
        results = pipeline(docs)
        
        assert len(results) == 20
        for i, result in enumerate(results):
            assert result["id"] == f"doc{i}"

    def test_multilingual_content(self, pipeline):
        """Test classification of multilingual content."""
        text = "À vendre / Zu verkaufen: Belle maison / Schönes Haus. CHF 500'000. Tel: 021 123 45 67"
        result = pipeline(text)
        
        assert isinstance(result, dict)
        assert result["type"] in ["ad", "non-ad"]


@pytest.mark.parametrize("device", ["cpu"])
def test_device_specification(device):
    """Test that device can be specified explicitly."""
    pipeline = AdClassificationPipeline(device=device)
    assert pipeline.device == device


def test_custom_parameters():
    """Test that custom parameters can be set."""
    pipeline = AdClassificationPipeline(
        batch_size=8,
        ad_threshold=0.5,
        temperature=1.0
    )
    
    assert pipeline.batch_size == 8
    assert pipeline.ad_threshold == 0.5
    assert pipeline.temperature == 1.0

"""
Tests for offline/cache-first initialization of pipelines.

This module validates that both OCRQAPipeline and LangIdentPipeline can
initialize without network access when local_files_only=True and the
cache is already populated.
"""

import os

import pytest

from impresso_pipelines.langident import LangIdentPipeline
from impresso_pipelines.ocrqa import OCRQAPipeline


@pytest.fixture(scope="module")
def warm_cache():
    """
    Ensure the cache is populated before running offline tests.

    This fixture runs once per module and downloads required models
    in online mode so that subsequent offline tests can succeed.
    """
    # Warm OCRQA cache
    try:
        ocrqa = OCRQAPipeline(local_files_only=False)
        ocrqa("Test text for cache warming", language="en")
    except Exception as e:
        pytest.skip(f"Could not warm OCRQA cache: {e}")

    # Warm LangIdent cache
    try:
        langident = LangIdentPipeline(local_files_only=False)
        langident("Test text for cache warming")
    except Exception as e:
        pytest.skip(f"Could not warm LangIdent cache: {e}")


class TestLangIdentOfflineMode:
    """Tests for LangIdentPipeline offline initialization."""

    def test_langident_online_mode(self):
        """Test that LangIdentPipeline works in default online mode."""
        pipeline = LangIdentPipeline(local_files_only=False)
        result = pipeline("This is a test in English")

        assert isinstance(result, dict)
        assert "language" in result
        assert "score" in result
        assert isinstance(result["language"], str)

    def test_langident_offline_mode(self, warm_cache):
        """Test that LangIdentPipeline works with local_files_only=True."""
        pipeline = LangIdentPipeline(local_files_only=True)
        result = pipeline("Ceci est un texte en français")

        assert isinstance(result, dict)
        assert "language" in result
        assert "score" in result
        assert result["language"] == "fr"

    def test_langident_offline_with_env_var(self, warm_cache):
        """Test offline mode with HF_HUB_OFFLINE environment variable."""
        # Set environment variable to enforce offline mode
        original_value = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"

            pipeline = LangIdentPipeline(local_files_only=True)
            result = pipeline("Das ist ein deutscher Text")

            assert isinstance(result, dict)
            assert "language" in result
            assert result["language"] == "de"
        finally:
            # Restore original environment
            if original_value is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_value

    def test_langident_offline_with_diagnostics(self, warm_cache):
        """Test offline mode with diagnostics enabled."""
        pipeline = LangIdentPipeline(local_files_only=True)
        result = pipeline("English text sample", diagnostics=True)

        assert isinstance(result, dict)
        assert "language" in result
        assert "diagnostics" in result
        assert isinstance(result["diagnostics"], dict)


class TestOCRQAOfflineMode:
    """Tests for OCRQAPipeline offline initialization."""

    def test_ocrqa_online_mode(self):
        """Test that OCRQAPipeline works in default online mode."""
        pipeline = OCRQAPipeline(local_files_only=False)
        result = pipeline("This is good quality text", language="en")

        assert isinstance(result, dict)
        assert "language" in result
        assert "score" in result
        assert result["language"] == "en"
        assert 0 <= result["score"] <= 1

    def test_ocrqa_offline_mode(self, warm_cache):
        """Test that OCRQAPipeline works with local_files_only=True."""
        pipeline = OCRQAPipeline(local_files_only=True)
        result = pipeline("This is good quality text", language="en")

        assert isinstance(result, dict)
        assert "language" in result
        assert "score" in result
        assert result["language"] == "en"
        assert 0 <= result["score"] <= 1

    def test_ocrqa_offline_with_env_var(self, warm_cache):
        """Test offline mode with HF_HUB_OFFLINE environment variable."""
        original_value = os.environ.get("HF_HUB_OFFLINE")
        try:
            os.environ["HF_HUB_OFFLINE"] = "1"

            pipeline = OCRQAPipeline(local_files_only=True)
            result = pipeline("Das ist ein guter deutscher Text", language="de")

            assert isinstance(result, dict)
            assert "language" in result
            assert result["language"] == "de"
            assert 0 <= result["score"] <= 1
        finally:
            # Restore original environment
            if original_value is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_value

    def test_ocrqa_offline_with_diagnostics(self, warm_cache):
        """Test offline mode with diagnostics enabled."""
        pipeline = OCRQAPipeline(local_files_only=True)
        result = pipeline("Good quality English text", language="en", diagnostics=True)

        assert isinstance(result, dict)
        assert "language" in result
        assert "diagnostics" in result
        assert isinstance(result["diagnostics"], dict)
        assert "known_tokens" in result["diagnostics"]
        assert "unknown_tokens" in result["diagnostics"]
        assert "model_id" in result["diagnostics"]

    def test_ocrqa_offline_with_auto_language_detection(self, warm_cache):
        """Test offline mode with automatic language detection."""
        pipeline = OCRQAPipeline(local_files_only=True)

        # Test with German text (should auto-detect)
        result = pipeline("Ein deutscher Text mit guter Qualität")

        assert isinstance(result, dict)
        assert "language" in result
        assert "score" in result
        assert result["language"] == "de"

    def test_ocrqa_offline_score_precision(self, warm_cache):
        """Test offline mode with custom score precision."""
        pipeline = OCRQAPipeline(local_files_only=True, score_precision=3)
        result = pipeline("Test text", language="en")

        assert isinstance(result, dict)
        assert "score" in result
        # Score should be rounded to 3 decimal places
        score_str = str(result["score"])
        if "." in score_str:
            decimals = len(score_str.split(".")[1])
            assert decimals <= 3

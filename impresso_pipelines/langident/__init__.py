"""
impresso_pipelines.langident: Tools for identifying languages in historical newspaper texts.

Includes pipelines for applying language models (e.g., fastText/floret) at the article or paragraph level,
designed for multilingual and noisy OCR input. Outputs include predictions, confidences, and optional diagnostics.
"""

try:
    import huggingface_hub
    import floret

    from .langident_pipeline import LangIdentPipeline

except ImportError:
    raise ImportError(
        "The langident subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[langident]'"
    )

"""
impresso_pipelines.ocrqa: Tools for assessing OCR quality in historical newspaper collections.

Includes sentence- and article-level quality scoring components, as well as aggregate visualizations
over time, by newspaper, or other metadata.
"""

try:
    import huggingface_hub
    import floret
    import pybloomfilter  # Change this to match what's actually needed
    
    # Only import this after checking dependencies
    from .ocrqa_pipeline import OCRQAPipeline
except ImportError:
    raise ImportError(
        "The ocrqa subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[ocrqa]'"
    )
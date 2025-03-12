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
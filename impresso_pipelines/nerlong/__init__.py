try:
    import torch
    import transformers
   

    # Only import this after checking dependencies
    from .nerlong_pipeline import NERLongPipeline
except ImportError:
    raise ImportError(
        "The ocrqa subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[nerlong]'"
    )
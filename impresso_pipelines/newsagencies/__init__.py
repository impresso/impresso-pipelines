try:
    import torch
    import transformers
   

    # Only import this after checking dependencies
    from .newsagencies_pipeline import NewsAgenciesPipeline
except ImportError:
    raise ImportError(
        "The ocrqa subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[nerlong]'"
    )
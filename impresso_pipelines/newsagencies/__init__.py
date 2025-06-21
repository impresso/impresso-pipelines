"""
impresso_pipelines.newsagencies: Named entity recognition for identifying news agency mentions.

Provides pipelines for token classification and chunk-based annotation of press agencies
such as Reuters or Havas, including diagnostics and export options.
"""

try:
    import torch
    import transformers
   

    # Only import this after checking dependencies
    from .newsagencies_pipeline import NewsAgenciesPipeline
except ImportError:
    raise ImportError(
        "The newsagencies subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[newsagencies]'"
    )
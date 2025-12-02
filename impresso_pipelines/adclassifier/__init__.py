"""
impresso_pipelines.adclassifier: Advertisement classifier for historical newspapers.

Identifies advertisements in historical newspaper text using a fine-tuned
XLM-RoBERTa model combined with rule-based features and adaptive thresholding.
"""

try:
    import torch
    import numpy
    import transformers
    
    # Only import after checking dependencies
    from .adclassifier_pipeline import AdClassifierPipeline
except ImportError:
    raise ImportError(
        "The adclassifier subpackage requires additional dependencies. "
        "Please install them with: pip install impresso_pipelines[adclassifier]"
    )

__all__ = ["AdClassifierPipeline"]

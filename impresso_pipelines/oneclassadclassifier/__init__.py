"""
impresso_pipelines.oneclassadclassifier: Ad classification for historical newspapers.

Identifies advertisements in historical newspaper text using a fine-tuned
XLM-RoBERTa model combined with rule-based features and adaptive thresholding.
"""

try:
    import torch
    import numpy
    import transformers
    
    # Only import after checking dependencies
    from .oneclassadclassifier_pipeline import AdClassificationPipeline
except ImportError:
    raise ImportError(
        "The oneclassadclassifier subpackage requires additional dependencies. "
        "Please install them with: pip install impresso_pipelines[oneclassadclassifier]"
    )

__all__ = ["AdClassificationPipeline"]

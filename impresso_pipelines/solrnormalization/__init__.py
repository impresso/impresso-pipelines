"""
impresso_pipelines.solrnormalization: Tools for text normalization using Apache Lucene analyzers.

Provides pipelines for language detection, tokenization, and normalization of historical newspaper texts.
Supports multiple languages (e.g., German and French) with a potential of custom stopword handling and stemming.
"""

try:
    import jpype
    # import pybloomfilter  # Change this to match what's actually needed
    
    # Only import this after checking dependencies
    from .solrnormalization_pipeline import SolrNormalizationPipeline
except ImportError:
    raise ImportError(
        "The solrnormalization subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[solrnormalization]'"
    )
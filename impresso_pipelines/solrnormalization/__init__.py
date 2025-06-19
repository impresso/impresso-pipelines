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
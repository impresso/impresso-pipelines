try:
    import huggingface_hub
    import floret
    import spacy
    
    # Only import this after checking dependencies
    from .mallet_pipeline import MalletPipeline
except ImportError:
    raise ImportError(
        "The mallet subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[mallet]'"
    )
from .langident_pipeline import LangIdentPipeline


try:
    import huggingface_hub
    import floret
except ImportError:
    raise ImportError(
        "The langident subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[langident]'"
    )

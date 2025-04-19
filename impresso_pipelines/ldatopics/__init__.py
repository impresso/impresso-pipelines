try:
    import huggingface_hub
    import floret
    import spacy
    import jpype
    import smart_open
    import boto3
    import dotenv


    from .mallet_pipeline import LDATopicsPipeline
except ImportError:
    raise ImportError(
        "The mallet subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[mallet]'"
    )
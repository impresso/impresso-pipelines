"""
impresso_pipelines.ldatopics: Components for extracting and analyzing topics using Latent Dirichlet Allocation.

Designed to work on preprocessed newspaper texts. Includes tools for training models, inferring topic distributions,
and inspecting topic-term associations over time.
"""


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
        "Please install them with: pip install 'impresso-pipelines[ldatopics]'"
    )
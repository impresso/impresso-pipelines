"""
Joint media-source named entity recognition for Impresso text.

Provides a model-protocol-aware pipeline for press agencies, radio stations,
and future media-source classes without changing the legacy news-agency API.
"""

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from .mediasources_pipeline import MediaSourcesPipeline
except ImportError as exc:
    raise ImportError(
        "The mediasources subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[mediasources]'"
    ) from exc

"""
Joint media-agency named entity recognition for Impresso text.

Provides a model-protocol-aware pipeline for press agencies, radio stations,
and future media-agency classes without changing the legacy news-agency API.
"""

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401

    from .mediaagencies_pipeline import MediaAgenciesPipeline
except ImportError as exc:
    raise ImportError(
        "The mediaagencies subpackage requires additional dependencies. "
        "Please install them with: pip install 'impresso-pipelines[mediaagencies]'"
    ) from exc

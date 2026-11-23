"""Deprecated alias for :mod:`impresso_pipelines.mediaagencies`."""

import warnings

from impresso_pipelines.mediaagencies import *  # noqa: F401,F403
from impresso_pipelines.mediaagencies import MediaAgenciesPipeline

MediaSourcesPipeline = MediaAgenciesPipeline

warnings.warn(
    "impresso_pipelines.mediasources is deprecated; use impresso_pipelines.mediaagencies",
    DeprecationWarning,
    stacklevel=2,
)

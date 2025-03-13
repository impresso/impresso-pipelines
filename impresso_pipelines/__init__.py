try:
    from .langident import langident_pipeline
except ImportError:
    pass

try:
    from .ocrqa import ocrqa_pipeline
except ImportError:
    pass
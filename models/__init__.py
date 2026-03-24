"""
Model definitions for the Fake Image Detection project.
"""

from .cnn_model import build_cnn_model
from .hybrid_model import build_hybrid_model

__all__ = [
    "build_cnn_model",
    "build_hybrid_model",
]

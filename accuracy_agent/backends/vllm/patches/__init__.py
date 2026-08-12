"""Patch content for vLLM source files"""

from .weight_filter_patch import get_weight_filter_patch
from .layer_init_patch import get_layer_init_patch

__all__ = ["get_weight_filter_patch", "get_layer_init_patch"]

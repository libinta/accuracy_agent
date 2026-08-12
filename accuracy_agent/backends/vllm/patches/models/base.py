"""Base class for model-specific patch providers"""

from abc import ABC, abstractmethod
from typing import Dict, Optional


class ModelPatchProvider(ABC):
    """
    Base class for model-specific patch providers.

    Provides patches for layer extraction in vLLM models.
    Each model family may have different layer naming patterns and structures.
    """

    @abstractmethod
    def get_layer_pattern(self) -> str:
        r"""
        Return regex pattern for matching layer weight names.

        Returns:
            Regex pattern string that captures layer index in group 1
            Example: r'layers\.(\d+)\.' for standard Llama-style models
            Example: r'transformer\.encoder\.layers\.(\d+)\.' for transformer-style
        """
        pass

    @abstractmethod
    def get_model_class_name(self) -> str:
        """
        Return the vLLM model class name (e.g., 'LlamaModel', 'Glm4Model').

        Returns:
            Model class name string
        """
        pass

    @abstractmethod
    def get_model_file_path(self) -> str:
        """
        Return the relative path to model file in vLLM.

        Returns:
            Path relative to vllm root, e.g., 'vllm/model_executor/models/llama.py'
        """
        pass

    @abstractmethod
    def get_weight_filter_patch(self) -> str:
        """
        Return weight filter patch code.

        Returns:
            Python code string to inject into default_loader.py
        """
        pass

    @abstractmethod
    def get_layer_init_patch(self) -> str:
        """
        Return layer initialization patch code.

        Returns:
            Python code string to inject into model file
        """
        pass

    @abstractmethod
    def get_forward_patch(self) -> str:
        """
        Return forward pass patch code.

        Returns:
            Python code string to modify forward pass
        """
        pass

    @abstractmethod
    def get_anchor_points(self) -> Dict[str, str]:
        """
        Return anchor points for patch insertion.

        Returns:
            Dictionary with keys:
            - 'weight_filter': Anchor line in default_loader.py
            - 'layer_init': Anchor line in model file for layer initialization
            - 'forward': Anchor line in model file for forward pass
        """
        pass

    def get_layer_list_creation_anchor(self) -> Optional[str]:
        """
        Return anchor for layer list creation (optional).

        Returns:
            Line pattern to find where layers ModuleList is created,
            or None if model uses make_layers() function
        """
        return None

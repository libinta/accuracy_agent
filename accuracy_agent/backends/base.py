"""Abstract backend interface for model inference"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class BackendConfig:
    """Configuration for a backend instance"""
    host: str
    docker: str
    vllm_path: str
    cards: str
    device_type: str
    user: str = "root"
    ssh_key_path: str = None


class Backend(ABC):
    """Abstract backend for running layer-wise model inference"""

    def __init__(self, config: BackendConfig, model_path: str, shared_fs: str):
        self.config = config
        self.model_path = model_path
        self.shared_fs = shared_fs

    @abstractmethod
    def setup(self) -> None:
        """
        Initialize backend on remote host.
        For vLLM: applies patches to enable layer extraction.
        For PyTorch: generates test harness.
        """
        pass

    @abstractmethod
    def run_layer_range(
        self,
        layer_start: int,
        layer_end: int,
        prompt: str
    ) -> "torch.Tensor":
        """
        Run layers [layer_start, layer_end) on the given prompt.
        Returns hidden states at layer_end.

        Memory strategy:
        - If full model fits: load all, extract at layer_end (fast)
        - If too large: load only [layer_start, layer_end) (memory-efficient)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up backend state.
        For vLLM: restore original files.
        For PyTorch: remove temporary scripts.
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend dependencies are available on remote host"""
        pass

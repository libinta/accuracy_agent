"""Backend factory for creating backend instances"""

import logging
from typing import TYPE_CHECKING

from .base import Backend, BackendConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_backend(
    backend_type: str,
    config: BackendConfig,
    model_path: str,
    shared_fs: str
) -> Backend:
    """
    Create backend instance based on type.

    Args:
        backend_type: Backend type ("vllm", "pytorch", "sglang")
        config: Backend configuration
        model_path: Path to model directory
        shared_fs: Shared filesystem path

    Returns:
        Backend instance

    Raises:
        ValueError: If backend_type is unknown
    """
    if backend_type == "vllm":
        from .vllm.backend import VLLMBackend
        logger.info(f"Creating vLLM backend for {config.device_type}")
        return VLLMBackend(config, model_path, shared_fs)

    elif backend_type == "pytorch":
        # TODO: Implement PyTorch backend
        raise NotImplementedError("PyTorch backend not yet refactored to new interface")

    elif backend_type == "sglang":
        # TODO: Implement SGLang backend
        raise NotImplementedError("SGLang backend not yet implemented")

    else:
        raise ValueError(
            f"Unknown backend type: {backend_type}. "
            f"Supported: vllm, pytorch, sglang"
        )

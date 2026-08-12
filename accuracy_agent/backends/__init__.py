"""Backend abstraction for different inference engines"""

from .base import Backend, BackendConfig
from .factory import create_backend

__all__ = ["Backend", "BackendConfig", "create_backend"]

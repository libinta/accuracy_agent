"""vLLM backend for layer extraction via source patching"""

__all__ = ["VLLMBackend"]

def __getattr__(name):
    """Lazy import to avoid loading dependencies that may not be installed"""
    if name == "VLLMBackend":
        from .backend import VLLMBackend
        return VLLMBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

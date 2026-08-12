import pytest
from pathlib import Path
from accuracy_agent.backends.vllm.memory_check import estimate_memory_gb, MemoryStatus

def test_estimate_memory_gb_fp8():
    """Test memory estimation for FP8 model"""
    # Mock config for GLM-5.2-FP8: 25B params, 48 layers
    config = {
        "num_parameters": 25_000_000_000,
        "torch_dtype": "float8_e4m3fn",
        "max_position_embeddings": 8192,
        "num_hidden_layers": 48,
        "hidden_size": 4096,
        "num_key_value_heads": 32,
        "num_attention_heads": 32,
    }

    required_gb = estimate_memory_gb(config)

    # FP8 model: ~25GB weights + ~2GB KV cache + ~5GB overhead ≈ 32GB
    assert 28.0 < required_gb < 36.0, f"Expected ~32GB, got {required_gb}GB"

def test_memory_status_full_mode():
    """Test full mode when enough memory"""
    status = MemoryStatus.from_estimates(
        required_gb=32.0,
        available_gb=80.0
    )
    assert status.mode == "full"
    assert status.has_enough_memory()

def test_memory_status_partial_mode():
    """Test partial mode when insufficient memory"""
    status = MemoryStatus.from_estimates(
        required_gb=32.0,
        available_gb=20.0
    )
    assert status.mode == "partial"
    assert not status.has_enough_memory()

"""End-to-end integration test for vLLM backend"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.bisector import Bisector
from accuracy_agent.backends.vllm.memory_check import MemoryStatus


@pytest.fixture
def mock_config():
    """Create test config"""
    return DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        gpu_user="root",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        xpu_user="root",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3,
        test_prompt="Hello world"
    )


@pytest.fixture
def mock_model_info():
    """Create test model info"""
    return ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )


def test_e2e_vllm_backend_matching_layers(mock_config, mock_model_info):
    """Test end-to-end flow with matching GPU/XPU outputs"""

    # Create bisector
    bisector = Bisector(mock_config, mock_model_info)

    # Mock memory check to return full mode for both
    mock_gpu_status = MemoryStatus(required_gb=32.0, available_gb=80.0, mode="full")
    mock_xpu_status = MemoryStatus(required_gb=32.0, available_gb=80.0, mode="full")

    # Mock hidden states (same for matching case)
    hidden_states = torch.randn(10, 4096)

    with patch('accuracy_agent.backends.vllm.backend.check_memory') as mock_check:
        with patch('accuracy_agent.backends.vllm.backend.VLLMPatcher') as mock_patcher_class:
            # Setup mock patcher
            mock_patcher = MagicMock()
            mock_patcher_class.return_value = mock_patcher

            # Memory check returns appropriate status
            mock_check.side_effect = [mock_gpu_status, mock_xpu_status]

            # Mock backend execution to return same hidden states
            with patch('accuracy_agent.backends.vllm.backend.torch.load', return_value=hidden_states):
                with patch('accuracy_agent.backends.vllm.backend.Path.exists', return_value=True):
                    # Run bisection
                    result = bisector.bisect_layers(0, 3)

                    # Verify result
                    assert result.divergent_layer is None
                    assert "match" in result.report.lower()
                    assert len(result.comparison_results) > 0
                    assert result.comparison_results[0].match


def test_e2e_vllm_backend_divergent_layers(mock_config, mock_model_info):
    """Test end-to-end flow with divergent GPU/XPU outputs"""

    bisector = Bisector(mock_config, mock_model_info)

    # Mock memory check
    mock_gpu_status = MemoryStatus(required_gb=32.0, available_gb=80.0, mode="full")
    mock_xpu_status = MemoryStatus(required_gb=32.0, available_gb=20.0, mode="partial")

    # Different hidden states for divergence
    gpu_hidden = torch.randn(10, 4096)
    xpu_hidden = torch.randn(10, 4096) * 0.5  # Different values

    with patch('accuracy_agent.backends.vllm.backend.check_memory') as mock_check:
        with patch('accuracy_agent.backends.vllm.backend.VLLMPatcher') as mock_patcher_class:
            mock_patcher = MagicMock()
            mock_patcher_class.return_value = mock_patcher

            mock_check.side_effect = [mock_gpu_status, mock_xpu_status, mock_gpu_status, mock_xpu_status]

            # Alternate between GPU and XPU outputs
            load_call_count = 0
            def mock_load(path):
                nonlocal load_call_count
                load_call_count += 1
                return gpu_hidden if load_call_count % 2 == 1 else xpu_hidden

            with patch('accuracy_agent.backends.vllm.backend.torch.load', side_effect=mock_load):
                with patch('accuracy_agent.backends.vllm.backend.Path.exists', return_value=True):
                    # Run bisection on single layer
                    result = bisector.bisect_layers(0, 1)

                    # Verify divergence detected
                    assert result.divergent_layer == 0
                    assert len(result.comparison_results) > 0
                    assert not result.comparison_results[0].match


def test_e2e_mixed_memory_modes(mock_config, mock_model_info):
    """Test GPU full mode + XPU partial mode"""

    bisector = Bisector(mock_config, mock_model_info)

    # GPU has enough memory, XPU doesn't
    mock_gpu_status = MemoryStatus(required_gb=32.0, available_gb=80.0, mode="full")
    mock_xpu_status = MemoryStatus(required_gb=32.0, available_gb=20.0, mode="partial")

    hidden_states = torch.randn(10, 4096)

    with patch('accuracy_agent.backends.vllm.backend.check_memory') as mock_check:
        with patch('accuracy_agent.backends.vllm.backend.VLLMPatcher') as mock_patcher_class:
            mock_patcher = MagicMock()
            mock_patcher_class.return_value = mock_patcher

            mock_check.side_effect = [mock_gpu_status, mock_xpu_status]

            with patch('accuracy_agent.backends.vllm.backend.torch.load', return_value=hidden_states):
                with patch('accuracy_agent.backends.vllm.backend.Path.exists', return_value=True):
                    result = bisector.bisect_layers(0, 3)

                    # Verify both backends were setup
                    assert bisector.gpu_backend.memory_mode == "full"
                    assert bisector.xpu_backend.memory_mode == "partial"
                    assert result.divergent_layer is None  # Matching outputs

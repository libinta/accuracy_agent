import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from unittest.mock import Mock, patch
from accuracy_agent.bisector import Bisector
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

def test_parallel_backend_setup():
    """Test parallel GPU/XPU backend setup"""
    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Mock backend setup to avoid actual SSH
    with patch('accuracy_agent.bisector.create_backend') as mock_create:
        mock_gpu_backend = Mock()
        mock_xpu_backend = Mock()
        mock_create.side_effect = [mock_gpu_backend, mock_xpu_backend]

        gpu_backend, xpu_backend = bisector._parallel_setup()

        assert gpu_backend == mock_gpu_backend
        assert xpu_backend == mock_xpu_backend
        # Verify setup was called on both
        mock_gpu_backend.setup.assert_called_once()
        mock_xpu_backend.setup.assert_called_once()

        # Verify create_backend was called with correct arguments
        assert mock_create.call_count == 2
        call_args_list = mock_create.call_args_list

        # Verify GPU backend call
        gpu_call_args = call_args_list[0][0]
        assert gpu_call_args[0] == "vllm"  # backend type
        assert gpu_call_args[1].device_type == "cuda"
        assert gpu_call_args[1].host == "gpu-host"
        assert gpu_call_args[2] == "/mnt/weka/model"  # model_path
        assert gpu_call_args[3] == "/mnt/weka"  # shared_fs

        # Verify XPU backend call
        xpu_call_args = call_args_list[1][0]
        assert xpu_call_args[0] == "vllm"  # backend type
        assert xpu_call_args[1].device_type == "xpu"
        assert xpu_call_args[1].host == "xpu-host"
        assert xpu_call_args[2] == "/mnt/weka/model"  # model_path
        assert xpu_call_args[3] == "/mnt/weka"  # shared_fs

def test_parallel_setup_gpu_fails():
    """Test that GPU backend setup failure is handled properly"""
    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Mock backend setup to test GPU failure
    with patch('accuracy_agent.bisector.create_backend') as mock_create:
        mock_gpu_backend = Mock()
        mock_gpu_backend.setup.side_effect = RuntimeError("GPU connection failed")
        mock_create.return_value = mock_gpu_backend

        # Verify that setup failure is propagated
        try:
            bisector._parallel_setup()
            raise AssertionError("Expected RuntimeError but no error was raised")
        except RuntimeError as e:
            if "GPU connection failed" not in str(e):
                raise AssertionError(f"Expected 'GPU connection failed' in error, got: {e}")
            # Success - error was raised as expected


def test_parallel_layer_execution():
    """Test parallel GPU/XPU layer execution"""
    import torch

    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Hello world"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Mock backends
    gpu_hidden = torch.randn(10, 4096)
    xpu_hidden = torch.randn(10, 4096)

    bisector.gpu_backend = Mock()
    bisector.xpu_backend = Mock()
    bisector.gpu_backend.run_layer_range.return_value = gpu_hidden
    bisector.xpu_backend.run_layer_range.return_value = xpu_hidden

    result = bisector._test_layer_range_parallel(0, 3)

    # Verify both backends called with same params
    bisector.gpu_backend.run_layer_range.assert_called_once_with(0, 3, "Hello world")
    bisector.xpu_backend.run_layer_range.assert_called_once_with(0, 3, "Hello world")

    # Verify comparison result returned
    assert hasattr(result, 'match')
    assert hasattr(result, 'cosine_similarity')


if __name__ == "__main__":
    test_parallel_backend_setup()
    test_parallel_setup_gpu_fails()
    test_parallel_layer_execution()
    print("All tests passed!")

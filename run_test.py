#!/usr/bin/env python3
"""Simple test runner for bisector parallel setup tests"""
import sys
sys.path.insert(0, '.')

from unittest.mock import Mock, patch
from accuracy_agent.bisector import Bisector
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo

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

        assert gpu_backend == mock_gpu_backend, "GPU backend not returned correctly"
        assert xpu_backend == mock_xpu_backend, "XPU backend not returned correctly"
        # Verify setup was called on both
        mock_gpu_backend.setup.assert_called_once()
        mock_xpu_backend.setup.assert_called_once()

        print("✓ All assertions passed!")
        return True

if __name__ == "__main__":
    try:
        test_parallel_backend_setup()
        print("\nTest PASSED: test_parallel_backend_setup")
    except Exception as e:
        print(f"\nTest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

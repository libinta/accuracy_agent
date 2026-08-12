#!/usr/bin/env python3
"""Test runner for bisector parallel setup tests"""
import sys
sys.path.insert(0, '.')

from unittest.mock import Mock, patch, MagicMock
import typing

# Create proper mock modules structure
class FakeTensor:
    pass

class FakeTorch:
    Tensor = FakeTensor
    class nn:
        class functional:
            pass
    class cuda:
        pass

# Mock required modules before importing
torch_mock = FakeTorch()
sys.modules['torch'] = torch_mock
sys.modules['torch.nn'] = torch_mock.nn
sys.modules['torch.nn.functional'] = torch_mock.nn.functional
sys.modules['torch.cuda'] = torch_mock.cuda
sys.modules['yaml'] = MagicMock()
sys.modules['paramiko'] = MagicMock()
sys.modules['click'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['safetensors'] = MagicMock()
sys.modules['rich'] = MagicMock()
sys.modules['rich.console'] = MagicMock()

# Now we can import
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

        assert gpu_backend == mock_gpu_backend, "GPU backend mismatch"
        assert xpu_backend == mock_xpu_backend, "XPU backend mismatch"
        # Verify setup was called on both
        mock_gpu_backend.setup.assert_called_once()
        mock_xpu_backend.setup.assert_called_once()

        # Verify create_backend was called with correct arguments
        assert mock_create.call_count == 2, f"Expected 2 calls, got {mock_create.call_count}"
        call_args_list = mock_create.call_args_list

        # Verify GPU backend call
        gpu_call_args = call_args_list[0][0]
        assert gpu_call_args[0] == "vllm", f"Expected backend type 'vllm', got {gpu_call_args[0]}"
        assert gpu_call_args[1].device_type == "cuda", f"Expected device_type 'cuda', got {gpu_call_args[1].device_type}"
        assert gpu_call_args[1].host == "gpu-host", f"Expected host 'gpu-host', got {gpu_call_args[1].host}"
        assert gpu_call_args[2] == "/mnt/weka/model", f"Expected model_path '/mnt/weka/model', got {gpu_call_args[2]}"
        assert gpu_call_args[3] == "/mnt/weka", f"Expected shared_fs '/mnt/weka', got {gpu_call_args[3]}"

        # Verify XPU backend call
        xpu_call_args = call_args_list[1][0]
        assert xpu_call_args[0] == "vllm", f"Expected backend type 'vllm', got {xpu_call_args[0]}"
        assert xpu_call_args[1].device_type == "xpu", f"Expected device_type 'xpu', got {xpu_call_args[1].device_type}"
        assert xpu_call_args[1].host == "xpu-host", f"Expected host 'xpu-host', got {xpu_call_args[1].host}"
        assert xpu_call_args[2] == "/mnt/weka/model", f"Expected model_path '/mnt/weka/model', got {xpu_call_args[2]}"
        assert xpu_call_args[3] == "/mnt/weka", f"Expected shared_fs '/mnt/weka', got {xpu_call_args[3]}"

        print("✓ test_parallel_backend_setup PASSED")
        return True

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
            print("✓ test_parallel_setup_gpu_fails PASSED")
            return True

if __name__ == "__main__":
    try:
        print("Running tests...")
        test_parallel_backend_setup()
        test_parallel_setup_gpu_fails()
        print("\n✓✓✓ All tests PASSED ✓✓✓")
    except Exception as e:
        print(f"\n✗✗✗ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

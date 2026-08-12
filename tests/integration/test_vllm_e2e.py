"""End-to-end integration test for vLLM backend"""

import pytest
from unittest.mock import patch, MagicMock
import torch

from accuracy_agent.config import DebugConfig
from accuracy_agent.backends import create_backend
from accuracy_agent.backends.base import BackendConfig


@pytest.fixture
def vllm_config(tmp_path) -> DebugConfig:
    """Create test vLLM config"""
    # Create mock model and output paths
    model_path = tmp_path / "model"
    model_path.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    return DebugConfig(
        backend="vllm",
        model_path=str(model_path),
        gpu_host="test-gpu.com",
        gpu_user="test_user",
        gpu_docker="gpu_container",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0,1",
        xpu_host="test-xpu.com",
        xpu_docker="xpu_container",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0,1,2,3",
        shared_fs=str(tmp_path),
        output_dir=str(output_dir),
        layer_start=0,
        layer_end=3,
        test_prompt="Test prompt"
    )


@patch('accuracy_agent.backends.vllm.patcher.VLLMPatcher')
@patch('torch.load')
def test_vllm_backend_end_to_end(mock_torch_load, mock_patcher_class, vllm_config, tmp_path) -> None:
    """Test full workflow: setup -> run -> cleanup"""
    # Setup mocks
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("Success", "")
    mock_patcher_class.return_value = mock_patcher

    # Mock hidden states output
    mock_hidden_states = torch.randn(1, 10, 768)

    # Create fake output file that run_layer_range expects
    output_file = tmp_path / "output" / "hidden_states_0_3.pt"
    output_file.parent.mkdir(exist_ok=True, parents=True)
    torch.save(mock_hidden_states, output_file)
    mock_torch_load.return_value = mock_hidden_states

    # Create GPU backend
    gpu_backend_config = BackendConfig(
        host=vllm_config.gpu_host,
        user=vllm_config.gpu_user,
        docker=vllm_config.gpu_docker,
        vllm_path=vllm_config.gpu_vllm_path,
        cards=vllm_config.gpu_cards,
        device_type="cuda"
    )

    gpu_backend = create_backend(
        vllm_config.backend,
        gpu_backend_config,
        vllm_config.model_path,
        vllm_config.shared_fs
    )

    # Verify initial state
    assert gpu_backend.is_patched is False

    # Setup
    gpu_backend.setup()
    assert gpu_backend.is_patched is True
    mock_patcher.connect.assert_called_once()
    mock_patcher.apply_all_patches.assert_called_once()

    # Run layer extraction
    hidden_states = gpu_backend.run_layer_range(
        vllm_config.layer_start,
        vllm_config.layer_end,
        vllm_config.test_prompt
    )

    assert isinstance(hidden_states, torch.Tensor)
    assert hidden_states.shape == mock_hidden_states.shape

    # Verify debug_runner was called
    mock_patcher.exec_in_docker.assert_called()

    # Cleanup
    gpu_backend.cleanup()
    assert gpu_backend.is_patched is False
    mock_patcher.cleanup.assert_called_once()
    mock_patcher.disconnect.assert_called_once()


@patch('accuracy_agent.backends.vllm.patcher.VLLMPatcher')
def test_vllm_backend_error_handling(mock_patcher_class, vllm_config) -> None:
    """Test error handling in backend operations"""
    # Setup mock to fail
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("", "Error: Command failed")
    mock_patcher_class.return_value = mock_patcher

    gpu_backend_config = BackendConfig(
        host=vllm_config.gpu_host,
        user=vllm_config.gpu_user,
        docker=vllm_config.gpu_docker,
        vllm_path=vllm_config.gpu_vllm_path,
        cards=vllm_config.gpu_cards,
        device_type="cuda"
    )

    gpu_backend = create_backend(
        vllm_config.backend,
        gpu_backend_config,
        vllm_config.model_path,
        vllm_config.shared_fs
    )

    gpu_backend.setup()
    gpu_backend.is_patched = True  # Pretend patched

    try:
        # Should raise error when execution fails
        with pytest.raises(RuntimeError, match="vLLM execution failed"):
            gpu_backend.run_layer_range(0, 3, "Test prompt")
    finally:
        gpu_backend.cleanup()

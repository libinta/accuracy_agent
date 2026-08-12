import pytest
from unittest.mock import Mock, patch
from accuracy_agent.backends.vllm.backend import VLLMBackend
from accuracy_agent.backends.vllm.memory_check import MemoryStatus
from accuracy_agent.backends.base import BackendConfig


def test_backend_setup_with_memory_check():
    """Test backend setup performs memory check"""
    config = BackendConfig(
        host="test-host",
        docker="test-docker",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda",
        user="root"
    )

    backend = VLLMBackend(config, "/mnt/weka/model", "/mnt/weka")

    # Mock memory check to return full mode
    mock_status = MemoryStatus(required_gb=32.0, available_gb=80.0, mode="full")

    with patch('accuracy_agent.backends.vllm.backend.check_memory', return_value=mock_status):
        with patch.object(backend.patcher, 'connect'):
            with patch.object(backend.patcher, 'apply_all_patches'):
                backend.setup()

                assert backend.memory_mode == "full"
                assert backend.is_patched


def test_backend_partial_mode_when_low_memory():
    """Test backend uses partial mode when memory insufficient"""
    config = BackendConfig(
        host="test-host",
        docker="test-docker",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    backend = VLLMBackend(config, "/mnt/weka/model", "/mnt/weka")

    # Mock memory check to return partial mode
    mock_status = MemoryStatus(required_gb=32.0, available_gb=20.0, mode="partial")

    with patch('accuracy_agent.backends.vllm.backend.check_memory', return_value=mock_status):
        with patch.object(backend.patcher, 'connect'):
            with patch.object(backend.patcher, 'apply_all_patches'):
                backend.setup()

                assert backend.memory_mode == "partial"

import pytest
from unittest.mock import Mock, patch, MagicMock
import torch
from pathlib import Path

from accuracy_agent.backends.base import BackendConfig
from accuracy_agent.backends.vllm.backend import VLLMBackend


@pytest.fixture
def backend_config():
    """Create test backend config"""
    return BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0,1",
        device_type="cuda",
        user="testuser"
    )


def test_vllm_backend_init(backend_config):
    """Test VLLMBackend initialization"""
    backend = VLLMBackend(
        config=backend_config,
        model_path="/model/path",
        shared_fs="/shared/fs"
    )

    assert backend.config == backend_config
    assert backend.model_path == "/model/path"
    assert backend.shared_fs == "/shared/fs"
    assert backend.is_patched is False
    assert backend.patcher is not None


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_setup(mock_patcher_class, backend_config):
    """Test backend setup calls patcher correctly"""
    mock_patcher = MagicMock()
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")
    backend.setup()

    assert backend.is_patched is True
    mock_patcher.connect.assert_called_once()
    mock_patcher.apply_all_patches.assert_called_once()


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_cleanup(mock_patcher_class, backend_config):
    """Test backend cleanup restores files"""
    mock_patcher = MagicMock()
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")
    backend.is_patched = True

    backend.cleanup()

    assert backend.is_patched is False
    mock_patcher.cleanup.assert_called_once()
    mock_patcher.disconnect.assert_called_once()


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_is_available(mock_patcher_class, backend_config):
    """Test checking vLLM availability"""
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("exists\n", "")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")
    backend.patcher.ssh_client = MagicMock()  # Pretend connected

    assert backend.is_available() is True

    # Test when vLLM not found
    mock_patcher.exec_in_docker.return_value = ("", "")
    assert backend.is_available() is False


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
@patch('torch.load')
def test_vllm_backend_run_layer_range(mock_torch_load, mock_patcher_class,
                                      backend_config, tmp_path):
    """Test running layer range extraction"""
    # Setup mocks
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("Success", "")
    mock_patcher_class.return_value = mock_patcher

    # Mock tensor output
    mock_tensor = torch.randn(1, 10, 768)
    mock_torch_load.return_value = mock_tensor

    # Use tmp_path as shared_fs
    backend = VLLMBackend(backend_config, "/model/path", str(tmp_path))
    backend.is_patched = True
    backend.patcher.ssh_client = MagicMock()  # Pretend connected

    # Create fake output file
    output_file = tmp_path / "hidden_states_0_3.pt"
    torch.save(mock_tensor, output_file)

    result = backend.run_layer_range(0, 3, "Test prompt")

    assert isinstance(result, torch.Tensor)
    assert result.shape == mock_tensor.shape
    mock_patcher.exec_in_docker.assert_called_once()

    # Verify command was constructed correctly
    call_args = mock_patcher.exec_in_docker.call_args[0][0]
    assert "debug_runner" in call_args
    assert "--layer-start 0" in call_args
    assert "--layer-end 3" in call_args
    assert "--prompt 'Test prompt'" in call_args


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_run_layer_range_not_setup(mock_patcher_class, backend_config):
    """Test that run_layer_range raises error when setup() not called"""
    mock_patcher = MagicMock()
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")
    # Don't call setup(), is_patched should be False

    with pytest.raises(RuntimeError, match="Backend not setup"):
        backend.run_layer_range(0, 3, "Test prompt")


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
@patch('torch.load')
def test_vllm_backend_run_layer_range_stderr_error(mock_torch_load, mock_patcher_class,
                                                    backend_config, tmp_path):
    """Test that run_layer_range raises error when stderr contains error"""
    mock_patcher = MagicMock()
    # Simulate command returning stderr with error
    mock_patcher.exec_in_docker.return_value = ("", "Error: CUDA out of memory")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", str(tmp_path))
    backend.is_patched = True
    backend.patcher.ssh_client = MagicMock()

    with pytest.raises(RuntimeError, match="vLLM execution failed"):
        backend.run_layer_range(0, 3, "Test prompt")


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_run_layer_range_case_insensitive_error(mock_patcher_class,
                                                              backend_config, tmp_path):
    """Test that error detection is case-insensitive"""
    mock_patcher = MagicMock()
    # Simulate lowercase 'error' in stderr
    mock_patcher.exec_in_docker.return_value = ("", "error: some problem")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", str(tmp_path))
    backend.is_patched = True
    backend.patcher.ssh_client = MagicMock()

    with pytest.raises(RuntimeError, match="vLLM execution failed"):
        backend.run_layer_range(0, 3, "Test prompt")


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_run_layer_range_missing_output_file(mock_patcher_class,
                                                           backend_config, tmp_path):
    """Test that run_layer_range raises error when output file not created"""
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("Success", "")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", str(tmp_path))
    backend.is_patched = True
    backend.patcher.ssh_client = MagicMock()

    # Don't create output file - it should raise error
    with pytest.raises(RuntimeError, match="Output file not created"):
        backend.run_layer_range(0, 3, "Test prompt")


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
@patch('torch.load')
def test_vllm_backend_prompt_escaping(mock_torch_load, mock_patcher_class,
                                      backend_config, tmp_path):
    """Test that prompts with single quotes are properly escaped"""
    mock_patcher = MagicMock()
    mock_patcher.exec_in_docker.return_value = ("Success", "")
    mock_patcher_class.return_value = mock_patcher

    mock_tensor = torch.randn(1, 10, 768)
    mock_torch_load.return_value = mock_tensor

    backend = VLLMBackend(backend_config, "/model/path", str(tmp_path))
    backend.is_patched = True
    backend.patcher.ssh_client = MagicMock()

    # Create fake output file
    output_file = tmp_path / "hidden_states_0_3.pt"
    torch.save(mock_tensor, output_file)

    # Test prompt with single quote
    result = backend.run_layer_range(0, 3, "What's Paris?")

    # Verify command has escaped single quote
    call_args = mock_patcher.exec_in_docker.call_args[0][0]
    # Single quote should be escaped as '\''
    assert "What'\\''s Paris?" in call_args


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_setup_failure(mock_patcher_class, backend_config):
    """Test that setup() handles failures properly"""
    mock_patcher = MagicMock()
    # Simulate apply_all_patches failing
    mock_patcher.apply_all_patches.side_effect = Exception("Connection failed")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")

    with pytest.raises(RuntimeError, match="Failed to apply vLLM patches"):
        backend.setup()

    # Verify is_patched is False after failure
    assert backend.is_patched is False


@patch('accuracy_agent.backends.vllm.backend.VLLMPatcher')
def test_vllm_backend_is_available_connection_failure(mock_patcher_class, backend_config):
    """Test that is_available() handles connection failures"""
    mock_patcher = MagicMock()
    mock_patcher.ssh_client = None
    # Simulate connection failure
    mock_patcher.connect.side_effect = Exception("SSH connection refused")
    mock_patcher_class.return_value = mock_patcher

    backend = VLLMBackend(backend_config, "/model/path", "/shared/fs")

    with pytest.raises(RuntimeError, match="Failed to connect or check vLLM installation"):
        backend.is_available()

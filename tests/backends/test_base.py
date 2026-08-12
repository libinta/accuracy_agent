import pytest
from accuracy_agent.backends.base import BackendConfig, Backend
from accuracy_agent.backends import create_backend


def test_backend_config_creation():
    """Test BackendConfig dataclass initialization"""
    config = BackendConfig(
        host="test-host.com",
        user="testuser",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0,1",
        device_type="cuda"
    )

    assert config.host == "test-host.com"
    assert config.user == "testuser"
    assert config.docker == "test_container"
    assert config.vllm_path == "/workspace/vllm"
    assert config.cards == "0,1"
    assert config.device_type == "cuda"


def test_backend_config_defaults():
    """Test BackendConfig default values"""
    config = BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    assert config.user == "root"  # Default user


def test_backend_cannot_instantiate_directly():
    """Test that Backend abstract class cannot be instantiated"""
    config = BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        Backend(config, "/model/path", "/shared/fs")


def test_backend_subclass_requires_all_methods():
    """Test that Backend subclass must implement all abstract methods"""
    config = BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    class IncompleteBackend(Backend):
        def setup(self):
            pass
        # Missing: run_layer_range, cleanup, is_available

    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        IncompleteBackend(config, "/model/path", "/shared/fs")


def test_create_backend_vllm():
    """Test backend factory creates vLLM backend"""
    config = BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    backend = create_backend("vllm", config, "/model/path", "/shared/fs")

    from accuracy_agent.backends.vllm import VLLMBackend
    assert isinstance(backend, VLLMBackend)


def test_create_backend_invalid():
    """Test backend factory rejects invalid backend type"""
    config = BackendConfig(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    with pytest.raises(ValueError, match="Unknown backend type"):
        create_backend("invalid", config, "/model/path", "/shared/fs")

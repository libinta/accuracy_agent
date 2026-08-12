import pytest
from accuracy_agent.backends.factory import create_backend
from accuracy_agent.backends.base import BackendConfig
from accuracy_agent.backends.vllm.backend import VLLMBackend

def test_create_vllm_backend():
    """Test vLLM backend creation"""
    config = BackendConfig(
        host="test-host",
        docker="test-docker",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda",
        user="root"
    )

    backend = create_backend(
        backend_type="vllm",
        config=config,
        model_path="/mnt/weka/model",
        shared_fs="/mnt/weka"
    )

    assert isinstance(backend, VLLMBackend)
    assert backend.config.host == "test-host"

def test_create_unknown_backend():
    """Test error on unknown backend type"""
    config = BackendConfig(
        host="test-host",
        docker="test-docker",
        vllm_path="/workspace/vllm",
        cards="0",
        device_type="cuda"
    )

    with pytest.raises(ValueError, match="Unknown backend"):
        create_backend(
            backend_type="unknown",
            config=config,
            model_path="/mnt/weka/model",
            shared_fs="/mnt/weka"
        )

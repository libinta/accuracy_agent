import pytest
from pathlib import Path
from accuracy_agent.config import DebugConfig

def test_debug_config_valid():
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/accuracy_debug_output",
        layer_start=0,
        layer_end=3
    )
    assert config.model_path == "/mnt/weka/model"
    assert config.layer_start < config.layer_end

def test_debug_config_invalid_layer_range():
    with pytest.raises(ValueError, match="layer_start must be < layer_end"):
        DebugConfig(
            model_path="/mnt/weka/model",
            gpu_host="gpu.example.com",
            gpu_docker="gpu_container",
            xpu_host="xpu.example.com",
            xpu_docker="xpu_container",
            shared_fs="/mnt/weka",
            output_dir="/mnt/weka/output",
            layer_start=5,
            layer_end=3
        )


def test_debug_config_with_backend_fields():
    """Test DebugConfig with new backend fields"""
    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host.com",
        gpu_docker="gpu_container",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0,1",
        xpu_host="xpu-host.com",
        xpu_docker="xpu_container",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0,1,2,3",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/accuracy_debug_output",
        layer_start=0,
        layer_end=3,
    )

    assert config.backend == "vllm"
    assert config.gpu_vllm_path == "/workspace/vllm"
    assert config.xpu_vllm_path == "/workspace/vllm"


def test_debug_config_from_yaml(tmp_path):
    """Test loading DebugConfig from YAML"""
    yaml_content = """backend: vllm

model:
  path: /mnt/weka/model

gpu:
  host: gpu-host.com
  docker: gpu_container
  vllm_path: /workspace/vllm
  cards: "0,1"

xpu:
  host: xpu-host.com
  docker: xpu_container
  vllm_path: /workspace/vllm
  cards: "0,1,2,3"

shared_fs: /mnt/weka
output_dir: /mnt/weka/accuracy_debug_output

test:
  layer_start: 0
  layer_end: 3
  prompt: "Test prompt"
"""

    yaml_file = tmp_path / "test_config.yaml"
    yaml_file.write_text(yaml_content)

    config = DebugConfig.from_yaml(str(yaml_file))

    assert config.backend == "vllm"
    assert config.model_path == "/mnt/weka/model"
    assert config.gpu_vllm_path == "/workspace/vllm"
    assert config.xpu_vllm_path == "/workspace/vllm"
    assert config.test_prompt == "Test prompt"

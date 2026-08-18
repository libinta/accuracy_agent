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


def test_debug_config_gpu_image_defaults():
    """GPU docker automation is on by default, with nothing else preset."""
    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        xpu_host="localhost",
        xpu_docker="xpu_container",
    )
    assert config.gpu_auto_image is True
    assert config.gpu_image == ""
    assert config.gpu_docker == ""
    assert config.gpu_inside_container is None


def test_debug_config_from_yaml_gpu_image_fields(tmp_path):
    """gpu.image / auto_image / container_name / docker_run_args round-trip"""
    yaml_content = """backend: vllm

model:
  path: /mnt/weka/model

gpu:
  image: vllm/vllm-openai:v0.11.0
  auto_image: false
  container_name: my_gpu_container
  docker_run_args: "--network=host"
  cards: "0"

xpu:
  host: localhost
  docker: xpu_container
  inside_container: true

shared_fs: /mnt/weka
output_dir: /mnt/weka/accuracy_debug_output
"""

    yaml_file = tmp_path / "auto_gpu_config.yaml"
    yaml_file.write_text(yaml_content)

    config = DebugConfig.from_yaml(str(yaml_file))

    assert config.gpu_image == "vllm/vllm-openai:v0.11.0"
    assert config.gpu_auto_image is False
    assert config.gpu_container_name == "my_gpu_container"
    assert config.gpu_docker_run_args == "--network=host"
    assert config.xpu_inside_container is True
    assert config.gpu_inside_container is None  # not specified -> auto-detect


def test_debug_config_vllm_commit_defaults():
    """Building peers from a commit is opt-in, and off by default."""
    config = DebugConfig(model_path="/mnt/weka/model")

    assert config.vllm_commit == ""
    assert config.vllm_repo_path == ""
    assert config.vllm_build_root == ""
    assert config.vllm_build_kernels is False
    assert config.vllm_build_rebuild is False
    assert config.gpu_base_image == ""
    assert config.xpu_base_image == ""
    assert config.xpu_image == ""


def test_debug_config_from_yaml_vllm_commit_fields(tmp_path):
    yaml_content = """
backend: vllm

model:
  path: /mnt/weka/model

vllm:
  commit: 7794b1e08bf505ff28664515ffaaeeec955ab796
  repo_path: /home/me/vllm
  build_root: /mnt/weka/accuracy_agent_builds
  build_kernels: true
  rebuild: true

gpu:
  base_image: nvcr.io/nvidia/pytorch:26.07-py3

xpu:
  base_image: intel/intel-extension-for-pytorch:2.8.10-xpu
  container_name: my_xpu_peer
  docker_run_args: "--group-add 110"

shared_fs: /mnt/weka
output_dir: /mnt/weka/accuracy_debug_output
"""

    yaml_file = tmp_path / "commit_config.yaml"
    yaml_file.write_text(yaml_content)

    config = DebugConfig.from_yaml(str(yaml_file))

    assert config.vllm_commit == "7794b1e08bf505ff28664515ffaaeeec955ab796"
    assert config.vllm_repo_path == "/home/me/vllm"
    assert config.vllm_build_root == "/mnt/weka/accuracy_agent_builds"
    assert config.vllm_build_kernels is True
    assert config.vllm_build_rebuild is True
    assert config.gpu_base_image == "nvcr.io/nvidia/pytorch:26.07-py3"
    assert config.xpu_base_image == "intel/intel-extension-for-pytorch:2.8.10-xpu"
    assert config.xpu_container_name == "my_xpu_peer"
    assert config.xpu_docker_run_args == "--group-add 110"

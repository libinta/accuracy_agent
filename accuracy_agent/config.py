from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class DebugConfig:
    """Configuration for accuracy debugging session."""

    # Backend selection
    backend: str = "pytorch"  # "vllm", "pytorch", or "sglang"

    # Model and paths
    model_path: str = ""
    shared_fs: str = "/mnt/weka"
    output_dir: str = "/mnt/weka/accuracy_debug"

    # Remote hosts
    gpu_host: str = ""
    gpu_docker: str = ""
    xpu_host: str = ""
    xpu_docker: str = ""

    # GPU vLLM configuration
    gpu_vllm_path: str = "/workspace/vllm"
    gpu_user: str = "youruser"
    gpu_cards: str = "0"
    gpu_ssh_key_path: Optional[str] = None

    # GPU docker image handling. When gpu_docker is empty and the XPU docker is
    # reachable locally, gpu_auto_image lets the tool read the exact vLLM COMMIT
    # out of the XPU container and build that commit into the NVIDIA PyTorch base
    # image itself (see vllm_source_builder.autoconfigure_gpu_from_xpu_commit),
    # so the GPU side needs no configuration and runs the same vLLM code.
    gpu_image: str = ""              # explicit image ref; skips detection and building
    gpu_auto_image: bool = True      # derive+launch the GPU docker automatically
    gpu_container_name: str = ""     # name for the auto-launched container
    gpu_docker_run_args: str = ""    # extra raw `docker run` flags
    # None = auto-detect whether we run inside the target container; set to False
    # for an auto-launched GPU container, which is always a separate container.
    gpu_inside_container: Optional[bool] = None

    # XPU vLLM configuration
    xpu_inside_container: Optional[bool] = None  # None = auto-detect
    xpu_vllm_path: str = "/workspace/vllm"
    xpu_user: str = "root"
    xpu_cards: str = "0"
    xpu_ssh_key_path: Optional[str] = None
    xpu_image: str = ""              # set when the XPU peer was built from a commit
    xpu_container_name: str = ""
    xpu_docker_run_args: str = ""

    # Build both peers from one vllm-project/vllm commit (see vllm_source_builder).
    # Set vllm_commit to compare a KNOWN commit: it is installed from source into
    # the vendor PyTorch base images below, so both sides run identical vLLM code.
    # Left empty, the commit is instead read out of the existing XPU container and
    # only the GPU peer is built from it; the settings below apply to that build
    # too, and vllm_commit is filled in with the commit that was detected.
    vllm_commit: str = ""            # sha / tag / branch in vllm-project/vllm
    vllm_repo_path: str = ""         # local clone to resolve it in (default: ~/vllm)
    vllm_build_root: str = ""        # where per-commit checkouts go
    vllm_build_kernels: bool = False  # CUDA: compile kernels instead of using
                                      # the precompiled nightly wheel (1-2 h)
    vllm_build_rebuild: bool = False  # ignore cached images and install again
    gpu_base_image: str = ""         # default: newest nvcr.io/nvidia/pytorch
    xpu_base_image: str = ""         # default: newest intel/intel-extension-for-pytorch *-xpu

    # Test scope
    # layer_select: "auto" (default) tests one representative per UNIQUE layer
    # type the model has (see model_loader.layer_groups) -- e.g. a dense and a
    # MoE layer for GLM-MoE. "range" tests the explicit [layer_start, layer_end)
    # sweep instead (manual override).
    layer_select: str = "auto"
    layer_start: int = 0
    layer_end: int = 3
    test_prompt: str = "What is the capital of France?"

    # SSH settings
    ssh_user: Optional[str] = None  # Default to current user
    ssh_key_path: Optional[str] = None  # Default to ~/.ssh/id_rsa

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DebugConfig":
        """Load configuration from YAML file"""
        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        config_dict = {
            "backend": data.get("backend", "pytorch"),
            "model_path": data.get("model", {}).get("path", ""),

            "gpu_host": data.get("gpu", {}).get("host", ""),
            "gpu_user": data.get("gpu", {}).get("user", "youruser"),
            "gpu_docker": data.get("gpu", {}).get("docker", ""),
            "gpu_vllm_path": data.get("gpu", {}).get("vllm_path", "/workspace/vllm"),
            "gpu_cards": data.get("gpu", {}).get("cards", "0"),
            "gpu_ssh_key_path": data.get("gpu", {}).get("ssh_key_path"),
            "gpu_image": data.get("gpu", {}).get("image", ""),
            "gpu_auto_image": data.get("gpu", {}).get("auto_image", True),
            "gpu_container_name": data.get("gpu", {}).get("container_name", ""),
            "gpu_docker_run_args": data.get("gpu", {}).get("docker_run_args", ""),
            "gpu_inside_container": data.get("gpu", {}).get("inside_container"),
            "gpu_base_image": data.get("gpu", {}).get("base_image", ""),

            "xpu_host": data.get("xpu", {}).get("host", ""),
            "xpu_user": data.get("xpu", {}).get("user", "root"),
            "xpu_docker": data.get("xpu", {}).get("docker", ""),
            "xpu_vllm_path": data.get("xpu", {}).get("vllm_path", "/workspace/vllm"),
            "xpu_cards": data.get("xpu", {}).get("cards", "0"),
            "xpu_ssh_key_path": data.get("xpu", {}).get("ssh_key_path"),
            "xpu_inside_container": data.get("xpu", {}).get("inside_container"),
            "xpu_base_image": data.get("xpu", {}).get("base_image", ""),
            "xpu_image": data.get("xpu", {}).get("image", ""),
            "xpu_container_name": data.get("xpu", {}).get("container_name", ""),
            "xpu_docker_run_args": data.get("xpu", {}).get("docker_run_args", ""),

            "vllm_commit": data.get("vllm", {}).get("commit", ""),
            "vllm_repo_path": data.get("vllm", {}).get("repo_path", ""),
            "vllm_build_root": data.get("vllm", {}).get("build_root", ""),
            "vllm_build_kernels": data.get("vllm", {}).get("build_kernels", False),
            "vllm_build_rebuild": data.get("vllm", {}).get("rebuild", False),

            "shared_fs": data.get("shared_fs", "/mnt/weka"),
            "output_dir": data.get("output_dir", "/mnt/weka/accuracy_debug"),

            "layer_select": data.get("test", {}).get("select", "auto"),
            "layer_start": data.get("test", {}).get("layer_start", 0),
            "layer_end": data.get("test", {}).get("layer_end", 3),
            "test_prompt": data.get("test", {}).get("prompt", "What is the capital of France?"),

            "ssh_user": data.get("ssh_user"),
            "ssh_key_path": data.get("ssh_key_path"),
        }

        return cls(**config_dict)

    def __post_init__(self):
        """Call validate() to ensure configuration is valid."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration using Path.relative_to() logic."""
        from pathlib import Path

        if self.layer_start >= self.layer_end:
            raise ValueError(f"layer_start must be < layer_end, got {self.layer_start} >= {self.layer_end}")

        # Validate model_path is on shared filesystem
        if self.model_path:
            model_path = Path(self.model_path)
            shared_fs = Path(self.shared_fs)
            try:
                model_path.relative_to(shared_fs)
            except ValueError:
                raise ValueError(
                    f"model_path must be on shared filesystem: "
                    f"{self.model_path} not under {self.shared_fs}"
                )

        # Validate output_dir is on shared filesystem (if not empty)
        if self.output_dir:
            output_dir = Path(self.output_dir)
            shared_fs = Path(self.shared_fs)
            try:
                output_dir.relative_to(shared_fs)
            except ValueError:
                raise ValueError(
                    f"output_dir must be on shared filesystem: "
                    f"{self.output_dir} not under {self.shared_fs}"
                )

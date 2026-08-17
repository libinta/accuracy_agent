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
    dut_host: str = ""
    dut_docker: str = ""

    # GPU vLLM configuration
    gpu_vllm_path: str = "/workspace/vllm"
    gpu_user: str = "youruser"
    gpu_cards: str = "0"
    gpu_ssh_key_path: Optional[str] = None

    # Device-under-test (DUT) vLLM configuration
    dut_vllm_path: str = "/workspace/vllm"
    dut_user: str = "root"
    dut_cards: str = "0"
    dut_ssh_key_path: Optional[str] = None
    # Device type for the device-under-test slot: "cuda" (NVIDIA GPU),
    # "xpu" (Intel GPU), or "hpu" (Gaudi).
    dut_device_type: str = "xpu"

    # Test scope
    # layer_select: "auto" (default) tests one representative per UNIQUE layer
    # type the model has (see model_loader.layer_groups) -- e.g. a dense and a
    # MoE layer for GLM-MoE. "range" tests the explicit [layer_start, layer_end)
    # sweep instead (manual override).
    layer_select: str = "auto"
    layer_start: int = 0
    layer_end: int = 3
    test_prompt: str = "What is the capital of France?"

    # Comparison thresholds (a hidden-state MATCH needs cos >= cos_threshold AND
    # max rel error <= rel_threshold). bf16 cross-hardware hidden states differ
    # by ~1e-2 rel error / ~1e-4 cos distance even when numerically equivalent,
    # so the comparator's strict library defaults (rel 1e-4, cos 0.999) flag a
    # true MATCH as DIVERGE and trigger needless per-layer bisection -- which
    # crashes on HPU hybrid sub-windows that lack a full-attention layer. These
    # bf16-realistic defaults are overridable via test.cos_threshold /
    # test.rel_threshold in the yaml.
    cos_threshold: float = 0.99
    rel_threshold: float = 5e-2

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

            "dut_host": data.get("dut", {}).get("host", ""),
            "dut_user": data.get("dut", {}).get("user", "root"),
            "dut_docker": data.get("dut", {}).get("docker", ""),
            "dut_vllm_path": data.get("dut", {}).get("vllm_path", "/workspace/vllm"),
            "dut_cards": data.get("dut", {}).get("cards", "0"),
            "dut_ssh_key_path": data.get("dut", {}).get("ssh_key_path"),
            "dut_device_type": data.get("dut", {}).get("device_type", "xpu"),

            "shared_fs": data.get("shared_fs", "/mnt/weka"),
            "output_dir": data.get("output_dir", "/mnt/weka/accuracy_debug"),

            "layer_select": data.get("test", {}).get("select", "auto"),
            "layer_start": data.get("test", {}).get("layer_start", 0),
            "layer_end": data.get("test", {}).get("layer_end", 3),
            "test_prompt": data.get("test", {}).get("prompt", "What is the capital of France?"),
            "cos_threshold": data.get("test", {}).get("cos_threshold", 0.99),
            "rel_threshold": data.get("test", {}).get("rel_threshold", 5e-2),

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

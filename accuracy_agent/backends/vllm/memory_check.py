"""Memory estimation and checking for vLLM models"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryStatus:
    """Memory check result"""
    required_gb: float
    available_gb: float
    mode: str  # "full" or "partial"

    @classmethod
    def from_estimates(cls, required_gb: float, available_gb: float) -> "MemoryStatus":
        """Create MemoryStatus with mode decision"""
        # 10% safety margin
        mode = "full" if available_gb >= required_gb * 1.1 else "partial"
        return cls(
            required_gb=required_gb,
            available_gb=available_gb,
            mode=mode
        )

    def has_enough_memory(self) -> bool:
        """Check if enough memory for full model"""
        return self.mode == "full"


def estimate_memory_gb(config: Dict[str, Any]) -> float:
    """
    Estimate model memory requirement from config.json.

    Args:
        config: Model config dict with architecture parameters

    Returns:
        Estimated memory in GB

    Raises:
        ValueError: If required config fields are missing
    """
    # Validate required config fields
    required_fields = ["num_hidden_layers", "num_attention_heads", "hidden_size"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Config missing required fields: {missing}")

    # Model weights
    num_params = config.get("num_parameters")
    if not num_params:
        # Estimate from architecture if num_parameters not in config
        num_params = _estimate_params_from_architecture(config)

    dtype = config.get("torch_dtype", "float16")
    dtype_bytes = {
        "float16": 2,
        "bfloat16": 2,
        "float8_e4m3fn": 1,
        "float8_e5m2": 1,
        "float32": 4,
    }.get(dtype, 2)  # Default to FP16

    model_weights_gb = (num_params * dtype_bytes) / (1024**3)

    # KV cache estimate
    max_len = config.get("max_position_embeddings", 8192)
    num_layers = config["num_hidden_layers"]
    num_kv_heads = config.get("num_key_value_heads", config["num_attention_heads"])
    num_attn_heads = config["num_attention_heads"]
    hidden_size = config["hidden_size"]
    head_dim = hidden_size // num_attn_heads

    # KV cache: K and V, each layer, per token
    kv_size_per_token = 2 * num_layers * num_kv_heads * head_dim * dtype_bytes
    kv_cache_gb = (max_len * kv_size_per_token) / (1024**3)

    # Overhead: activations, fragmentation (20%)
    overhead_gb = (model_weights_gb + kv_cache_gb) * 0.2

    total_gb = model_weights_gb + kv_cache_gb + overhead_gb

    logger.info(
        f"Memory estimate: weights={model_weights_gb:.2f}GB, "
        f"kv_cache={kv_cache_gb:.2f}GB, overhead={overhead_gb:.2f}GB, "
        f"total={total_gb:.2f}GB"
    )

    return total_gb


def _estimate_params_from_architecture(config: Dict[str, Any]) -> int:
    """
    Rough parameter estimation from architecture config.

    Formula: params ≈ 12 * num_layers * hidden_size^2 (for transformer models)
    """
    num_layers = config["num_hidden_layers"]
    hidden_size = config["hidden_size"]

    # Transformer formula (rough approximation)
    params = 12 * num_layers * (hidden_size ** 2)

    logger.warning(
        f"num_parameters not in config, estimated {params:,} from architecture "
        f"(may be inaccurate for non-standard architectures)"
    )

    return params


def load_model_config(model_path: str) -> Dict[str, Any]:
    """Load config.json from model path"""
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def query_available_memory(host: str, docker: str, device_type: str, patcher) -> float:
    """
    Query available device memory via SSH.

    Args:
        host: SSH host
        docker: Docker container name
        device_type: "cuda" or "xpu"
        patcher: VLLMPatcher instance with SSH connection

    Returns:
        Available memory in GB
    """
    if device_type == "cuda":
        cmd = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1"
    elif device_type == "xpu":
        # XPU memory query (get first available card)
        cmd = "xpu-smi dump -m 2>/dev/null | grep 'GPU Memory Available' | head -1 | awk '{print $4}'"
    else:
        raise ValueError(f"Unknown device type: {device_type}")

    try:
        stdout, stderr = patcher.exec_in_docker(cmd)

        if not stdout.strip():
            logger.warning(f"No memory info returned for {device_type}, defaulting to partial mode")
            return 0.0  # Conservative: assume no memory available

        # Parse memory value with format validation
        try:
            memory_mb = float(stdout.strip())
        except ValueError:
            logger.error(
                f"Failed to parse {device_type} memory output: '{stdout.strip()}' "
                f"(expected numeric value in MB)"
            )
            return 0.0  # Conservative: assume no memory available

        # Validate memory value is in expected range (MB units)
        if not (0 < memory_mb < 1_000_000):
            raise ValueError(
                f"Invalid memory value {memory_mb}MB for {device_type}: "
                f"expected range 0-1000000 MB"
            )

        memory_gb = memory_mb / 1024

        logger.info(f"Available {device_type} memory: {memory_gb:.2f}GB")
        return memory_gb

    except Exception as e:
        logger.error(f"Failed to query {device_type} memory: {e}")
        return 0.0  # Conservative: assume no memory available


def check_memory(
    model_path: str,
    host: str,
    docker: str,
    device_type: str,
    patcher,
    cards: str = "0"
) -> MemoryStatus:
    """
    Check if model fits in device memory.

    Args:
        model_path: Path to model directory with config.json
        host: SSH host
        docker: Docker container
        device_type: "cuda" or "xpu"
        patcher: VLLMPatcher instance with SSH connection
        cards: GPU/XPU card specification (e.g., "0", "0,1,2,3" for multi-GPU)

    Returns:
        MemoryStatus with mode decision
    """
    logger.info(f"Checking memory for {model_path} on {device_type}")

    # Load model config and estimate requirement
    config = load_model_config(model_path)
    required_gb = estimate_memory_gb(config)

    # Parse card count from cards string (e.g., "0,1,2,3" -> 4 cards, "0" -> 1 card)
    card_count = len(cards.split(',')) if cards else 1
    logger.info(f"Using {card_count} card(s) for tensor parallelism")

    # Divide required memory by card count for distributed inference
    required_gb_per_card = required_gb / card_count

    # Query available memory (queries a single card)
    available_gb = query_available_memory(host, docker, device_type, patcher)

    # Create status with mode decision using per-card requirement
    status = MemoryStatus.from_estimates(required_gb_per_card, available_gb)

    logger.info(
        f"Memory check: required_total={required_gb:.2f}GB, "
        f"required_per_card={required_gb_per_card:.2f}GB, "
        f"available_per_card={available_gb:.2f}GB, "
        f"cards={card_count}, mode={status.mode}"
    )

    return status

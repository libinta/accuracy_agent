import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass
class ModelInfo:
    """Model architecture information."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    layer_type: str  # "standard" or "sliding_window"
    sliding_window: Optional[int] = None
    sliding_window_pattern: Optional[List[int]] = None

def load_model_info(model_path: str) -> ModelInfo:
    """Load model architecture info from config.json.

    Args:
        model_path: Path to model directory containing config.json

    Returns:
        ModelInfo with architecture details

    Raises:
        FileNotFoundError: If config.json doesn't exist
        ValueError: If config is missing required fields
    """
    config_path = Path(model_path) / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    # Required fields
    try:
        num_layers = config["num_hidden_layers"]
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
    except KeyError as e:
        raise ValueError(f"Config missing required field: {e}")

    # Detect layer type
    has_sliding_window = "sliding_window" in config
    layer_type = "sliding_window" if has_sliding_window else "standard"

    return ModelInfo(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        layer_type=layer_type,
        sliding_window=config.get("sliding_window"),
        sliding_window_pattern=config.get("sliding_window_pattern")
    )

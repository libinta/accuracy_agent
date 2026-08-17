import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

@dataclass
class ModelInfo:
    """Model architecture information."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    layer_type: str  # "standard" or "sliding_window"
    sliding_window: Optional[int] = None
    sliding_window_pattern: Optional[List[int]] = None
    # Representative layers to test: one (type_name, layer_index) per UNIQUE
    # layer type the model has. The index is always the FIRST occurrence of
    # that type, so the debug window [0, index+1) stays as small as possible
    # (deeper occurrences would force building most of the model, which does
    # not fit on a single card). Testing one representative per type is the
    # "set of unique layers" the design calls for -- e.g. a MoE model gets a
    # dense representative and a MoE representative rather than a trivial 0-N
    # range that only exercises the dense prefix.
    layer_groups: List[Tuple[str, int]] = field(default_factory=list)


def _compute_layer_groups(config: dict, num_layers: int) -> List[Tuple[str, int]]:
    """Derive representative unique layers from a model's config.json.

    Returns a list of (type_name, first_occurrence_index). See ModelInfo.
    """
    # --- MoE lineage (DeepSeek / GLM-MoE): a dense prefix then MoE layers. ---
    # `first_k_dense_replace` = number of leading dense layers; the rest are MoE.
    n_experts = config.get("n_routed_experts") or config.get("num_experts")
    first_k_dense = config.get("first_k_dense_replace")
    if n_experts and first_k_dense is not None:
        groups: List[Tuple[str, int]] = []
        if first_k_dense > 0:
            groups.append(("dense", 0))
        if first_k_dense < num_layers:
            groups.append(("moe", min(first_k_dense, num_layers - 1)))
        return groups or [("standard", 0)]

    # --- Qwen3-MoE lineage: num_experts present, dense layers selected by
    # mlp_only_layers / decoder_sparse_step (no first_k_dense_replace like the
    # DeepSeek/GLM family). Pick the first dense and first MoE layer as the two
    # unique representatives. ---
    qwen_experts = config.get("num_experts")
    if qwen_experts and first_k_dense is None and (
        "decoder_sparse_step" in config or "mlp_only_layers" in config
    ):
        mlp_only = set(config.get("mlp_only_layers") or [])
        sparse_step = config.get("decoder_sparse_step", 1) or 1

        def _is_moe(idx: int) -> bool:
            if idx in mlp_only:
                return False
            return (idx + 1) % sparse_step == 0

        dense_idx = next((i for i in range(num_layers) if not _is_moe(i)), None)
        moe_idx = next((i for i in range(num_layers) if _is_moe(i)), None)
        groups: List[Tuple[str, int]] = []
        if dense_idx is not None:
            groups.append(("dense", dense_idx))
        if moe_idx is not None:
            groups.append(("moe", moe_idx))
        return groups or [("standard", 0)]

    # --- Explicit per-layer type list (e.g. Qwen3.5/3.6 hybrid): the config
    # carries a `layer_types` array naming each decoder layer's attention kind
    # ("linear_attention" / "full_attention", interleaved every
    # full_attention_interval). This family has NO first_k_dense_replace /
    # decoder_sparse_step / mlp_only_layers (every layer is MoE), so the earlier
    # branches don't fire; the UNIQUE axis here is the attention type. Pick the
    # first occurrence of each distinct type, preserving order. This also makes
    # the debug window reach the first full_attention layer, which is required
    # for the hybrid KV-cache group indexing to build a full-attention group
    # (a linear-only prefix trips _get_attention_group_id_for_hybrid). ---
    layer_types = config.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        seen = set()
        groups = []
        for idx, ltype in enumerate(layer_types[:num_layers]):
            if ltype not in seen:
                seen.add(ltype)
                groups.append((str(ltype), idx))
        return groups or [("standard", 0)]

    # --- Sliding-window hybrid (e.g. Gemma): one full pattern cycle. ---
    # sliding_window_pattern like [5, 1] = 5 sliding + 1 full. Test the first
    # sliding layer and the first full layer (the two unique attention types).
    pattern = config.get("sliding_window_pattern")
    if isinstance(pattern, list) and len(pattern) >= 2 and "sliding_window" in config:
        sliding_count = pattern[0]
        groups = [("sliding_window", 0)]
        if sliding_count < num_layers:
            groups.append(("full_attention", min(sliding_count, num_layers - 1)))
        return groups

    # --- Fallback: a homogeneous model has one representative type. ---
    return [("standard", 0)]

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

    # Multimodal configs (e.g. Qwen3.5/3.6 VL/MoE) nest the language-model
    # fields under "text_config". Merge them up so the loader (and the
    # layer-group heuristics) can read them transparently.
    if isinstance(config.get("text_config"), dict):
        config = {**config, **config["text_config"]}

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
        sliding_window_pattern=config.get("sliding_window_pattern"),
        layer_groups=_compute_layer_groups(config, num_layers),
    )

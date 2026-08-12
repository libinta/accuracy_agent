import json
import tempfile
import os
from pathlib import Path
from accuracy_agent.model_loader import load_model_info, ModelInfo

def test_load_model_info_standard():
    """Test parsing standard transformer config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }
        config_path.write_text(json.dumps(config))

        info = load_model_info(tmpdir)
        assert info.num_layers == 32
        assert info.hidden_size == 4096
        assert info.layer_type == "standard"

def test_load_model_info_sliding_window():
    """Test parsing Gemma4-style sliding window config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config = {
            "model_type": "gemma2",
            "num_hidden_layers": 42,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "sliding_window": 1024,
            "sliding_window_pattern": [5, 1]
        }
        config_path.write_text(json.dumps(config))

        info = load_model_info(tmpdir)
        assert info.num_layers == 42
        assert info.layer_type == "sliding_window"
        assert info.sliding_window == 1024
        assert info.sliding_window_pattern == [5, 1]


def _write_config(tmpdir, config):
    (Path(tmpdir) / "config.json").write_text(json.dumps(config))


def test_layer_groups_glm_moe():
    """GLM-MoE: 3 dense layers then MoE -> dense@0 and moe@3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, {
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 78,
            "hidden_size": 6144,
            "num_attention_heads": 96,
            "first_k_dense_replace": 3,
            "n_routed_experts": 256,
        })
        info = load_model_info(tmpdir)
        assert info.layer_groups == [("dense", 0), ("moe", 3)]


def test_layer_groups_moe_no_dense_prefix():
    """MoE model with no dense prefix -> only a moe representative at 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, {
            "model_type": "deepseek_v2",
            "num_hidden_layers": 60,
            "hidden_size": 5120,
            "num_attention_heads": 128,
            "first_k_dense_replace": 0,
            "n_routed_experts": 160,
        })
        info = load_model_info(tmpdir)
        assert info.layer_groups == [("moe", 0)]


def test_layer_groups_sliding_window_pattern():
    """Gemma-style [5,1] pattern -> first sliding layer and first full layer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, {
            "model_type": "gemma2",
            "num_hidden_layers": 42,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "sliding_window": 1024,
            "sliding_window_pattern": [5, 1],
        })
        info = load_model_info(tmpdir)
        assert info.layer_groups == [("sliding_window", 0), ("full_attention", 5)]


def test_layer_groups_standard_fallback():
    """A homogeneous model has a single representative type."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(tmpdir, {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32,
        })
        info = load_model_info(tmpdir)
        assert info.layer_groups == [("standard", 0)]

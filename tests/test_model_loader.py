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

import tempfile
from pathlib import Path
from accuracy_agent.test_harness_generator import generate_test_harness, save_test_harness
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo

def test_generate_gpu_test_harness():
    """Test generating GPU test harness script."""
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )

    model_info = ModelInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="standard"
    )

    script = generate_test_harness(
        config=config,
        model_info=model_info,
        layer_start=0,
        layer_end=3,
        platform="gpu"
    )

    # Verify script contains expected elements
    assert "import torch" in script
    assert "device = 'cuda'" in script or 'DEVICE = "cuda"' in script
    assert "layer_start = 0" in script or "LAYER_START = 0" in script
    assert "layer_end = 3" in script or "LAYER_END = 3" in script
    assert "/mnt/weka/model" in script
    assert "torch.save" in script

    # The harness must run ONLY the requested layer subset and save hidden
    # states (not final-model logits).
    assert "embed_tokens" in script
    assert "for i in range(start, end)" in script
    assert '"hidden_states"' in script
    # It must NOT fall back to full-model logits.
    assert "outputs.logits" not in script

    # Generated script must be valid Python.
    compile(script, "<generated_gpu_harness>", "exec")

def test_generate_xpu_test_harness():
    """Test generating XPU test harness script."""
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )

    model_info = ModelInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="standard"
    )

    script = generate_test_harness(
        config=config,
        model_info=model_info,
        layer_start=0,
        layer_end=3,
        platform="xpu"
    )

    assert "device = 'xpu'" in script or 'DEVICE = "xpu"' in script or "device = torch.device('xpu')" in script

def test_save_test_harness():
    """Test saving test harness to file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_content = "#!/usr/bin/env python3\nprint('test')"
        output_path = Path(tmpdir) / "test_harness.py"

        save_test_harness(script_content, str(output_path))

        assert output_path.exists()
        assert output_path.read_text() == script_content
        # Check executable bit
        assert output_path.stat().st_mode & 0o111  # At least one execute bit set

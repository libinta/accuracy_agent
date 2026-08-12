"""Integration tests for end-to-end accuracy debugging workflow."""

import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import patch

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.bisector import Bisector


def test_end_to_end_bisection_mock():
    """End-to-end test with mocked remote execution.

    This validates the full flow without requiring actual GPU/XPU hosts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup config
        config = DebugConfig(
            model_path=f"{tmpdir}/model",
            gpu_host="gpu.example.com",
            gpu_docker="gpu_container",
            xpu_host="xpu.example.com",
            xpu_docker="xpu_container",
            shared_fs=tmpdir,
            output_dir=f"{tmpdir}/output",
            layer_start=0,
            layer_end=3
        )

        model_info = ModelInfo(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layer_type="standard"
        )

        bisector = Bisector(config, model_info)

        # Mock remote executor
        with patch.object(bisector.executor, 'execute_test_script') as mock_exec:
            # Mock successful execution
            def mock_execute(script_path, output_path, platform):
                # Create mock output file
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

                # Layer 0 diverges, layers 1-2 match
                if 'layer_0_1' in output_path or ('layer_0_3' in output_path and '0' in str(Path(output_path).name)):
                    # Divergent output (hidden states, not final logits)
                    data = {
                        "hidden_states": torch.randn(1, 5, 4096) * (2.0 if platform == "xpu" else 1.0),
                        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                        "layer_start": 0,
                        "layer_end": 1,
                        "platform": platform
                    }
                else:
                    # Matching output (same seed)
                    torch.manual_seed(42)
                    data = {
                        "hidden_states": torch.randn(1, 5, 4096),
                        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                        "layer_start": 1,
                        "layer_end": 2,
                        "platform": platform
                    }

                torch.save(data, output_path)

                from accuracy_agent.remote_executor import ExecutionResult
                return ExecutionResult(
                    success=True,
                    stdout="Test completed",
                    stderr="",
                    output_path=output_path
                )

            mock_exec.side_effect = mock_execute

            # Run bisection
            result = bisector.bisect_layers(0, 3)

            # Should find divergence in layer 0
            assert result.divergent_layer == 0
            assert len(result.comparison_results) > 0


def test_integration_readme_example():
    """Verify README example is valid."""
    readme = (Path(__file__).parent.parent / "README.md").read_text()

    # Check that example command is present
    assert "accuracy-debug" in readme
    assert "--model" in readme
    assert "--gpu-host" in readme

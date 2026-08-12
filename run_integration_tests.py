#!/usr/bin/env python3
"""Integration test runner that doesn't require pytest."""
import sys
import traceback
import tempfile
import torch
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

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
            assert result.divergent_layer == 0, f"Expected divergent_layer=0, got {result.divergent_layer}"
            assert len(result.comparison_results) > 0, "Expected at least one comparison result"


def test_integration_readme_example():
    """Verify README example is valid."""
    readme = (Path(__file__).parent.parent / "README.md").read_text()

    # Check that example command is present
    assert "accuracy-debug" in readme, "README should contain 'accuracy-debug' command"
    assert "--model" in readme, "README should contain '--model' flag"
    assert "--gpu-host" in readme, "README should contain '--gpu-host' flag"


def run_tests():
    """Run all integration tests and report results."""
    passed = 0
    failed = 0

    # Test 1: End-to-end bisection with mocked execution
    try:
        test_end_to_end_bisection_mock()
        print("✓ test_end_to_end_bisection_mock PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_end_to_end_bisection_mock FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: Verify README example
    try:
        test_integration_readme_example()
        print("✓ test_integration_readme_example PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_integration_readme_example FAILED")
        traceback.print_exc()
        failed += 1

    print(f"\n{'='*50}")
    print(f"Tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

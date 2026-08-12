#!/usr/bin/env python3
"""Test runner for bisector tests."""
import sys
import traceback
import torch
from unittest.mock import patch, Mock

def run_tests():
    """Run all bisector tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_bisect_finds_divergence_in_layer_0
    try:
        # This will fail with ModuleNotFoundError until we implement bisector.py
        from accuracy_agent.bisector import Bisector, BisectionResult
        from accuracy_agent.config import DebugConfig
        from accuracy_agent.model_loader import ModelInfo
        from accuracy_agent.remote_executor import ExecutionResult
        from accuracy_agent.comparator import ComparisonResult

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

        bisector = Bisector(config, model_info)

        # Mock remote executor to simulate divergence in layer 0
        with patch.object(bisector, 'executor') as mock_executor:
            # Mock execution results
            def mock_execute(script_path, output_path, platform):
                # Simulate outputs saved to files
                return ExecutionResult(
                    success=True,
                    stdout=f"Saved to {output_path}",
                    stderr="",
                    output_path=output_path
                )

            mock_executor.execute_test_script = mock_execute

            # Mock torch.load to return divergent tensors
            with patch('torch.load') as mock_load:
                def mock_load_fn(path):
                    if 'gpu' in path:
                        return {"hidden_states": torch.randn(1, 10, 4096)}
                    else:
                        return {"hidden_states": torch.randn(1, 10, 4096)}  # Different values

                mock_load.side_effect = mock_load_fn

                result = bisector.bisect_layers(layer_start=0, layer_end=3)

                # Should detect divergence
                assert result.divergent_layer is not None, "Should detect divergence"
                assert len(result.comparison_results) > 0, "Should have comparison results"
                print("✓ test_bisect_finds_divergence_in_layer_0 PASSED")
                passed += 1
    except Exception as e:
        print(f"✗ test_bisect_finds_divergence_in_layer_0 FAILED")
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

#!/usr/bin/env python3
"""Mock test for bisector without requiring torch installation."""
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Mock torch before importing anything
sys.modules['torch'] = MagicMock()

# Mock other dependencies
sys.modules['paramiko'] = MagicMock()

# Now we can import
from accuracy_agent.bisector import Bisector, BisectionResult
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.remote_executor import ExecutionResult
from accuracy_agent.comparator import ComparisonResult

def test_bisect_finds_divergence():
    """Test that bisection correctly identifies divergent layer."""
    print("Running test_bisect_finds_divergence...")

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

    # Mock the _test_layer_range method to simulate divergence
    original_test = bisector._test_layer_range

    def mock_test_layer_range(layer_start, layer_end):
        print(f"  Mock testing layers {layer_start}-{layer_end}")
        # Simulate divergence in layer 0
        if layer_start == 0 and layer_end == 1:
            return ComparisonResult(
                match=False,
                cosine_similarity=0.95,
                max_rel_error=0.01,
                max_abs_error=0.5
            )
        # All other ranges match
        return ComparisonResult(
            match=True,
            cosine_similarity=0.9999,
            max_rel_error=1e-6,
            max_abs_error=1e-5
        )

    bisector._test_layer_range = mock_test_layer_range

    # Run bisection
    result = bisector.bisect_layers(layer_start=0, layer_end=3)

    # Verify results
    assert result.divergent_layer == 0, f"Expected divergent_layer=0, got {result.divergent_layer}"
    assert len(result.comparison_results) > 0, "Should have comparison results"
    assert "layer 0" in result.report.lower(), f"Report should mention layer 0, got: {result.report}"

    print("✓ test_bisect_finds_divergence PASSED")
    return True

def test_bisect_no_divergence():
    """Test that bisection handles no divergence case."""
    print("Running test_bisect_no_divergence...")

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

    # Mock the _test_layer_range method to simulate no divergence
    def mock_test_layer_range(layer_start, layer_end):
        print(f"  Mock testing layers {layer_start}-{layer_end}")
        # All ranges match
        return ComparisonResult(
            match=True,
            cosine_similarity=0.9999,
            max_rel_error=1e-6,
            max_abs_error=1e-5
        )

    bisector._test_layer_range = mock_test_layer_range

    # Run bisection
    result = bisector.bisect_layers(layer_start=0, layer_end=3)

    # Verify results
    assert result.divergent_layer is None, f"Expected no divergence, got layer {result.divergent_layer}"
    assert len(result.comparison_results) > 0, "Should have comparison results"
    assert "match" in result.report.lower(), f"Report should indicate match, got: {result.report}"

    print("✓ test_bisect_no_divergence PASSED")
    return True

def test_bisection_result_dataclass():
    """Test BisectionResult dataclass."""
    print("Running test_bisection_result_dataclass...")

    result = BisectionResult(
        divergent_layer=5,
        comparison_results=[],
        report="Test report"
    )

    assert result.divergent_layer == 5
    assert result.comparison_results == []
    assert result.report == "Test report"

    print("✓ test_bisection_result_dataclass PASSED")
    return True

if __name__ == "__main__":
    passed = 0
    failed = 0

    tests = [
        test_bisection_result_dataclass,
        test_bisect_finds_divergence,
        test_bisect_no_divergence,
    ]

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

    sys.exit(0 if failed == 0 else 1)

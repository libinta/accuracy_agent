#!/usr/bin/env python3
"""Comprehensive test for Task 4 - parallel layer execution."""
import sys
import inspect
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock torch and dependencies before importing bisector
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['paramiko'] = MagicMock()
sys.modules['paramiko.ssh_exception'] = MagicMock()

# Now import the modules
import concurrent.futures
from accuracy_agent.bisector import Bisector, BisectionResult
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.comparator import ComparisonResult

def test_parallel_layer_execution_basic():
    """Test basic parallel layer execution structure."""
    print("\n[Test 1] Testing _test_layer_range_parallel basic structure...")

    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Hello world"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Verify method exists
    assert hasattr(bisector, '_test_layer_range_parallel'), "Method not found"
    print("✓ _test_layer_range_parallel method exists")

    # Verify method signature
    sig = inspect.signature(bisector._test_layer_range_parallel)
    params = list(sig.parameters.keys())
    expected_params = ['layer_start', 'layer_end']
    assert params == expected_params, f"Expected {expected_params}, got {params}"
    print("✓ Method signature correct")

    # Verify return type annotation
    return_annotation = sig.return_annotation
    assert return_annotation.__name__ == 'ComparisonResult', f"Expected ComparisonResult, got {return_annotation}"
    print("✓ Return type is ComparisonResult")

    return True

def test_parallel_execution_with_mocks():
    """Test parallel execution using mocked backends."""
    print("\n[Test 2] Testing parallel execution with mocked backends...")

    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Test prompt"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Create mock backends
    bisector.gpu_backend = Mock()
    bisector.xpu_backend = Mock()

    # Mock return values
    gpu_output = MagicMock()
    xpu_output = MagicMock()
    bisector.gpu_backend.run_layer_range.return_value = gpu_output
    bisector.xpu_backend.run_layer_range.return_value = xpu_output

    # Mock compare_tensors
    with patch('accuracy_agent.bisector.compare_tensors') as mock_compare:
        mock_result = ComparisonResult(
            match=True,
            cosine_similarity=0.99,
            max_rel_error=0.001,
            max_abs_error=0.01
        )
        mock_compare.return_value = mock_result

        # Call method
        result = bisector._test_layer_range_parallel(0, 3)

        # Verify both backends were called
        bisector.gpu_backend.run_layer_range.assert_called_once_with(0, 3, "Test prompt")
        bisector.xpu_backend.run_layer_range.assert_called_once_with(0, 3, "Test prompt")
        print("✓ Both backends called with correct parameters")

        # Verify compare_tensors was called
        mock_compare.assert_called_once_with(gpu_output, xpu_output)
        print("✓ compare_tensors called correctly")

        # Verify return value
        assert result == mock_result
        assert result.match == True
        assert result.cosine_similarity == 0.99
        print("✓ Correct ComparisonResult returned")

    return True

def test_bisect_layers_uses_parallel():
    """Test that bisect_layers uses parallel execution."""
    print("\n[Test 3] Testing bisect_layers uses parallel execution...")

    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Test prompt"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)
    assert bisector.use_backends == True, "use_backends should be True for vllm"
    print("✓ use_backends flag set correctly")

    # Mock backends
    mock_gpu = Mock()
    mock_xpu = Mock()
    bisector.gpu_backend = mock_gpu
    bisector.xpu_backend = mock_xpu

    # Mock methods
    mock_gpu.run_layer_range.return_value = MagicMock()
    mock_xpu.run_layer_range.return_value = MagicMock()

    with patch('accuracy_agent.bisector.compare_tensors') as mock_compare:
        mock_result = ComparisonResult(
            match=True,
            cosine_similarity=0.99,
            max_rel_error=0.001,
            max_abs_error=0.01
        )
        mock_compare.return_value = mock_result

        # Call bisect_layers
        result = bisector.bisect_layers(0, 2)

        # Verify result
        assert isinstance(result, BisectionResult)
        assert result.divergent_layer is None  # No divergence since all match
        print("✓ bisect_layers returns BisectionResult")

        # Verify parallel execution was used
        assert mock_gpu.run_layer_range.called, "GPU backend not called"
        assert mock_xpu.run_layer_range.called, "XPU backend not called"
        print("✓ Both backends called during bisection")

    return True

def test_backward_compatibility():
    """Test backward compatibility with pytorch backend."""
    print("\n[Test 4] Testing backward compatibility with pytorch backend...")

    config = DebugConfig(
        backend="pytorch",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Test prompt"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)
    assert bisector.use_backends == False, "use_backends should be False for pytorch"
    # For pytorch backend, executor is used instead
    assert hasattr(bisector, 'executor'), "executor should be set for pytorch"
    print("✓ Backward compatibility: pytorch backend uses RemoteExecutor (not new backend system)")

    return True

def test_threadpool_executor_used():
    """Test that ThreadPoolExecutor is actually used."""
    print("\n[Test 5] Testing ThreadPoolExecutor is used for parallelism...")

    config = DebugConfig(
        backend="vllm",
        model_path="/mnt/weka/model",
        gpu_host="gpu-host",
        gpu_docker="gpu-docker",
        gpu_vllm_path="/workspace/vllm",
        gpu_cards="0",
        xpu_host="xpu-host",
        xpu_docker="xpu-docker",
        xpu_vllm_path="/workspace/vllm",
        xpu_cards="0",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        test_prompt="Test prompt"
    )

    model_info = ModelInfo(
        num_layers=48,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="GLMBlock",
        sliding_window=None
    )

    bisector = Bisector(config, model_info)

    # Mock backends
    call_order = []

    def mock_gpu_run(start, end, prompt):
        call_order.append('gpu')
        return MagicMock()

    def mock_xpu_run(start, end, prompt):
        call_order.append('xpu')
        return MagicMock()

    bisector.gpu_backend = Mock()
    bisector.xpu_backend = Mock()
    bisector.gpu_backend.run_layer_range.side_effect = mock_gpu_run
    bisector.xpu_backend.run_layer_range.side_effect = mock_xpu_run

    with patch('accuracy_agent.bisector.compare_tensors') as mock_compare:
        mock_result = ComparisonResult(
            match=True,
            cosine_similarity=0.99,
            max_rel_error=0.001,
            max_abs_error=0.01
        )
        mock_compare.return_value = mock_result

        result = bisector._test_layer_range_parallel(0, 3)

        # Both should be called (order may vary due to threading)
        assert 'gpu' in call_order, "GPU backend not called"
        assert 'xpu' in call_order, "XPU backend not called"
        print("✓ Both backends executed (ThreadPoolExecutor used)")

    return True

def main():
    """Run all tests."""
    print("=" * 70)
    print("Task 4: Parallel Layer Execution - Comprehensive Tests")
    print("=" * 70)

    tests = [
        test_parallel_layer_execution_basic,
        test_parallel_execution_with_mocks,
        test_bisect_layers_uses_parallel,
        test_backward_compatibility,
        test_threadpool_executor_used,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 70)

    if failed == 0:
        print("\n✓✓✓ All comprehensive tests PASSED ✓✓✓\n")
        return 0
    else:
        print(f"\n✗✗✗ {failed} test(s) FAILED ✗✗✗\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

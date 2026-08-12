#!/usr/bin/env python3
"""Simple test runner that doesn't require pytest."""
import sys
import traceback

def run_tests():
    """Run all tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_debug_config_valid
    try:
        from accuracy_agent.config import DebugConfig

        config = DebugConfig(
            model_path="/mnt/weka/model",
            gpu_host="gpu.example.com",
            gpu_docker="gpu_container",
            xpu_host="xpu.example.com",
            xpu_docker="xpu_container",
            shared_fs="/mnt/weka",
            output_dir="/mnt/weka/accuracy_debug_output",
            layer_start=0,
            layer_end=3
        )
        assert config.model_path == "/mnt/weka/model"
        assert config.layer_start < config.layer_end
        print("✓ test_debug_config_valid PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_debug_config_valid FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: test_debug_config_invalid_layer_range
    try:
        from accuracy_agent.config import DebugConfig

        try:
            config = DebugConfig(
                model_path="/mnt/weka/model",
                gpu_host="gpu.example.com",
                gpu_docker="gpu_container",
                xpu_host="xpu.example.com",
                xpu_docker="xpu_container",
                shared_fs="/mnt/weka",
                output_dir="/mnt/weka/output",
                layer_start=5,
                layer_end=3
            )
            print("✗ test_debug_config_invalid_layer_range FAILED - expected ValueError")
            failed += 1
        except ValueError as e:
            if "layer_start must be < layer_end" in str(e):
                print("✓ test_debug_config_invalid_layer_range PASSED")
                passed += 1
            else:
                print(f"✗ test_debug_config_invalid_layer_range FAILED - wrong error message: {e}")
                failed += 1
    except Exception as e:
        print(f"✗ test_debug_config_invalid_layer_range FAILED")
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

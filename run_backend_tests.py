#!/usr/bin/env python3
"""Test runner for backend tests."""
import sys
import traceback

def run_tests():
    """Run all backend tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_backend_config_creation
    try:
        from accuracy_agent.backends.base import BackendConfig

        config = BackendConfig(
            host="test-host.com",
            user="testuser",
            docker="test_container",
            vllm_path="/workspace/vllm",
            cards="0,1",
            device_type="cuda"
        )

        assert config.host == "test-host.com"
        assert config.user == "testuser"
        assert config.docker == "test_container"
        assert config.vllm_path == "/workspace/vllm"
        assert config.cards == "0,1"
        assert config.device_type == "cuda"
        print("✓ test_backend_config_creation PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_backend_config_creation FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: test_backend_config_defaults
    try:
        from accuracy_agent.backends.base import BackendConfig

        config = BackendConfig(
            host="test-host.com",
            docker="test_container",
            vllm_path="/workspace/vllm",
            cards="0",
            device_type="cuda"
        )

        assert config.user == "root"  # Default user
        print("✓ test_backend_config_defaults PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_backend_config_defaults FAILED")
        traceback.print_exc()
        failed += 1

    # Test 3: test_backend_cannot_instantiate_directly
    try:
        from accuracy_agent.backends.base import BackendConfig, Backend

        config = BackendConfig(
            host="test-host.com",
            docker="test_container",
            vllm_path="/workspace/vllm",
            cards="0",
            device_type="cuda"
        )

        try:
            Backend(config, "/model/path", "/shared/fs")
            print("✗ test_backend_cannot_instantiate_directly FAILED - expected TypeError")
            failed += 1
        except TypeError as e:
            if "Can't instantiate abstract class" in str(e):
                print("✓ test_backend_cannot_instantiate_directly PASSED")
                passed += 1
            else:
                print(f"✗ test_backend_cannot_instantiate_directly FAILED - wrong error: {e}")
                failed += 1
    except Exception as e:
        print(f"✗ test_backend_cannot_instantiate_directly FAILED")
        traceback.print_exc()
        failed += 1

    # Test 4: test_backend_subclass_requires_all_methods
    try:
        from accuracy_agent.backends.base import BackendConfig, Backend

        config = BackendConfig(
            host="test-host.com",
            docker="test_container",
            vllm_path="/workspace/vllm",
            cards="0",
            device_type="cuda"
        )

        class IncompleteBackend(Backend):
            def setup(self):
                pass
            # Missing: run_layer_range, cleanup, is_available

        try:
            IncompleteBackend(config, "/model/path", "/shared/fs")
            print("✗ test_backend_subclass_requires_all_methods FAILED - expected TypeError")
            failed += 1
        except TypeError as e:
            if "Can't instantiate abstract class" in str(e):
                print("✓ test_backend_subclass_requires_all_methods PASSED")
                passed += 1
            else:
                print(f"✗ test_backend_subclass_requires_all_methods FAILED - wrong error: {e}")
                failed += 1
    except Exception as e:
        print(f"✗ test_backend_subclass_requires_all_methods FAILED")
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

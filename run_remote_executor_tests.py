#!/usr/bin/env python3
"""Test runner for remote executor tests."""
import sys
import traceback
from unittest.mock import Mock, patch

def run_tests():
    """Run all remote executor tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_execute_gpu_script_success
    try:
        from accuracy_agent.remote_executor import RemoteExecutor, ExecutionResult
        from accuracy_agent.config import DebugConfig

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

        executor = RemoteExecutor(config)

        # Mock paramiko module
        with patch('accuracy_agent.remote_executor.paramiko') as mock_paramiko:
            mock_ssh_class = Mock()
            mock_paramiko.SSHClient = mock_ssh_class
            mock_paramiko.AutoAddPolicy = Mock()
            mock_ssh = Mock()
            mock_ssh_class.return_value = mock_ssh

            # Mock successful execution
            mock_stdin = Mock()
            mock_stdout = Mock()
            mock_stdout.read.return_value = b"Test output\n"
            mock_stdout.channel.recv_exit_status.return_value = 0
            mock_stderr = Mock()
            mock_stderr.read.return_value = b""

            mock_ssh.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

            result = executor.execute_test_script(
                script_path="/mnt/weka/test.py",
                output_path="/mnt/weka/output.pt",
                platform="gpu"
            )

            assert result.success is True, f"Expected success=True, got {result.success}"
            assert "Test output" in result.stdout, f"Expected 'Test output' in stdout, got: {result.stdout}"
            assert result.output_path == "/mnt/weka/output.pt", f"Expected output_path=/mnt/weka/output.pt, got {result.output_path}"

        print("✓ test_execute_gpu_script_success PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_execute_gpu_script_success FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: test_execute_xpu_script_failure
    try:
        from accuracy_agent.remote_executor import RemoteExecutor, ExecutionResult
        from accuracy_agent.config import DebugConfig

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

        executor = RemoteExecutor(config)

        # Mock paramiko module
        with patch('accuracy_agent.remote_executor.paramiko') as mock_paramiko:
            mock_ssh_class = Mock()
            mock_paramiko.SSHClient = mock_ssh_class
            mock_paramiko.AutoAddPolicy = Mock()
            mock_ssh = Mock()
            mock_ssh_class.return_value = mock_ssh

            # Mock failed execution
            mock_stdin = Mock()
            mock_stdout = Mock()
            mock_stdout.read.return_value = b""
            mock_stdout.channel.recv_exit_status.return_value = 1
            mock_stderr = Mock()
            mock_stderr.read.return_value = b"Error: CUDA out of memory\n"

            mock_ssh.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)

            result = executor.execute_test_script(
                script_path="/mnt/weka/test.py",
                output_path="/mnt/weka/output.pt",
                platform="xpu"
            )

            assert result.success is False, f"Expected success=False, got {result.success}"
            assert "Error" in result.stderr, f"Expected 'Error' in stderr, got: {result.stderr}"

        print("✓ test_execute_xpu_script_failure PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_execute_xpu_script_failure FAILED")
        traceback.print_exc()
        failed += 1

    # Test 3: test_invalid_platform
    try:
        from accuracy_agent.remote_executor import RemoteExecutor, ExecutionResult
        from accuracy_agent.config import DebugConfig

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

        executor = RemoteExecutor(config)

        try:
            result = executor.execute_test_script(
                script_path="/mnt/weka/test.py",
                output_path="/mnt/weka/output.pt",
                platform="invalid"
            )
            print(f"✗ test_invalid_platform FAILED - expected ValueError")
            failed += 1
        except ValueError as e:
            if "Invalid platform" in str(e):
                print("✓ test_invalid_platform PASSED")
                passed += 1
            else:
                print(f"✗ test_invalid_platform FAILED - wrong error message: {e}")
                failed += 1
    except Exception as e:
        print(f"✗ test_invalid_platform FAILED")
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

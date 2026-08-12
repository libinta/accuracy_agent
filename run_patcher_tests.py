#!/usr/bin/env python3
"""Test runner for VLLMPatcher tests."""
import sys
import traceback
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock paramiko before importing patcher
sys.modules['paramiko'] = MagicMock()


def run_tests():
    """Run all VLLMPatcher tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_patcher_init
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        patcher = VLLMPatcher(
            host="test-host.com",
            docker="test_container",
            vllm_path="/workspace/vllm",
            user="testuser"
        )

        assert patcher.host == "test-host.com"
        assert patcher.docker == "test_container"
        assert patcher.vllm_path == "/workspace/vllm"
        assert patcher.user == "testuser"
        assert patcher.ssh_client is None
        print("✓ test_patcher_init PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_patcher_init FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: test_patcher_connect
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        with patch('paramiko.SSHClient') as mock_client:
            client_instance = MagicMock()
            mock_client.return_value = client_instance

            patcher = VLLMPatcher(
                host="test-host.com",
                docker="test_container",
                vllm_path="/workspace/vllm"
            )

            patcher.connect()

            assert patcher.ssh_client is not None
            client_instance.set_missing_host_key_policy.assert_called_once()
            client_instance.connect.assert_called_once_with("test-host.com", username="root")
            print("✓ test_patcher_connect PASSED")
            passed += 1
    except Exception as e:
        print(f"✗ test_patcher_connect FAILED")
        traceback.print_exc()
        failed += 1

    # Test 3: test_exec_in_docker
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        with patch('paramiko.SSHClient') as mock_client:
            mock_ssh_client = MagicMock()
            mock_client.return_value = mock_ssh_client

            # Setup mock
            mock_stdout = MagicMock()
            mock_stdout.read.return_value = b"command output"
            mock_stderr = MagicMock()
            mock_stderr.read.return_value = b""
            mock_ssh_client.exec_command.return_value = (None, mock_stdout, mock_stderr)

            patcher = VLLMPatcher("test-host.com", "test_container", "/workspace/vllm")
            patcher.ssh_client = mock_ssh_client

            stdout, stderr = patcher.exec_in_docker("ls /workspace")

            assert stdout == "command output"
            assert stderr == ""
            mock_ssh_client.exec_command.assert_called_once_with(
                'docker exec test_container bash -c "ls /workspace"'
            )
            print("✓ test_exec_in_docker PASSED")
            passed += 1
    except Exception as e:
        print(f"✗ test_exec_in_docker FAILED")
        traceback.print_exc()
        failed += 1

    # Test 4: test_backup_file
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        with patch('paramiko.SSHClient') as mock_client:
            mock_ssh_client = MagicMock()
            mock_client.return_value = mock_ssh_client

            mock_ssh_client.exec_command.return_value = (
                None,
                MagicMock(read=lambda: b""),
                MagicMock(read=lambda: b"")
            )

            patcher = VLLMPatcher("test-host.com", "test_container", "/workspace/vllm")
            patcher.ssh_client = mock_ssh_client

            patcher.backup_file("/workspace/vllm/some_file.py")

            # Should execute backup command
            call_args = mock_ssh_client.exec_command.call_args[0][0]
            assert "cp /workspace/vllm/some_file.py" in call_args
            assert "/workspace/vllm/some_file.py.original" in call_args
            print("✓ test_backup_file PASSED")
            passed += 1
    except Exception as e:
        print(f"✗ test_backup_file FAILED")
        traceback.print_exc()
        failed += 1

    # Test 5: test_restore_original
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        with patch('paramiko.SSHClient') as mock_client:
            mock_ssh_client = MagicMock()
            mock_client.return_value = mock_ssh_client

            mock_ssh_client.exec_command.return_value = (
                None,
                MagicMock(read=lambda: b""),
                MagicMock(read=lambda: b"")
            )

            patcher = VLLMPatcher("test-host.com", "test_container", "/workspace/vllm")
            patcher.ssh_client = mock_ssh_client

            patcher.restore_original("/workspace/vllm/some_file.py")

            # Should execute restore command
            call_args = mock_ssh_client.exec_command.call_args[0][0]
            assert "mv /workspace/vllm/some_file.py.original" in call_args
            assert "/workspace/vllm/some_file.py" in call_args
            print("✓ test_restore_original PASSED")
            passed += 1
    except Exception as e:
        print(f"✗ test_restore_original FAILED")
        traceback.print_exc()
        failed += 1

    # Test 6: test_apply_all_patches
    try:
        from accuracy_agent.backends.vllm.patcher import VLLMPatcher

        with patch('paramiko.SSHClient') as mock_client:
            mock_ssh_client = MagicMock()
            mock_client.return_value = mock_ssh_client

            # Mock file operations
            mock_sftp = MagicMock()
            mock_ssh_client.open_sftp.return_value.__enter__.return_value = mock_sftp
            mock_ssh_client.exec_command.return_value = (
                None,
                MagicMock(read=lambda: b"# existing file content"),
                MagicMock(read=lambda: b"")
            )

            patcher = VLLMPatcher("test-host.com", "test_container", "/workspace/vllm")
            patcher.ssh_client = mock_ssh_client

            # Should not raise exception
            patcher.apply_all_patches()

            # Verify files were backed up and patched
            assert mock_ssh_client.exec_command.call_count > 0
            print("✓ test_apply_all_patches PASSED")
            passed += 1
    except Exception as e:
        print(f"✗ test_apply_all_patches FAILED")
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

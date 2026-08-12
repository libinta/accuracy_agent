"""Tests for VLLMPatcher SSH-based patching"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


@pytest.fixture
def mock_ssh_client():
    """Mock paramiko SSH client"""
    with patch('paramiko.SSHClient') as mock_client:
        client_instance = MagicMock()
        mock_client.return_value = client_instance
        yield client_instance


def test_patcher_init():
    """Test VLLMPatcher initialization"""
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


def test_patcher_connect(mock_ssh_client):
    """Test SSH connection establishment"""
    from accuracy_agent.backends.vllm.patcher import VLLMPatcher

    patcher = VLLMPatcher(
        host="test-host.com",
        docker="test_container",
        vllm_path="/workspace/vllm"
    )

    patcher.connect()

    assert patcher.ssh_client is not None
    mock_ssh_client.set_missing_host_key_policy.assert_called_once()
    mock_ssh_client.connect.assert_called_once_with("test-host.com", username="root")


def test_exec_in_docker(mock_ssh_client):
    """Test command execution in docker container"""
    from accuracy_agent.backends.vllm.patcher import VLLMPatcher

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


def test_backup_file(mock_ssh_client):
    """Test file backup creation"""
    from accuracy_agent.backends.vllm.patcher import VLLMPatcher

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


def test_restore_original(mock_ssh_client):
    """Test file restoration from backup"""
    from accuracy_agent.backends.vllm.patcher import VLLMPatcher

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


def test_apply_all_patches(mock_ssh_client):
    """Test applying all patches"""
    from accuracy_agent.backends.vllm.patcher import VLLMPatcher

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

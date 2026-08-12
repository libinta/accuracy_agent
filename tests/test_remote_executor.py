import pytest
from unittest.mock import Mock, patch
from accuracy_agent.remote_executor import RemoteExecutor, ExecutionResult
from accuracy_agent.config import DebugConfig

@pytest.fixture
def config():
    return DebugConfig(
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

def test_execute_gpu_script_success(config):
    """Test successful GPU script execution."""
    executor = RemoteExecutor(config)

    with patch('paramiko.SSHClient') as mock_ssh_class:
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

        assert result.success is True
        assert "Test output" in result.stdout
        assert result.output_path == "/mnt/weka/output.pt"

def test_execute_xpu_script_failure(config):
    """Test XPU script execution failure."""
    executor = RemoteExecutor(config)

    with patch('paramiko.SSHClient') as mock_ssh_class:
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

        assert result.success is False
        assert "Error" in result.stderr

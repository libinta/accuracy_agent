from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from accuracy_agent.config import DebugConfig

# Try to import paramiko, but allow tests to run without it
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    paramiko = None

@dataclass
class ExecutionResult:
    """Result of remote test execution."""
    success: bool
    stdout: str
    stderr: str
    output_path: str
    exit_code: int = 0

class RemoteExecutor:
    """Execute test scripts on remote GPU/XPU hosts via SSH.

    SSH Authentication:
        - Requires key-based authentication (no password prompts supported)
        - SSH user and key path are configured via DebugConfig.ssh_user and
          DebugConfig.ssh_key_path respectively
        - If ssh_user is None, defaults to current system user (paramiko default)
        - If ssh_key_path is None, defaults to keys in ~/.ssh/ (paramiko default)
        - Ensure SSH keys are properly configured before using RemoteExecutor
    """

    def __init__(self, config: DebugConfig):
        """Initialize remote executor.

        Args:
            config: Debug configuration with host details. The config should have:
                - gpu_host: GPU host address
                - gpu_docker: GPU container name
                - xpu_host: XPU host address
                - xpu_docker: XPU container name
                - ssh_user (Optional[str]): SSH username. None uses current system user
                - ssh_key_path (Optional[str]): Path to SSH private key. None uses keys
                  from ~/.ssh/ directory (e.g., ~/.ssh/id_rsa, ~/.ssh/id_ed25519)

        Note:
            SSH key-based authentication is required. Password-based authentication
            is not supported.
        """
        self.config = config

    def execute_test_script(
        self,
        script_path: str,
        output_path: str,
        platform: str,
        timeout: int = 600
    ) -> ExecutionResult:
        """Execute test script on remote host via SSH and docker exec.

        Establishes an SSH connection to the remote host configured in DebugConfig,
        then executes a docker exec command to run the test script inside the
        specified container. Uses key-based authentication only (no passwords).

        Args:
            script_path: Path to test script on shared filesystem
            output_path: Path where output will be saved (on shared FS)
            platform: "gpu" or "xpu" - determines which host/container to use
            timeout: Execution timeout in seconds (default: 600s / 10 minutes)

        Returns:
            ExecutionResult with execution status and output

        Raises:
            RuntimeError: If paramiko is not installed
            ValueError: If platform is not "gpu" or "xpu"

        Note:
            SSH authentication uses the credentials from DebugConfig:
            - If ssh_user is None, uses current system user (paramiko default)
            - If ssh_key_path is None, uses keys from ~/.ssh/ (paramiko default)
            Only key-based authentication is supported; password prompts are not used.
        """
        # Select host and container
        if platform == "gpu":
            host = self.config.gpu_host
            container = self.config.gpu_docker
        elif platform == "xpu":
            host = self.config.xpu_host
            container = self.config.xpu_docker
        else:
            raise ValueError(f"Invalid platform: {platform}")

        # Build docker exec command
        cmd = f"docker exec {container} python {script_path}"

        print(f"[{platform.upper()}] Executing on {host}: {cmd}")

        # Execute via SSH
        if not PARAMIKO_AVAILABLE and paramiko is None:
            raise RuntimeError("paramiko is not installed")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # Connect via SSH with key-based authentication
            # username=None uses current system user (paramiko default behavior)
            # key_filename=None uses keys from ~/.ssh/ (paramiko default behavior)
            # No password authentication is used - ensure SSH keys are configured
            ssh.connect(
                host,
                username=self.config.ssh_user or None,
                key_filename=self.config.ssh_key_path or None,
                timeout=30
            )

            # Execute command
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)

            # Wait for completion
            exit_code = stdout.channel.recv_exit_status()

            # Read output
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')

            success = (exit_code == 0)

            if success:
                print(f"[{platform.upper()}] Success")
            else:
                print(f"[{platform.upper()}] Failed (exit code {exit_code})")
                print(f"[{platform.upper()}] stderr: {stderr_text}")

            return ExecutionResult(
                success=success,
                stdout=stdout_text,
                stderr=stderr_text,
                output_path=output_path,
                exit_code=exit_code
            )

        except Exception as e:
            print(f"[{platform.upper()}] Exception: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                output_path=output_path,
                exit_code=-1
            )

        finally:
            ssh.close()

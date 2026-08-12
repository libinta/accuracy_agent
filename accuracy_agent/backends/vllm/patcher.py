"""Auto-patch vLLM source to enable layer extraction via SSH"""

import paramiko
import subprocess
import socket
from pathlib import Path
from typing import Optional, Callable
import logging
from .patches.models import ModelPatchProvider, GLMPatchProvider, GLM52PatchProvider
from .patches.models.base import ModelPatchProvider as BaseModelPatchProvider

logger = logging.getLogger(__name__)


class VLLMPatcher:
    """Applies patches to vLLM source in docker container via SSH"""

    def __init__(
        self,
        host: str,
        docker: str,
        vllm_path: str,
        user: str = "root",
        model_name: Optional[str] = None
    ):
        """
        Initialize patcher

        Args:
            host: SSH host (e.g., "gpu-host.example.com")
            docker: Docker container name (e.g., "your_gpu_container")
            vllm_path: Path to vLLM inside docker (e.g., "/workspace/vllm")
            user: SSH username (default: "root")
            model_name: Model name for model-specific patches (e.g., "glm-5.2", "llama-3")
                       If None, uses generic patches
        """
        self.host = host
        self.docker = docker
        self.vllm_path = vllm_path
        self.user = user
        self.model_name = model_name
        self.ssh_client: Optional[paramiko.SSHClient] = None
        self.patch_provider: Optional[BaseModelPatchProvider] = None
        self.is_local = self._detect_local()

        # Initialize model-specific patch provider if model_name is given
        if model_name:
            self.patch_provider = self._get_patch_provider(model_name)
            logger.info(f"Using {self.patch_provider.__class__.__name__} for model {model_name}")
        else:
            logger.info("Using generic patches (no model name specified)")

        if self.is_local:
            logger.info(f"Detected local execution inside docker container {self.docker}")

    @staticmethod
    def _get_patch_provider(model_name: str) -> BaseModelPatchProvider:
        """
        Detect model type and return appropriate patch provider.

        Args:
            model_name: Model name or path (e.g., "glm-5.2", "THUDM/glm-4-9b")

        Returns:
            Model-specific patch provider instance
        """
        model_name_lower = model_name.lower()

        # GLM models (GLM-4, GLM-5, GLM-5.2, etc.)
        if 'glm' in model_name_lower:
            if 'glm-5.2' in model_name_lower or 'glm5.2' in model_name_lower:
                logger.info("Detected GLM-5.2 model")
                return GLM52PatchProvider()
            else:
                logger.info("Detected GLM model (generic)")
                return GLMPatchProvider()

        # TODO: Add more model types here as needed
        # elif 'llama' in model_name_lower:
        #     return LlamaPatchProvider()
        # elif 'qwen' in model_name_lower:
        #     return QwenPatchProvider()

        # Default: use GLM provider as fallback (since many models use Llama structure)
        logger.warning(
            f"Unknown model type '{model_name}', using GLMPatchProvider as fallback "
            "(most models use Llama-style layer structure)"
        )
        return GLMPatchProvider()

    def _detect_local(self) -> bool:
        """
        Detect if we're running inside the target docker container.

        Returns:
            True if running locally inside the target docker, False otherwise
        """
        try:
            # Check if we're inside a docker container
            if not Path("/.dockerenv").exists():
                return False

            # Primary detection: check if vllm_path exists locally
            # If we're inside the container, the vllm path should be accessible
            if Path(self.vllm_path).exists():
                logger.debug(f"Found vllm_path {self.vllm_path} locally, assuming local execution")
                return True

            # Secondary: check hostname - may match docker name or host name
            current_hostname = socket.gethostname()
            if current_hostname == self.docker or self.docker in current_hostname:
                logger.debug(f"Hostname {current_hostname} matches docker {self.docker}")
                return True

            # Tertiary: try to get container ID from cgroup
            result = subprocess.run(
                ["sh", "-c", "cat /proc/self/cgroup 2>/dev/null | grep -o -E 'docker[/-][a-f0-9]{12,}' | head -1 | cut -d/ -f2 | cut -c1-12"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                container_id = result.stdout.strip()
                if container_id and self.docker.startswith(container_id):
                    logger.debug(f"Container ID {container_id} matches docker {self.docker}")
                    return True

            return False
        except Exception as e:
            logger.debug(f"Error detecting local execution: {e}")
            return False

    def connect(self, ssh_key_path: str = None) -> None:
        """Establish SSH connection to host (or skip if running locally)"""
        if self.is_local:
            logger.info(f"Running locally inside {self.docker}, skipping SSH connection")
            return

        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        connect_kwargs = {
            "hostname": self.host,
            "username": self.user,
            "look_for_keys": True,
            "allow_agent": True,
        }

        if ssh_key_path:
            logger.info(f"Using SSH key: {ssh_key_path}")
            # Load the key explicitly for better compatibility with paramiko 5.0+
            try:
                pkey = paramiko.RSAKey.from_private_key_file(ssh_key_path)
                connect_kwargs["pkey"] = pkey
                connect_kwargs["look_for_keys"] = False
                connect_kwargs["allow_agent"] = False
            except Exception as e:
                logger.error(f"Failed to load SSH key from {ssh_key_path}: {e}")
                raise

        self.ssh_client.connect(**connect_kwargs)
        logger.info(f"Connected to {self.user}@{self.host}")

    def disconnect(self) -> None:
        """Close SSH connection (no-op if running locally)"""
        if self.is_local:
            logger.debug("Running locally, no SSH connection to close")
            return

        if self.ssh_client:
            self.ssh_client.close()
            self.ssh_client = None
            logger.info(f"Disconnected from {self.host}")

    def exec_in_docker(self, cmd: str) -> tuple[str, str]:
        """
        Execute command in docker container (or locally if running inside it)

        Args:
            cmd: Command to execute inside container

        Returns:
            Tuple of (stdout, stderr) as strings
        """
        if self.is_local:
            # Execute locally using subprocess
            logger.debug(f"Executing locally: {cmd}")
            try:
                result = subprocess.run(
                    ["bash", "-c", cmd],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                return result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return "", "Command timed out after 300s"
            except Exception as e:
                return "", f"Local execution failed: {e}"

        if not self.ssh_client:
            raise RuntimeError("Not connected - call connect() first")

        # Escape quotes in command
        cmd_escaped = cmd.replace('"', '\\"')
        docker_cmd = f'docker exec {self.docker} bash -c "{cmd_escaped}"'

        logger.debug(f"Executing: {docker_cmd}")
        stdin, stdout, stderr = self.ssh_client.exec_command(docker_cmd)

        stdout_str = stdout.read().decode()
        stderr_str = stderr.read().decode()

        return stdout_str, stderr_str

    def backup_file(self, filepath: str) -> None:
        """
        Create backup of original file if backup doesn't exist

        Args:
            filepath: Absolute path to file inside container
        """
        backup_path = f"{filepath}.original"
        cmd = f"[ ! -f {backup_path} ] && cp {filepath} {backup_path} || true"
        stdout, stderr = self.exec_in_docker(cmd)

        if stderr:
            logger.warning(f"Backup stderr: {stderr}")

        logger.info(f"Backed up {filepath} -> {backup_path}")

    def restore_original(self, filepath: str) -> None:
        """
        Restore original file from backup

        Args:
            filepath: Absolute path to file inside container
        """
        backup_path = f"{filepath}.original"
        cmd = f"[ -f {backup_path} ] && mv {backup_path} {filepath} || true"
        stdout, stderr = self.exec_in_docker(cmd)

        if stderr:
            logger.warning(f"Restore stderr: {stderr}")

        logger.info(f"Restored {filepath} from {backup_path}")

    def apply_patch_to_file(
        self,
        target_file: str,
        patch_content: str,
        anchor: Optional[str] = None,
        insert_before: bool = True
    ) -> None:
        """
        Apply patch content to target file with anchored insertion

        Args:
            target_file: Absolute path to file inside container
            patch_content: Content to insert
            anchor: Line pattern to search for (if None, appends to end)
            insert_before: If True, insert before anchor; if False, insert after
        """
        # Backup first
        self.backup_file(target_file)

        # Read current content
        stdout, stderr = self.exec_in_docker(f"cat {target_file}")
        if stderr:
            raise RuntimeError(f"Failed to read {target_file}: {stderr}")

        original_content = stdout

        # Apply patch with anchored insertion
        if anchor is None:
            # No anchor specified, append to end (legacy behavior)
            patched_content = original_content + "\n" + patch_content
        else:
            # Find anchor and insert at correct location with proper indentation
            lines = original_content.splitlines(keepends=True)
            patched_lines = []
            anchor_found = False

            for line in lines:
                if not anchor_found and anchor in line:
                    anchor_found = True
                    # Detect indentation from anchor line
                    indent = len(line) - len(line.lstrip())
                    indent_str = line[:indent]

                    # Indent patch content to match
                    patch_lines = patch_content.splitlines(keepends=True)
                    indented_patch = "".join(
                        indent_str + patch_line if patch_line.strip() else patch_line
                        for patch_line in patch_lines
                    )

                    if insert_before:
                        patched_lines.append(indented_patch)
                        patched_lines.append(line)
                    else:
                        patched_lines.append(line)
                        patched_lines.append(indented_patch)
                else:
                    patched_lines.append(line)

            if not anchor_found:
                raise RuntimeError(
                    f"Anchor not found in {target_file}: '{anchor}'. "
                    f"Cannot apply patch safely."
                )

            patched_content = "".join(patched_lines)

        # Write patched content
        if self.is_local:
            # Write directly to file system when running locally
            with open(target_file, 'w') as f:
                f.write(patched_content)
        else:
            # Write via SSH when running remotely
            temp_file = f"/tmp/vllm_patch_{Path(target_file).name}"

            # Upload to host temp location via SFTP
            with self.ssh_client.open_sftp() as sftp:
                with sftp.open(temp_file, 'w') as f:
                    f.write(patched_content)

            # Copy from host into docker container using docker cp
            docker_cp_cmd = f"docker cp {temp_file} {self.docker}:{target_file}"
            stdin, stdout, stderr = self.ssh_client.exec_command(docker_cp_cmd)

            stderr_str = stderr.read().decode()
            if stderr_str:
                raise RuntimeError(f"Failed to copy patched file into container: {stderr_str}")

        logger.info(f"Applied patch to {target_file}")

    def copy_file_to_container(self, local_path: Path, container_path: str) -> None:
        """
        Copy file from local filesystem to docker container

        Args:
            local_path: Local file path
            container_path: Destination path in container
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        if self.is_local:
            # Copy directly when running locally
            import shutil
            self.exec_in_docker(f"mkdir -p {Path(container_path).parent}")
            shutil.copy2(str(local_path), container_path)
            logger.info(f"Copied {local_path} -> {container_path} (local)")
        else:
            # Upload to host temp location via SFTP
            temp_path = f"/tmp/{local_path.name}"
            with self.ssh_client.open_sftp() as sftp:
                sftp.put(str(local_path), temp_path)

            # Create parent directory in container
            self.exec_in_docker(f"mkdir -p {Path(container_path).parent}")

            # Copy from host into docker container using docker cp
            docker_cp_cmd = f"docker cp {temp_path} {self.docker}:{container_path}"
            stdin, stdout, stderr = self.ssh_client.exec_command(docker_cp_cmd)

            stderr_str = stderr.read().decode()
            if stderr_str:
                raise RuntimeError(f"Failed to copy file into container: {stderr_str}")

            logger.info(f"Copied {local_path} -> {container_path}")

    def apply_all_patches(self) -> None:
        """Apply all vLLM patches with proper anchored insertion"""
        logger.info("Applying vLLM patches...")

        # Use model-specific patches if available, otherwise fall back to generic
        if self.patch_provider:
            logger.info("Using model-specific patches")
            self._apply_model_specific_patches()
        else:
            logger.info("Using generic patches")
            self._apply_generic_patches()

        # Copy debug_runner.py
        debug_runner_local = Path(__file__).parent / "patches" / "debug_runner.py"
        debug_runner_container = f"{self.vllm_path}/vllm/model_executor/debug_runner.py"
        self.copy_file_to_container(debug_runner_local, debug_runner_container)

        # Apply XPU memory detection fix
        self._apply_xpu_memory_fix()

        # Apply architecture-agnostic layer-limiting fix (make_layers)
        self._apply_make_layers_fix()

        logger.info("All patches applied successfully")

    def _apply_model_specific_patches(self) -> None:
        """Apply patches using model-specific patch provider"""
        if not self.patch_provider:
            raise RuntimeError("No patch provider configured")

        anchors = self.patch_provider.get_anchor_points()

        # Patch 1: default_loader.py for weight filtering
        loader_path = f"{self.vllm_path}/vllm/model_executor/model_loader/default_loader.py"
        weight_patch = self.patch_provider.get_weight_filter_patch()
        self.apply_patch_to_file(
            loader_path,
            weight_patch,
            anchor=anchors['weight_filter'],
            insert_before=True
        )

        # Patch 2: Model-specific file for layer initialization
        model_file = self.patch_provider.get_model_file_path()
        model_path = f"{self.vllm_path}/{model_file}"
        layer_patch = self.patch_provider.get_layer_init_patch()

        # Try to apply layer init patch (optional - vLLM may already support partial loading)
        try:
            self.apply_patch_to_file(
                model_path,
                layer_patch,
                anchor=anchors['layer_init'],
                insert_before=True
            )
            logger.info("Layer initialization patch applied successfully")
        except RuntimeError as e:
            logger.warning(
                f"Could not apply layer initialization patch to {model_file}: {e}. "
                "This is OK if vLLM already supports partial layer loading via make_layers(). "
                "Weight filtering will still work."
            )

    def _apply_generic_patches(self) -> None:
        """Apply generic patches (original behavior)"""
        from .patches import get_weight_filter_patch, get_layer_init_patch

        # Patch 1: default_loader.py for weight filtering
        loader_path = f"{self.vllm_path}/vllm/model_executor/model_loader/default_loader.py"
        weight_patch = get_weight_filter_patch()
        # Try new anchor first (for newer vLLM versions)
        try:
            self.apply_patch_to_file(
                loader_path,
                weight_patch,
                anchor="return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)",
                insert_before=True
            )
        except RuntimeError:
            # Fall back to old anchor for older vLLM versions
            self.apply_patch_to_file(
                loader_path,
                weight_patch,
                anchor="for name, param in weights_iterator:",
                insert_before=True
            )

        # Patch 2: llama.py for layer initialization (optional)
        llama_path = f"{self.vllm_path}/vllm/model_executor/models/llama.py"
        layer_patch = get_layer_init_patch()
        # Try to find the layers initialization line
        try:
            self.apply_patch_to_file(
                llama_path,
                layer_patch,
                anchor="self.layers = nn.ModuleList(",
                insert_before=True
            )
            logger.info("Layer initialization patch applied successfully")
        except RuntimeError:
            # Try alternate pattern if first fails
            try:
                self.apply_patch_to_file(
                    llama_path,
                    layer_patch,
                    anchor="layers = nn.ModuleList(",
                    insert_before=True
                )
                logger.info("Layer initialization patch applied successfully")
            except RuntimeError:
                logger.warning(
                    "Could not find layer initialization anchor in llama.py. "
                    "This is OK if vLLM already supports partial layer loading via make_layers(). "
                    "Weight filtering will still work."
                )

    def _apply_xpu_memory_fix(self) -> None:
        """Apply XPU memory detection fix to work around torch.xpu.get_memory_info() bug"""
        logger.info("Applying XPU memory detection fix...")

        mem_utils_path = f"{self.vllm_path}/vllm/utils/mem_utils.py"

        # Backup original
        self.backup_file(mem_utils_path)

        # XPU memory fix patch (no leading indentation - patcher adds it automatically)
        xpu_fix = """
# ACCURACY_AGENT PATCH: Fix XPU memory detection bug
# torch.xpu.get_memory_info() incorrectly returns (0, total) instead of (free, total)
if device.type == 'xpu':
    # For XPU, get_memory_info returns incorrect free_memory (always 0)
    # Work around by calculating: free = total - allocated
    if self.free_memory == 0 and self.total_memory > 0:
        # Use torch's internal tracking instead
        try:
            allocated = torch.xpu.memory_allocated(device)
            reserved = torch.xpu.memory_reserved(device)
            # Free memory is total minus what's actually reserved by PyTorch
            # Be conservative: assume what PyTorch doesn't know about takes 10% of total
            self.free_memory = int(self.total_memory * 0.9 - reserved)
            if self.free_memory < 0:
                self.free_memory = 0
        except Exception:
            # Fallback: assume 80% is free if we can't measure
            self.free_memory = int(self.total_memory * 0.8)
# END ACCURACY_AGENT PATCH
"""

        try:
            # Insert after the get_memory_info line
            self.apply_patch_to_file(
                mem_utils_path,
                xpu_fix,
                anchor="self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)",
                insert_before=False
            )
            logger.info("XPU memory detection fix applied successfully")
        except RuntimeError as e:
            logger.warning(f"Could not apply XPU memory fix: {e}. This is OK for non-XPU backends.")

    def _apply_make_layers_fix(self) -> None:
        """Apply architecture-agnostic layer-limiting fix.

        Instead of patching each model file (glm4.py, deepseek_v2.py, llama.py, ...)
        to construct only a subset of layers, we patch the shared make_layers()
        utility that every model uses. make_layers() already builds real modules
        only for [start_layer, end_layer) and fills the rest with cheap
        PPMissingLayer() placeholders (reusing vLLM's pipeline-parallel path).
        We simply clamp that range to a debug window read from env vars, so only
        the requested layers allocate weight tensors -- avoiding OOM on large
        models with a small number of cards.
        """
        logger.info("Applying architecture-agnostic make_layers fix...")

        utils_path = f"{self.vllm_path}/vllm/model_executor/models/utils.py"

        # Backup original
        self.backup_file(utils_path)

        # No leading indentation on the base level -- patcher adds the anchor's
        # indent as a prefix to each line, preserving the nested if-block.
        make_layers_fix = """# ACCURACY_AGENT PATCH: clamp layer construction to debug window
import os as _aa_os
_aa_start = _aa_os.environ.get('ACCURACY_DEBUG_LAYER_START')
_aa_end = _aa_os.environ.get('ACCURACY_DEBUG_LAYER_END')
if _aa_start is not None and _aa_end is not None:
    start_layer = max(start_layer, int(_aa_start))
    end_layer = min(end_layer, int(_aa_end))
# END ACCURACY_AGENT PATCH
"""

        try:
            # Insert right before the ModuleList construction, after start_layer/
            # end_layer have been computed by get_pp_indices().
            self.apply_patch_to_file(
                utils_path,
                make_layers_fix,
                anchor="modules = torch.nn.ModuleList(",
                insert_before=True
            )
            logger.info("make_layers layer-limiting fix applied successfully")
        except RuntimeError as e:
            logger.warning(
                f"Could not apply make_layers fix: {e}. "
                "Falling back to model-specific layer patches (may OOM on large models)."
            )

    def cleanup(self) -> None:
        """Restore all original files"""
        logger.info("Cleaning up vLLM patches...")

        files_to_restore = [
            f"{self.vllm_path}/vllm/model_executor/model_loader/default_loader.py",
            f"{self.vllm_path}/vllm/model_executor/models/llama.py",
            f"{self.vllm_path}/vllm/utils/mem_utils.py",
            f"{self.vllm_path}/vllm/model_executor/models/utils.py",
        ]

        # Add model-specific file if using model-specific patches
        if self.patch_provider:
            model_file = self.patch_provider.get_model_file_path()
            model_path = f"{self.vllm_path}/{model_file}"
            if model_path not in files_to_restore:
                files_to_restore.append(model_path)

        for filepath in files_to_restore:
            self.restore_original(filepath)

        # Remove debug_runner.py
        self.exec_in_docker(f"rm -f {self.vllm_path}/vllm/model_executor/debug_runner.py")

        logger.info("Cleanup complete")

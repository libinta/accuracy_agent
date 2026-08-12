"""vLLM backend for accuracy testing with automatic patching"""

import shlex
from pathlib import Path
import logging
from typing import TYPE_CHECKING

from ..base import Backend, BackendConfig
from .patcher import VLLMPatcher
from .memory_check import check_memory

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


class VLLMBackend(Backend):
    """vLLM backend with automatic source patching for layer extraction"""

    def __init__(self, config: BackendConfig, model_path: str, shared_fs: str):
        """
        Initialize vLLM backend

        Args:
            config: Backend configuration
            model_path: Path to model checkpoint (accessible from docker)
            shared_fs: Shared filesystem path for exchanging tensors
        """
        super().__init__(config, model_path, shared_fs)
        model_name = Path(model_path).name
        self.patcher = VLLMPatcher(
            host=config.host,
            docker=config.docker,
            vllm_path=config.vllm_path,
            user=config.user,
            model_name=model_name,
        )
        self.is_patched = False
        self.memory_mode = None  # Will be set in setup()

    def setup(self) -> None:
        """Apply vLLM patches via SSH with memory check"""
        logger.info(f"Setting up vLLM backend on {self.config.host}/{self.config.docker}")

        try:
            self.patcher.connect(ssh_key_path=self.config.ssh_key_path)

            # Check memory to determine mode
            logger.info("Checking memory availability...")
            memory_status = check_memory(
                self.model_path,
                self.config.host,
                self.config.docker,
                self.config.device_type,
                self.patcher,
                cards=self.config.cards
            )

            self.memory_mode = memory_status.mode
            logger.info(
                f"Memory mode: {self.memory_mode} "
                f"(required={memory_status.required_gb:.1f}GB, "
                f"available={memory_status.available_gb:.1f}GB)"
            )

            # Apply patches
            self.patcher.apply_all_patches()
            self.is_patched = True
            logger.info("vLLM patches applied successfully")

        except Exception as e:
            logger.error(f"Failed to setup vLLM backend: {e}")
            self.is_patched = False
            raise RuntimeError(f"Failed to apply vLLM patches: {e}") from e

    def run_layer_range(
        self,
        layer_start: int,
        layer_end: int,
        prompt: str,
    ) -> "torch.Tensor":
        """
        Run layers [layer_start, layer_end) and return hidden states

        Args:
            layer_start: First layer to compute (inclusive)
            layer_end: Last layer to compute (exclusive)
            prompt: Input text prompt

        Returns:
            Hidden states tensor at layer_end
        """
        import torch

        if not self.is_patched:
            raise RuntimeError("Backend not setup - call setup() first")

        # Generate output path on shared filesystem
        output_path = Path(self.shared_fs) / f"hidden_states_{layer_start}_{layer_end}.pt"

        # Map memory mode vocabulary: "full"/"partial" -> "full_model"/"partial_layers"
        mode_map = {"full": "full_model", "partial": "partial_layers"}
        load_mode = mode_map.get(self.memory_mode, "full_model") if self.memory_mode else "full_model"

        # Escape single quotes in prompt for shell safety
        prompt_escaped = prompt.replace("'", "'\\''")

        # Construct command to run debug_runner.py
        # Use shlex.quote for all paths and user-provided strings to prevent shell injection
        cmd = (
            f"cd {shlex.quote(self.config.vllm_path)} && "
            f"python -m vllm.model_executor.debug_runner "
            f"--model-path {shlex.quote(self.model_path)} "
            f"--layer-start {layer_start} "
            f"--layer-end {layer_end} "
            f"--prompt '{prompt_escaped}' "
            f"--output {shlex.quote(str(output_path))} "
            f"--device {self.config.device_type} "
            f"--cards {self.config.cards} "
            f"--load-mode {load_mode}"
        )

        logger.info(f"Running layers [{layer_start}, {layer_end}) in {load_mode} mode")
        logger.debug(f"Command: {cmd}")

        # Execute in docker
        stdout, stderr = self.patcher.exec_in_docker(cmd)

        # Check for errors (case-insensitive)
        if stderr and "error" in stderr.lower():
            raise RuntimeError(f"vLLM execution failed: {stderr}")

        if not output_path.exists():
            raise RuntimeError(
                f"Output file not created: {output_path}\n"
                f"stdout: {stdout}\nstderr: {stderr}"
            )

        # Load hidden states from shared filesystem
        logger.info(f"Loading hidden states from {output_path}")
        hidden_states = torch.load(output_path)

        logger.info(f"Extracted hidden states shape: {hidden_states.shape}")
        return hidden_states

    def cleanup(self) -> None:
        """Restore original vLLM files"""
        if self.is_patched:
            logger.info("Cleaning up vLLM patches")
            self.patcher.cleanup()
            self.patcher.disconnect()
            self.is_patched = False

    def is_available(self) -> bool:
        """Check if vLLM is available at specified path"""
        try:
            if not self.patcher.ssh_client:
                self.patcher.connect()

            stdout, stderr = self.patcher.exec_in_docker(
                f"[ -d {shlex.quote(self.config.vllm_path)} ] && echo 'exists'"
            )

            available = 'exists' in stdout
            logger.info(f"vLLM availability at {self.config.vllm_path}: {available}")
            return available
        except Exception as e:
            logger.error(f"Failed to check vLLM availability: {e}")
            raise RuntimeError(f"Failed to connect or check vLLM installation: {e}") from e

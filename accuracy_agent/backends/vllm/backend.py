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
        # Model-type routing (VLLMPatcher._get_patch_provider) substring-matches
        # on this name. The HF hub cache resolves a model to
        # ".../models--Org--Model/snapshots/<commit-hash>", whose leaf dir name is
        # a bare commit hash -- so Path(model_path).name would be a hash and route
        # to the wrong (fallback) provider. Walk up to the "models--Org--Model"
        # component so a snapshot path still carries the real model name.
        model_name = Path(model_path).name
        _parts = Path(model_path).parts
        if "snapshots" in _parts:
            for _part in _parts:
                if _part.startswith("models--"):
                    model_name = _part
                    break
        self.patcher = VLLMPatcher(
            host=config.host,
            docker=config.docker,
            vllm_path=config.vllm_path,
            user=config.user,
            model_name=model_name,
            device_type=config.device_type,
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

        # Generate output path on shared filesystem. The device_type is part of
        # the filename because the GPU and XPU backends share the same shared_fs
        # and run in parallel for the same layer range -- without it both would
        # write (and then read back) the SAME file, racing so the comparison
        # loads one device's tensor twice (spurious cos=1.0).
        output_path = (
            Path(self.shared_fs)
            / f"hidden_states_{self.config.device_type}_{layer_start}_{layer_end}.pt"
        )

        # Map memory mode vocabulary: "full"/"partial" -> "full_model"/"partial_layers"
        mode_map = {"full": "full_model", "partial": "partial_layers"}
        load_mode = mode_map.get(self.memory_mode, "full_model") if self.memory_mode else "full_model"

        # Escape single quotes in prompt for shell safety
        prompt_escaped = prompt.replace("'", "'\\''")

        # Select the target device(s) in the ENVIRONMENT, before python starts.
        # `python -m vllm.model_executor.debug_runner` imports the vllm package
        # (which initializes the Level-Zero / CUDA driver via torch and
        # enumerates every visible device) BEFORE debug_runner's own module body
        # runs -- so setting the mask from inside debug_runner is too late and is
        # silently ignored, landing the run on the default (often busy) device 0.
        # Exporting it in the launch command is the only point that takes effect.
        cards = shlex.quote(str(self.config.cards))
        if self.config.device_type == "xpu":
            affinity_env = f"ZE_AFFINITY_MASK={cards} "
        elif self.config.device_type == "cuda":
            affinity_env = f"CUDA_VISIBLE_DEVICES={cards} "
        elif self.config.device_type == "hpu":
            # Gaudi (Habana) selects visible accelerators via module IDs.
            affinity_env = f"HABANA_VISIBLE_MODULES={cards} "
        else:
            affinity_env = ""

        # Construct command to run debug_runner.py
        # Use shlex.quote for all paths and user-provided strings to prevent shell injection
        cmd = (
            f"cd {shlex.quote(self.config.vllm_path)} && "
            f"{affinity_env}"
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
        # stream=True: this is the long-running vLLM engine run; mirror its
        # output live so shard-load %, HPU compile/warmup and forward progress
        # are visible instead of a multi-minute silent hang. All the short
        # patch-time file ops use the default (quiet) exec_in_docker.
        stdout, stderr = self.patcher.exec_in_docker(cmd, stream=True)

        # Dump the debug_runner subprocess output to a file so its prints
        # (enforce_eager value, compile/graph signals, capture diagnostics) are
        # inspectable -- exec_in_docker captures them, so they never reach the
        # parent CLI log otherwise. Best-effort; keyed by device+layer window.
        # Written to the current working directory (where the CLI is launched)
        # so it lands next to the run instead of at the shared_fs root.
        try:
            _dbg = (
                Path.cwd()
                / f"debugrunner_{self.config.device_type}_{layer_start}_{layer_end}.log"
            )
            _dbg.write_text(
                f"$ {cmd}\n\n===== STDOUT =====\n{stdout}\n\n===== STDERR =====\n{stderr}\n"
            )
        except Exception:
            pass

        # Only treat EXPLICIT failure signals as fatal here. A bare
        # "error" substring is too broad: HPU emits benign warnings
        # ("... does not have any effect"), vLLM logs can say "0 errors", and
        # the FP8-GEMM/sparse-MLA fixes intentionally log "Could not apply ...".
        # debug_runner.main() prints an "ERROR:" line + a traceback on real
        # failure, so key off those. Final success is still gated on the output
        # file existing below (so a missed signal cannot pass silently).
        # Local runs now STREAM output: exec_in_docker merges child stdout+stderr
        # and returns it as stdout (stderr empty), so scan BOTH streams for the
        # failure signals. Remote (SSH) runs still split the two, so the union
        # covers both transports.
        combined = f"{stdout}\n{stderr}"
        if "ERROR:" in combined or "Traceback (most recent call last)" in combined:
            raise RuntimeError(f"vLLM execution failed:\n{combined}")

        # A remote backend wrote output_path onto ITS filesystem, which the two
        # containers generally do not share. Pull it (and the per-layer
        # companion) back to this machine before the comparator reads them.
        if not self.patcher.is_local:
            fetched = self.patcher.copy_file_from_container(str(output_path), output_path)
            if fetched:
                self.patcher.copy_file_from_container(
                    str(output_path) + ".alllayers",
                    Path(str(output_path) + ".alllayers"),
                )

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

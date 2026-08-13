"""Bisection engine for finding GPU/XPU divergences."""
import torch
import concurrent.futures
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.remote_executor import RemoteExecutor
from accuracy_agent.test_harness_generator import generate_test_harness, save_test_harness
from accuracy_agent.comparator import compare_tensors, ComparisonResult
from accuracy_agent.backends.factory import create_backend
from accuracy_agent.backends.base import BackendConfig, Backend
from accuracy_agent.vllm_source_builder import maybe_autoconfigure_peers

def _extract_compare_tensor(data: dict) -> "torch.Tensor":
    """Pull the tensor to compare out of a harness output dict.

    Prefers the intermediate ``hidden_states`` saved by the layer-subset
    harness. Falls back to ``logits`` for backward compatibility with any
    older output files.
    """
    if "hidden_states" in data:
        return data["hidden_states"]
    if "logits" in data:
        return data["logits"]
    raise KeyError(
        "Harness output missing both 'hidden_states' and 'logits'; "
        f"got keys: {list(data.keys())}"
    )


@dataclass
class BisectionResult:
    """Result of bisection process."""
    divergent_layer: Optional[int]
    comparison_results: List[ComparisonResult] = field(default_factory=list)
    report: str = ""
    # Parallel to comparison_results: (type_name, layer_index) describing what
    # each comparison corresponds to. Populated by representative-layer runs so
    # the report can label rows by layer KIND ("moe") rather than a bare index.
    tested_layers: List[tuple] = field(default_factory=list)
    # XPU-only extraction: no GPU peer was configured, so hidden states were
    # captured on XPU alone (no comparison performed). extracted_path points at
    # the saved tensor for a later GPU-vs-XPU comparison.
    extracted_only: bool = False
    extracted_path: Optional[str] = None

class Bisector:
    """Hierarchical bisection engine for finding GPU/XPU divergences."""

    def __init__(self, config: DebugConfig, model_info: ModelInfo):
        """Initialize bisector.

        Args:
            config: Debug configuration
            model_info: Model architecture info
        """
        self.config = config
        self.model_info = model_info

        # Fill in the docker peers that were left unset: built from vllm_commit,
        # or matched to the XPU container's vLLM version. Must run BEFORE
        # xpu_only is computed below, since a successful setup is what turns a
        # would-be XPU-only run into a real GPU-vs-XPU comparison. No-op when
        # both sides are already configured.
        maybe_autoconfigure_peers(config)

        # XPU-only mode: no GPU host configured, so we cannot do a GPU-vs-XPU
        # comparison. Instead we extract XPU hidden states alone (the "small
        # steps for big model" capture phase) and save them for a later
        # comparison once a GPU peer is available.
        self.xpu_only = not (config.gpu_host or config.gpu_docker)

        # Use backends if backend is specified
        if config.backend != "pytorch":
            self.use_backends = True
            self.gpu_backend = None
            self.xpu_backend = None
        else:
            # Backward compatibility: use old RemoteExecutor
            self.use_backends = False
            self.executor = RemoteExecutor(config)

    def _parallel_setup(self) -> tuple[Backend, Backend]:
        """
        Setup GPU and XPU backends in parallel.

        Returns:
            Tuple of (gpu_backend, xpu_backend)
        """
        # Create backend configs
        gpu_config = BackendConfig(
            host=self.config.gpu_host,
            docker=self.config.gpu_docker,
            vllm_path=self.config.gpu_vllm_path,
            cards=self.config.gpu_cards,
            device_type="cuda",
            user=self.config.gpu_user,
            ssh_key_path=self.config.gpu_ssh_key_path,
            inside_container=self.config.gpu_inside_container
        )

        xpu_config = BackendConfig(
            host=self.config.xpu_host,
            docker=self.config.xpu_docker,
            vllm_path=self.config.xpu_vllm_path,
            cards=self.config.xpu_cards,
            device_type="xpu",
            user=self.config.xpu_user,
            ssh_key_path=self.config.xpu_ssh_key_path,
            inside_container=self.config.xpu_inside_container
        )

        def setup_backend(config: BackendConfig, device_name: str):
            """Setup a single backend"""
            print(f"Setting up {device_name} backend...")
            backend = create_backend(
                self.config.backend,
                config,
                self.config.model_path,
                self.config.shared_fs
            )
            backend.setup()
            print(f"✓ {device_name} backend ready")
            return backend

        # XPU-only mode: no GPU peer configured. Skip the GPU backend entirely
        # (on an XPU-only host a "cuda" backend has no device and would land on
        # a busy XPU card and fail) and set up only the XPU backend.
        if self.xpu_only:
            xpu_backend = setup_backend(xpu_config, "XPU")
            return None, xpu_backend

        # Run GPU and XPU setup in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            gpu_future = executor.submit(setup_backend, gpu_config, "GPU")
            xpu_future = executor.submit(setup_backend, xpu_config, "XPU")

            gpu_backend = gpu_future.result()
            xpu_backend = xpu_future.result()

        return gpu_backend, xpu_backend

    def _test_layer_range_parallel(
        self,
        layer_start: int,
        layer_end: int
    ) -> ComparisonResult:
        """
        Test layer range with parallel GPU/XPU execution.

        Args:
            layer_start: First layer (inclusive)
            layer_end: Last layer (exclusive)

        Returns:
            ComparisonResult from comparing GPU vs XPU hidden states
        """
        print(f"Testing layers [{layer_start}, {layer_end}) in parallel...")

        prompt = self.config.test_prompt

        # Run GPU and XPU in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            gpu_future = executor.submit(
                self.gpu_backend.run_layer_range,
                layer_start,
                layer_end,
                prompt
            )
            xpu_future = executor.submit(
                self.xpu_backend.run_layer_range,
                layer_start,
                layer_end,
                prompt
            )

            hidden_states_gpu = gpu_future.result()
            hidden_states_xpu = xpu_future.result()

        # Compare tensors
        result = compare_tensors(hidden_states_gpu, hidden_states_xpu)

        return result

    def _extract_xpu_only(
        self,
        layer_start: int,
        layer_end: int
    ) -> BisectionResult:
        """Extract XPU hidden states without a GPU peer to compare against.

        Used when no GPU host is configured. Runs the XPU backend for the
        requested layer window, saves the captured hidden states into
        ``output_dir`` (so they survive for a later GPU-vs-XPU comparison), and
        returns a BisectionResult flagged as extraction-only.
        """
        print(f"XPU-only extraction of layers [{layer_start}, {layer_end}) "
              f"(no GPU peer configured)...")

        hidden_states = self.xpu_backend.run_layer_range(
            layer_start, layer_end, self.config.test_prompt
        )

        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"hidden_states_xpu_{layer_start}_{layer_end}.pt"
        torch.save(hidden_states, out_path)

        report = (
            f"XPU hidden states for layers [{layer_start}, {layer_end}) "
            f"extracted to {out_path} (shape {tuple(hidden_states.shape)}). "
            f"No GPU peer configured, so no comparison was performed."
        )
        print(f"✓ {report}")

        return BisectionResult(
            divergent_layer=None,
            comparison_results=[],
            report=report,
            extracted_only=True,
            extracted_path=str(out_path),
        )

    def bisect_layer_set(
        self,
        layer_groups: List[tuple]
    ) -> BisectionResult:
        """Compare a set of representative unique layers, one per layer type.

        Unlike ``bisect_layers`` (which sweeps a contiguous range), this tests
        exactly the (type_name, layer_index) representatives the model loader
        derived -- so a MoE model exercises both a dense and a MoE layer instead
        of only the dense prefix. Each representative layer N is captured via the
        window [N, N+1); the make_layers clamp builds layers 0..N so layer N sees
        the correct upstream residual stream before it is hooked.

        Args:
            layer_groups: list of (type_name, layer_index) to test.

        Returns:
            BisectionResult with one comparison per representative, labeled via
            ``tested_layers``. divergent_layer is the first representative that
            does not match (or None if all match).
        """
        try:
            if self.use_backends:
                needs_setup = self.xpu_backend is None or (
                    not self.xpu_only and self.gpu_backend is None
                )
                if needs_setup:
                    self.gpu_backend, self.xpu_backend = self._parallel_setup()

            labels = [f"{name}@{idx}" for name, idx in layer_groups]
            print(f"\n{'='*60}")
            print(f"Comparing representative layers: {', '.join(labels)}")
            print(f"{'='*60}\n")

            # XPU-only mode has no GPU peer -- fall back to extracting each
            # representative layer's hidden states without comparison.
            if self.use_backends and self.xpu_only:
                last = None
                for _name, idx in layer_groups:
                    last = self._extract_xpu_only(idx, idx + 1)
                return last

            results: List[ComparisonResult] = []
            tested: List[tuple] = []
            divergent = None

            for name, idx in layer_groups:
                if self.use_backends:
                    result = self._test_layer_range_parallel(idx, idx + 1)
                else:
                    result = self._test_layer_range(idx, idx + 1)

                results.append(result)
                tested.append((name, idx))

                if result.match:
                    print(f"✓ Layer {idx} ({name}): {result.summary()}")
                else:
                    print(f"✗ Layer {idx} ({name}): {result.summary()}")
                    if divergent is None:
                        divergent = idx

            if divergent is None:
                report = "All representative layers match: " + ", ".join(labels)
            else:
                report = f"Divergence found in layer {divergent}"

            return BisectionResult(
                divergent_layer=divergent,
                comparison_results=results,
                report=report,
                tested_layers=tested,
            )
        finally:
            if self.use_backends:
                if self.gpu_backend is not None:
                    self.gpu_backend.cleanup()
                if self.xpu_backend is not None:
                    self.xpu_backend.cleanup()

    def bisect_layers(
        self,
        layer_start: int,
        layer_end: int
    ) -> BisectionResult:
        """Bisect to find divergent layer.

        Args:
            layer_start: First layer to test (inclusive)
            layer_end: Last layer to test (exclusive)

        Returns:
            BisectionResult with divergent layer (if found)
        """
        try:
            # Setup backends if using new backend system. In XPU-only mode the
            # GPU backend is intentionally None, so gate setup on the XPU one.
            if self.use_backends:
                needs_setup = self.xpu_backend is None or (
                    not self.xpu_only and self.gpu_backend is None
                )
                if needs_setup:
                    self.gpu_backend, self.xpu_backend = self._parallel_setup()

            print(f"\n{'='*60}")
            print(f"Bisecting layers {layer_start}-{layer_end}")
            print(f"{'='*60}\n")

            # XPU-only extraction: capture XPU hidden states, no comparison.
            if self.use_backends and self.xpu_only:
                return self._extract_xpu_only(layer_start, layer_end)

            # Use parallel execution if backends available
            if self.use_backends:
                comparison = self._test_layer_range_parallel(layer_start, layer_end)
            else:
                # Backward compatibility: use old serial execution
                comparison = self._test_layer_range(layer_start, layer_end)

            if comparison.match:
                print(f"✓ Layers {layer_start}-{layer_end}: {comparison.summary()}")
                return BisectionResult(
                    divergent_layer=None,
                    comparison_results=[comparison],
                    report=f"Layers {layer_start}-{layer_end} match"
                )

            print(f"✗ Layers {layer_start}-{layer_end}: {comparison.summary()}")

            # If testing single layer, we found the divergent one
            if layer_end - layer_start == 1:
                return BisectionResult(
                    divergent_layer=layer_start,
                    comparison_results=[comparison],
                    report=f"Divergence found in layer {layer_start}"
                )

            # Bisect recursively
            results = []
            divergent = None

            for layer_idx in range(layer_start, layer_end):
                if self.use_backends:
                    result = self._test_layer_range_parallel(layer_idx, layer_idx + 1)
                else:
                    result = self._test_layer_range(layer_idx, layer_idx + 1)

                results.append(result)

                if not result.match:
                    print(f"✗ Layer {layer_idx}: {result.summary()}")
                    if divergent is None:
                        divergent = layer_idx
                else:
                    print(f"✓ Layer {layer_idx}: {result.summary()}")

            return BisectionResult(
                divergent_layer=divergent,
                comparison_results=results,
                report=f"Bisection complete. Divergent layer: {divergent}"
            )
        finally:
            # Always clean up backends to reverse patches
            if self.use_backends:
                if self.gpu_backend is not None:
                    self.gpu_backend.cleanup()
                if self.xpu_backend is not None:
                    self.xpu_backend.cleanup()

    def _test_layer_range(
        self,
        layer_start: int,
        layer_end: int
    ) -> ComparisonResult:
        """Test a range of layers on GPU and XPU.

        Args:
            layer_start: First layer (inclusive)
            layer_end: Last layer (exclusive)

        Returns:
            ComparisonResult from comparing GPU vs XPU outputs
        """
        print(f"\nTesting layers {layer_start}-{layer_end}...")

        # Generate test scripts
        gpu_script = generate_test_harness(
            self.config, self.model_info, layer_start, layer_end, "gpu"
        )
        xpu_script = generate_test_harness(
            self.config, self.model_info, layer_start, layer_end, "xpu"
        )

        # Save scripts to shared FS
        gpu_script_path = f"{self.config.output_dir}/test_gpu_{layer_start}_{layer_end}.py"
        xpu_script_path = f"{self.config.output_dir}/test_xpu_{layer_start}_{layer_end}.py"

        save_test_harness(gpu_script, gpu_script_path)
        save_test_harness(xpu_script, xpu_script_path)

        # Output paths
        gpu_output_path = f"{self.config.output_dir}/layer_{layer_start}_{layer_end}_gpu.pt"
        xpu_output_path = f"{self.config.output_dir}/layer_{layer_start}_{layer_end}_xpu.pt"

        # Execute on both platforms
        gpu_result = self.executor.execute_test_script(
            gpu_script_path, gpu_output_path, "gpu"
        )

        xpu_result = self.executor.execute_test_script(
            xpu_script_path, xpu_output_path, "xpu"
        )

        # Check execution success
        if not gpu_result.success:
            raise RuntimeError(f"GPU execution failed: {gpu_result.stderr}")

        if not xpu_result.success:
            raise RuntimeError(f"XPU execution failed: {xpu_result.stderr}")

        # Load outputs from shared FS
        gpu_data = torch.load(gpu_output_path)
        xpu_data = torch.load(xpu_output_path)

        # Compare the intermediate hidden states produced by this layer range.
        # The harness saves hidden states (not final-model logits) so that each
        # layer range yields a distinct, isolatable output for bisection.
        gpu_tensor = _extract_compare_tensor(gpu_data)
        xpu_tensor = _extract_compare_tensor(xpu_data)
        comparison = compare_tensors(gpu_tensor, xpu_tensor)

        return comparison

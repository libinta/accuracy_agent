"""
Standalone script for extracting hidden states from vLLM at specific layers.
This script is copied into vLLM source tree during patching.
"""

import argparse
import os
import sys


def _set_device_affinity_from_argv() -> None:
    """Select the target device(s) BEFORE torch is imported.

    Importing torch initializes the Level-Zero (XPU) / CUDA driver and
    enumerates every visible device. Setting ZE_AFFINITY_MASK /
    CUDA_VISIBLE_DEVICES *after* that point is silently ignored, so vLLM would
    land on the default device 0 (often busy) regardless of --cards. We peek at
    --cards/--device from argv and set the mask up front. No-op when the flags
    are absent (e.g. imported as a module rather than run via CLI).
    """
    argv = sys.argv

    def _get(flag: str):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        prefix = flag + "="
        for a in argv:
            if a.startswith(prefix):
                return a.split("=", 1)[1]
        return None

    cards = _get("--cards")
    device = _get("--device") or "cuda"
    if cards:
        if device == "xpu":
            os.environ["ZE_AFFINITY_MASK"] = cards
        elif device == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = cards


_set_device_affinity_from_argv()

import torch
from pathlib import Path


def _aa_capture_layers(model, layer_start, layer_end, save_input_path=None,
                       inject_ref=None):
    """Register forward hooks on the decoder layers and return (captured, handles).

    Runs either in the driver (TP=1, in-process worker) OR inside each TP worker
    process (TP>1, via collective_rpc). Kept SELF-CONTAINED (imports locally, no
    reliance on run_partial_layers' closures) so it works identically in a
    separate worker process.

    captured[abs_idx] = post-layer residual stream (hidden_states + residual),
    detached to CPU float32. For TP>1 the decoder-layer output is all-reduced, so
    every rank holds identical full-width hidden states.
    """
    import torch
    from pathlib import Path

    # Locate the decoder stack regardless of wrapper depth. Multimodal
    # *ForConditionalGeneration wrappers (KimiK25...) keep the text model under
    # .language_model with the stack at language_model.model.layers.
    inner = getattr(model, "language_model", model)
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        layers = inner.model.layers
    elif hasattr(inner, "layers"):
        layers = inner.layers
    else:
        layers = model.model.layers if hasattr(model, "model") else model.layers

    captured = {}
    handles = []

    def make_hook(abs_idx):
        def hook(module, args, output):
            if isinstance(output, tuple):
                hs = output[0]
                res = output[1] if len(output) > 1 else None
                combined = hs + res if res is not None else hs
            else:
                combined = output
            captured[abs_idx] = combined.detach().to(torch.float32).cpu()
        return hook

    n = min(layer_end, len(layers))
    for i in range(layer_start, n):
        handles.append(layers[i].register_forward_hook(make_hook(i)))

    # Optional per-layer input isolation, acting on the deepest built layer.
    tested_pos = n - 1

    def _find_hs_res(args, kwargs):
        if "hidden_states" in kwargs:
            return kwargs.get("hidden_states"), kwargs.get("residual"), "kwargs"
        hs = args[1] if len(args) > 1 else None
        res = args[2] if len(args) > 2 else None
        return hs, res, "args"

    if save_input_path and tested_pos >= 0:
        def save_pre_hook(module, args, kwargs):
            hs, res, _ = _find_hs_res(args, kwargs)
            payload = {
                "hidden_states": hs.detach().to(torch.float32).cpu() if hs is not None else None,
                "residual": res.detach().to(torch.float32).cpu() if res is not None else None,
            }
            Path(save_input_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, save_input_path)
            return None
        handles.append(layers[tested_pos].register_forward_pre_hook(
            save_pre_hook, with_kwargs=True))

    if inject_ref is not None and tested_pos >= 0:
        def inject_pre_hook(module, args, kwargs):
            hs, res, kind = _find_hs_res(args, kwargs)
            hs_ref = inject_ref.get("hidden_states")
            res_ref = inject_ref.get("residual")
            dev = hs.device if hs is not None else (res.device if res is not None else None)
            dt = hs.dtype if hs is not None else (res.dtype if res is not None else None)
            hs_new = hs_ref.to(device=dev, dtype=dt) if hs_ref is not None else hs
            res_new = res_ref.to(device=dev, dtype=dt) if res_ref is not None else res
            if kind == "kwargs":
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hs_new
                if "residual" in kwargs:
                    kwargs["residual"] = res_new
                return args, kwargs
            new_args = list(args)
            if len(new_args) > 1:
                new_args[1] = hs_new
            if len(new_args) > 2:
                new_args[2] = res_new
            return tuple(new_args), kwargs
        handles.append(layers[tested_pos].register_forward_pre_hook(
            inject_pre_hook, with_kwargs=True))

    return captured, handles


def _aa_worker_register(worker, layer_start, layer_end, save_input_path, inject_ref):
    """collective_rpc entrypoint (runs inside a TP worker process): register the
    capture hooks and stash state on the worker for a later collect. Only rank 0
    writes the save-input reference (all ranks hold identical hidden states)."""
    is_rank0 = getattr(worker, "rank", 0) == 0
    captured, handles = _aa_capture_layers(
        worker.model_runner.model, layer_start, layer_end,
        save_input_path if is_rank0 else None, inject_ref)
    worker._aa_captured = captured
    worker._aa_handles = handles
    return len(handles)


def _aa_worker_collect(worker):
    """collective_rpc entrypoint: remove hooks and return this rank's captures."""
    for h in getattr(worker, "_aa_handles", []):
        h.remove()
    return getattr(worker, "_aa_captured", {})


def run_partial_layers(
    model_path: str,
    layer_start: int,
    layer_end: int,
    prompt: str,
    output_path: str,
    device: str = "cuda",
    cards: str = "0",
    load_mode: str = "full_model",
) -> None:
    """
    Load model with layers [layer_start, layer_end), run on prompt, save hidden states.

    Args:
        model_path: Path to model checkpoint
        layer_start: First layer to compute (inclusive)
        layer_end: Last layer to compute (exclusive)
        prompt: Input text prompt
        output_path: Where to save hidden states tensor
        device: "cuda" or "xpu"
        cards: Device card IDs (e.g., "0,1")
        load_mode: "full_model" or "partial_layers"

    Raises:
        ValueError: If layer_start >= layer_end or if layer ranges are invalid
    """
    # Validate layer range
    if layer_start >= layer_end:
        raise ValueError(
            f"Invalid layer range: layer_start ({layer_start}) must be < layer_end ({layer_end})"
        )
    if layer_start < 0:
        raise ValueError(f"layer_start must be non-negative, got {layer_start}")
    if layer_end < 0:
        raise ValueError(f"layer_end must be non-negative, got {layer_end}")

    # Set device visibility and calculate tensor parallel size. When invoked via
    # the CLI, _set_device_affinity_from_argv() already set this BEFORE the torch
    # import (which is the only point at which it takes effect); this re-set is a
    # harmless fallback for programmatic callers that import the module first.
    num_cards = len(cards.split(','))
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = cards
    elif device == "xpu":
        os.environ["ZE_AFFINITY_MASK"] = cards

    # Clamp layer construction to the requested window so only these layers
    # allocate weight tensors (read by our patched make_layers() in utils.py).
    # We build layers [0, layer_end) because a correct forward pass from the
    # embeddings must flow through every preceding layer. Set before importing
    # vLLM so the value is inherited by the EngineCore subprocess.
    os.environ["ACCURACY_DEBUG_LAYER_START"] = "0"
    os.environ["ACCURACY_DEBUG_LAYER_END"] = str(layer_end)

    # vLLM v1 runs the model inside a separate EngineCore process by default
    # (SyncMPClient), which makes the in-memory model object unreachable from
    # this driver process. Force the EngineCore in-process so we can walk into
    # model_executor.driver_worker.model_runner.model for manual layer forward.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # Import vLLM after setting environment
    from vllm import LLM
    from transformers import AutoTokenizer

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Inject debug config before model loading
    # This is read by our patches in default_loader.py and llama.py
    import vllm.config
    original_model_config_init = vllm.config.ModelConfig.__init__

    def patched_init(self: "vllm.config.ModelConfig", *args, **kwargs) -> None:  # type: ignore
        original_model_config_init(self, *args, **kwargs)
        # Inject debug flags
        if load_mode == "partial_layers":
            self.hf_config.debug_layer_start = layer_start
            self.hf_config.debug_layer_end = layer_end
        else:
            # Full model mode: load all layers but compute only subset
            # Use actual num_hidden_layers from config instead of 9999
            num_hidden_layers = getattr(
                self.hf_config, 'num_hidden_layers',
                getattr(self.hf_config, 'num_layers', 32)  # fallback
            )
            self.hf_config.debug_layer_start = 0
            self.hf_config.debug_layer_end = num_hidden_layers

    vllm.config.ModelConfig.__init__ = patched_init

    # Load model with vLLM. Memory knobs are env-tunable because a single MoE
    # layer's int4 weights + XPU weight-processing peak can consume the whole
    # gpu_memory_utilization budget on small (e.g. 24GB) cards, leaving nothing
    # for KV cache ("No available memory for the cache blocks"). Raise util
    # and/or shrink max_model_len (smaller profiling peak + KV requirement).
    #   ACCURACY_GPU_MEM_UTIL  (default 0.90)
    #   ACCURACY_MAX_MODEL_LEN (default 1024)
    gpu_mem_util = float(os.environ.get("ACCURACY_GPU_MEM_UTIL", "0.90"))
    max_model_len = int(os.environ.get("ACCURACY_MAX_MODEL_LEN", "1024"))
    print(f"Loading model from {model_path} with mode={load_mode}, "
          f"tensor_parallel_size={num_cards}, gpu_mem_util={gpu_mem_util}, "
          f"max_model_len={max_model_len}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem_util,
        tensor_parallel_size=num_cards,
        max_model_len=max_model_len,  # Small ctx: shrinks KV-cache + profiling peak
        max_num_seqs=1,  # Single-seq profiling → smaller dummy-run peak
        disable_custom_all_reduce=True,  # Avoid multiprocess worker init issues
    )

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print(f"Input shape: {input_ids.shape}")

    # Access the actual model. The path differs between vLLM v0 and v1:
    #   v0: llm.llm_engine.model_executor.driver_worker.model_runner.model
    #   v1: llm.llm_engine.engine_core.engine_core.model_executor
    #           .driver_worker.model_runner.model   (in-process EngineCore)
    def _resolve_worker_or_executor(engine):
        """Return ("worker", in_process_worker) for TP=1, or ("executor", exec)
        for TP>1 (workers are separate processes reached via collective_rpc).

        The EngineCore is forced in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0),
        so the executor object is always reachable here; only the TP workers may
        live in other processes.
        """
        executor = getattr(engine, "model_executor", None)
        if executor is None:
            # v1: engine.engine_core is an InprocClient wrapping the EngineCore.
            core = getattr(engine, "engine_core", None)
            core = getattr(core, "engine_core", core)  # unwrap InprocClient
            executor = getattr(core, "model_executor", None)
        if executor is None:
            raise AttributeError(
                "Could not locate model_executor on the vLLM engine. "
                f"engine attrs: {[a for a in dir(engine) if not a.startswith('__')]}"
            )
        worker = getattr(executor, "driver_worker", None)
        if worker is None:
            workers = getattr(executor, "workers", None)
            worker = workers[0] if workers else None
        worker = getattr(worker, "worker", worker) if worker is not None else None
        # TP=1: the worker is in THIS process and exposes model_runner directly.
        if worker is not None and hasattr(worker, "model_runner"):
            return "worker", worker
        # TP>1: workers are separate processes (WorkerProcHandle). Register and
        # collect hooks INSIDE them via the executor's collective_rpc, which runs
        # our callable on every rank and returns the per-rank results.
        if hasattr(executor, "collective_rpc"):
            return "executor", executor
        raise RuntimeError(
            "No in-process worker and the executor exposes no collective_rpc; "
            "cannot extract hidden states for this TP configuration."
        )

    kind, target = _resolve_worker_or_executor(llm.llm_engine)

    # Optional per-layer input isolation (see _aa_capture_layers): observe the
    # deepest layer's input, or override it with a saved reference.
    save_input_path = os.environ.get("ACCURACY_SAVE_INPUT_PATH")
    inject_input_path = os.environ.get("ACCURACY_INJECT_INPUT_PATH")
    inject_ref = torch.load(inject_input_path) if inject_input_path else None
    reg_args = (layer_start, layer_end, save_input_path, inject_ref)

    # Register capture hooks. TP=1 -> in-process; TP>1 -> ship the registration
    # into every worker process via collective_rpc (cloudpickled).
    local_state = None
    if kind == "worker":
        local_state = _aa_capture_layers(target.model_runner.model, *reg_args)
    else:
        target.collective_rpc(_aa_worker_register, args=reg_args)

    # Run a real 1-token generation so vLLM builds the forward context. Only
    # layers [0, layer_end) are real modules (make_layers clamp); the rest are
    # cheap PPMissingLayer, so the decoder stack stops after the window.
    from vllm import SamplingParams
    print(f"Running engine forward to capture layers [{layer_start}, {layer_end}) "
          f"(TP path: {kind})")
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))

    # Collect captures and drop hooks.
    if kind == "worker":
        captured, handles = local_state
        for h in handles:
            h.remove()
    else:
        per_rank = target.collective_rpc(_aa_worker_collect)
        # Decoder-layer output hidden states are all-reduced across TP ranks, so
        # every rank holds the identical full-width tensor -- take rank 0's.
        captured = next((c for c in per_rank if c), {})

    if not captured:
        raise RuntimeError(
            "No layer outputs captured — forward hooks did not fire. "
            "The decoder stack may not have reached the requested layers."
        )

    # The tool compares the hidden state after the deepest requested layer.
    deepest = max(captured)
    hidden_states = captured[deepest]
    print(f"Captured layers {sorted(captured)}; saving layer {deepest} "
          f"hidden state shape {tuple(hidden_states.shape)} to {output_path}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hidden_states, output_path)
    # Full per-layer dict alongside, for finer-grained bisection.
    torch.save(dict(captured), str(output_path) + ".alllayers")
    print(f"Successfully saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hidden states from vLLM at specific layers"
    )
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--layer-start", type=int, required=True,
                      help="First layer to compute (inclusive)")
    parser.add_argument("--layer-end", type=int, required=True,
                      help="Last layer to compute (exclusive)")
    parser.add_argument("--prompt", type=str, required=True,
                      help="Input text prompt")
    parser.add_argument("--output", type=str, required=True,
                      help="Output path for hidden states tensor")
    parser.add_argument("--device", type=str, default="cuda",
                      choices=["cuda", "xpu"],
                      help="Device type")
    parser.add_argument("--cards", type=str, default="0",
                      help="Device card IDs (comma-separated)")
    parser.add_argument("--load-mode", type=str, default="full_model",
                      choices=["full_model", "partial_layers"],
                      help="Whether to load full model or only partial layers")

    args = parser.parse_args()

    try:
        run_partial_layers(
            model_path=args.model_path,
            layer_start=args.layer_start,
            layer_end=args.layer_end,
            prompt=args.prompt,
            output_path=args.output,
            device=args.device,
            cards=args.cards,
            load_mode=args.load_mode,
        )
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

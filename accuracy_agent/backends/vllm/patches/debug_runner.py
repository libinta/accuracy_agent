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

    # Load model with vLLM
    print(f"Loading model from {model_path} with mode={load_mode}, tensor_parallel_size={num_cards}")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        enforce_eager=True,
        gpu_memory_utilization=0.8,  # Weights (debug window is a few layers) are
        # small; use most of the *free* card (shared card: ~26/30 GiB free, so
        # 0.9 trips the startup check) so KV-cache blocks can be allocated.
        tensor_parallel_size=num_cards,
        max_model_len=2048,  # Small ctx: shrinks KV-cache + profiling peak
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
    def _resolve_model(engine: object) -> object:
        # Find the executor regardless of v0/v1 wrapping.
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
            # UniProcExecutor may expose workers as a list.
            workers = getattr(executor, "workers", None)
            worker = workers[0] if workers else None
        if worker is None:
            raise AttributeError("Could not locate driver_worker on the executor.")
        # worker may be a WorkerWrapperBase; unwrap to the real worker.
        worker = getattr(worker, "worker", worker)
        return worker.model_runner.model

    model = _resolve_model(llm.llm_engine)

    # Locate the decoder layer list regardless of wrapper depth.
    if hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    # Capture hidden states via forward HOOKS during a real engine forward.
    # Manually calling a DeepseekV2DecoderLayer outside the engine fails: MLA
    # attention needs the per-step forward context (attn metadata + KV cache)
    # that only the engine's execute_model sets up. Hooks observe the true
    # forward without our having to reconstruct that context or the layer's
    # (positions, hidden_states, residual) call convention.
    captured = {}

    def make_hook(abs_idx: int):
        def hook(module, args, output):
            # Decoder layers return (hidden_states, residual). The post-layer
            # hidden state is hidden_states + residual (the next layer's
            # input_layernorm / final norm consumes their sum), so store that.
            if isinstance(output, tuple):
                hs = output[0]
                res = output[1] if len(output) > 1 else None
                combined = hs + res if res is not None else hs
            else:
                combined = output
            captured[abs_idx] = combined.detach().to(torch.float32).cpu()
        return hook

    handles = []
    for i in range(layer_start, min(layer_end, len(layers))):
        # The make_layers clamp always builds the window at its NATURAL
        # positions (ACCURACY_DEBUG_LAYER_START is pinned to "0", END to
        # layer_end), so layers[i] is model-layer i in every mode. Do NOT
        # offset by layer_start: that assumed the window was packed at the
        # front of the list, which it is not -- it made every single-layer
        # window [N, N+1) hook layers[0] and capture layer 0's output.
        layer_pos = i
        if layer_pos >= len(layers):
            print(f"Warning: layer index {layer_pos} out of range (max {len(layers)-1})")
            break
        handles.append(layers[layer_pos].register_forward_hook(make_hook(i)))

    # --- Per-layer input injection / reference capture (isolated-forward mode) ---
    # ACCURACY_SAVE_INPUT_PATH: during a normal forward, save the
    #   (hidden_states, residual) pair ENTERING the deepest tested layer. That
    #   pair is the golden reference input for that layer.
    # ACCURACY_INJECT_INPUT_PATH: override the deepest tested layer's input with
    #   a previously-saved reference so the layer's OWN kernel is measured on an
    #   identical input regardless of upstream drift (per-layer isolation).
    # Both act on the deepest built layer (layer_end-1). GLM/deepseek decoder
    # layers are called positionally as forward(positions, hidden_states,
    # residual); we fall back to kwargs by name if that ever changes.
    save_input_path = os.environ.get("ACCURACY_SAVE_INPUT_PATH")
    inject_input_path = os.environ.get("ACCURACY_INJECT_INPUT_PATH")
    tested_pos = min(layer_end, len(layers)) - 1

    def _find_hs_res(args, kwargs):
        """Locate (hidden_states, residual) in a decoder layer's call."""
        if "hidden_states" in kwargs:
            return kwargs.get("hidden_states"), kwargs.get("residual"), "kwargs"
        hs = args[1] if len(args) > 1 else None
        res = args[2] if len(args) > 2 else None
        return hs, res, "args"

    if save_input_path:
        def save_pre_hook(module, args, kwargs):
            hs, res, _ = _find_hs_res(args, kwargs)
            payload = {
                "hidden_states": hs.detach().to(torch.float32).cpu() if hs is not None else None,
                "residual": res.detach().to(torch.float32).cpu() if res is not None else None,
            }
            Path(save_input_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, save_input_path)
            print(f"[inject] saved layer {tested_pos} input reference "
                  f"(hs={None if hs is None else tuple(hs.shape)}, "
                  f"res={None if res is None else tuple(res.shape)}) -> {save_input_path}")
            return None  # observe only; do not modify inputs
        handles.append(layers[tested_pos].register_forward_pre_hook(
            save_pre_hook, with_kwargs=True))

    if inject_input_path:
        _ref = torch.load(inject_input_path)

        def inject_pre_hook(module, args, kwargs):
            hs, res, kind = _find_hs_res(args, kwargs)
            hs_ref = _ref.get("hidden_states")
            res_ref = _ref.get("residual")
            # Reference lives on CPU/f32; move to the live tensor's device+dtype.
            dev = hs.device if hs is not None else (res.device if res is not None else None)
            dt = hs.dtype if hs is not None else (res.dtype if res is not None else None)
            hs_new = hs_ref.to(device=dev, dtype=dt) if hs_ref is not None else hs
            res_new = res_ref.to(device=dev, dtype=dt) if res_ref is not None else res
            print(f"[inject] overriding layer {tested_pos} input with reference "
                  f"(hs={None if hs_new is None else tuple(hs_new.shape)}, "
                  f"res={None if res_new is None else tuple(res_new.shape)})")
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

    # Run a real 1-token generation so vLLM builds the forward context. The
    # model was constructed with only layers [0, layer_end) (the rest are cheap
    # PPMissingLayer via our make_layers clamp), so the decoder stack naturally
    # stops after the requested window.
    from vllm import SamplingParams
    print(f"Running engine forward to capture layers [{layer_start}, {layer_end})")
    llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))

    for h in handles:
        h.remove()

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

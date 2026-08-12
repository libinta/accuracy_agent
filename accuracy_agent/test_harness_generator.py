from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from pathlib import Path

def generate_test_harness(
    config: DebugConfig,
    model_info: ModelInfo,
    layer_start: int,
    layer_end: int,
    platform: str
) -> str:
    """Generate test harness script for GPU or XPU.

    The generated harness runs a forward pass through ONLY the decoder
    layers in the half-open range [layer_start, layer_end) and saves the
    intermediate hidden states after layer ``layer_end - 1`` (NOT the
    final-model logits). This is what makes per-layer bisection work:
    every layer range produces a distinct output, and we never need to run
    the (expensive, memory-hungry) full model / LM head on the XPU.

    Args:
        config: Debug configuration
        model_info: Model architecture info
        layer_start: First layer to run (inclusive)
        layer_end: Last layer to run (exclusive)
        platform: "gpu" or "xpu"

    Returns:
        Python script as string
    """
    device = "cuda" if platform == "gpu" else "xpu"

    script = f'''#!/usr/bin/env python3
"""Test harness for {platform.upper()} - layers {layer_start} to {layer_end}.

Runs a forward pass through the decoder layers in the half-open range
[LAYER_START, LAYER_END) and saves the resulting hidden states. It does
NOT run the full model or the LM head, so the output isolates the numeric
behaviour of exactly those layers.
"""
import torch
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# Configuration
MODEL_PATH = "{config.model_path}"
OUTPUT_PATH = "{config.output_dir}/layer_{layer_start}_{layer_end}_{platform}.pt"
DEVICE = "{device}"
LAYER_START = {layer_start}
LAYER_END = {layer_end}

# Fixed seed for determinism across platforms
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


def load_model():
    """Load the model and its config.

    For the POC we load the full checkpoint but only execute the requested
    layer subset in the forward pass. True shard-level subset loading (see
    design doc "Approach 1") is a later optimization; the correctness-critical
    piece is running only [LAYER_START, LAYER_END) and capturing hidden states.
    """
    print(f"Loading model from {{MODEL_PATH}} (will run layers {{LAYER_START}}-{{LAYER_END}})")
    model_config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True,
    )
    model.eval()
    return model, model_config


def get_decoder(model):
    """Return the inner decoder module that owns `.layers` and `.embed_tokens`.

    Handles the common `AutoModelForCausalLM` -> `.model` nesting used by
    LLaMA / Gemma / Mistral / Qwen etc., and falls back to the model itself.
    """
    inner = model
    # Most CausalLM wrappers expose the backbone at `.model`.
    if hasattr(inner, "model") and hasattr(inner.model, "layers"):
        inner = inner.model
    if not hasattr(inner, "layers"):
        raise AttributeError(
            "Could not locate decoder layers on the model; expected "
            "`model.model.layers` or `model.layers`."
        )
    return inner


def build_position_embeddings(decoder, hidden_states, position_ids):
    """Compute rotary position embeddings (cos, sin) if the model uses them.

    Newer transformers versions compute rotary embeddings once at the model
    level and pass them into every decoder layer via `position_embeddings`.
    Returns None when the model does not expose a top-level rotary module
    (older layers compute rotary internally from `position_ids`).
    """
    rotary = getattr(decoder, "rotary_emb", None)
    if rotary is None:
        return None
    try:
        return rotary(hidden_states, position_ids)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"rotary_emb call failed ({{exc}}); layers will derive rotary from position_ids")
        return None


def run_decoder_layer(layer, hidden_states, attention_mask, position_ids,
                      position_embeddings):
    """Call a single decoder layer, tolerant of transformers signature drift.

    Decoder-layer forward signatures differ across model families and
    transformers releases. We try the richest call first and progressively
    drop kwargs the layer does not accept.
    """
    seq_len = hidden_states.shape[1]
    cache_position = torch.arange(seq_len, device=hidden_states.device)

    attempts = []
    if position_embeddings is not None:
        attempts.append(dict(
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
            cache_position=cache_position,
        ))
    attempts.append(dict(
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        cache_position=cache_position,
    ))
    attempts.append(dict(
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ))
    attempts.append(dict(position_ids=position_ids, use_cache=False))
    attempts.append(dict())

    last_err = None
    for kwargs in attempts:
        try:
            out = layer(hidden_states, **kwargs)
            return out[0] if isinstance(out, tuple) else out
        except TypeError as exc:
            last_err = exc
            continue
    raise RuntimeError(f"Could not invoke decoder layer with any known signature: {{last_err}}")


def run_test():
    """Run forward pass through layers [LAYER_START, LAYER_END) and save hidden states."""
    model, model_config = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    decoder = get_decoder(model)
    layers = decoder.layers
    num_layers = len(layers)

    start = max(0, LAYER_START)
    end = min(LAYER_END, num_layers)
    if start >= end:
        raise ValueError(
            f"Empty layer range after clamping: start={{start}} end={{end}} "
            f"(model has {{num_layers}} layers)"
        )

    # Deterministic test input
    input_text = "The capital of France is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"Input shape: {{tuple(input_ids.shape)}}")

    with torch.no_grad():
        # Embedding lookup -> initial hidden states.
        hidden_states = decoder.embed_tokens(input_ids)

        # Some models (e.g. Gemma) scale embeddings by sqrt(hidden_size).
        normalizer = getattr(decoder, "embed_scale", None)
        if normalizer is not None:
            hidden_states = hidden_states * normalizer

        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        position_embeddings = build_position_embeddings(decoder, hidden_states, position_ids)

        # A single-sequence causal run needs no explicit padding mask; leave the
        # attention mask as None so SDPA/eager builds the causal mask itself.
        attention_mask = None

        # Run ONLY the requested layer subset.
        for i in range(start, end):
            hidden_states = run_decoder_layer(
                layers[i], hidden_states, attention_mask, position_ids,
                position_embeddings,
            )

    hidden_states = hidden_states.contiguous()

    output_data = {{
        "hidden_states": hidden_states.cpu(),
        "input_ids": input_ids.cpu(),
        "layer_start": start,
        "layer_end": end,
        "num_layers": num_layers,
        "platform": "{platform}",
        "device": DEVICE,
    }}

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, OUTPUT_PATH)

    print(f"Saved output to {{OUTPUT_PATH}}")
    print(f"Hidden states shape: {{tuple(hidden_states.shape)}}")
    print(f"Ran layers [{{start}}, {{end}}) of {{num_layers}}")
    print(f"First hidden values: {{hidden_states[0, 0, :10]}}")


if __name__ == "__main__":
    run_test()
'''

    return script

def save_test_harness(script: str, output_path: str) -> None:
    """Save test harness script to file.

    Args:
        script: Generated script content
        output_path: Path to save script
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    path.chmod(0o755)  # Make executable

# FP8-Aware Oracle + Per-Layer Input Injection — Design

**Date:** 2026-08-11
**Component:** accuracy_agent (GPU-vs-XPU layer-wise accuracy comparison for GLM-5.2-FP8)

## Problem

The tool runs a truncated forward on GPU and XPU, captures each tested layer's
hidden state, and compares them. Two gaps make its verdicts untrustworthy:

1. **The pass/fail oracle is miscalibrated.** `comparator.compare_tensors`
   gates on `max_rel_error < 1e-4` (worst *single* element) AND `cosine >
   0.999`. FP8-e4m3 element precision is ~single-digit percent, so two
   correct-but-different FP8 kernels (CUDA vs XPU) can never pass. Every layer
   reads DIVERGE — a false alarm. Observed: layer 0 at cosine 0.99989 is
   flagged divergent purely because one element's relative error exceeds 1e-4.

2. **Per-layer numbers measure cumulative drift, not per-layer error.** Each
   device feeds layer N *its own* layers-0..N-1 output. The two inputs to layer
   N are already ~0.9999-similar but not identical, so a divergence at layer N
   cannot be attributed to layer N's kernel versus drift inherited from
   upstream. There is no way to ask "given the *same* input, does XPU layer N
   compute the same output as GPU layer N?"

This design fixes both: a defensible FP8-aware oracle, and an always-on
input-injection pass that isolates each layer's own contribution.

## Goals

- Replace the oracle with FP8-scaled, percentile-based numeric criteria so a
  PASS/FAIL means something for FP8 tensors.
- For every tested layer, produce two verdicts side by side: **cumulative**
  (real end-to-end drift) and **isolated** (this layer's kernel on a shared
  golden input), and derive a per-layer attribution from the pair.
- Localize a real XPU kernel bug to a specific layer, or exonerate a layer as
  merely inheriting upstream drift.

## Non-Goals / Scope & Limitations

- **Eager only.** The mechanism is Python `nn.Module` hooks (capture) and a
  forward pre-hook (injection). These require `enforce_eager=True`, which the
  harness already sets ([debug_runner.py] `LLM(..., enforce_eager=True)`).
  This tool therefore validates the **eager per-layer math** of each device.
- **Compiled / piecewise-graph mode is out of scope for this build.** vLLM v1
  normally runs piecewise compilation (~one graph per layer, attention at the
  seams). The inter-layer boundary is preserved there, so graph-boundary
  capture is *plausible* later, but it is explicitly deferred (see Future
  Work). Eager localization remains valid for kernel-*math* bugs because the
  heavy GEMM/attention kernels are identical eager vs piecewise; only pointwise
  fusion and cross-piece stream/graph-replay effects differ.
- **No behavioral oracle here.** A full-model token/logit comparison (the
  production-faithful ground truth) needs all 78 layers loaded and is deferred.
- **TP1 only**, unchanged from today (in-process EngineCore forward hook cannot
  reach sharded worker subprocesses).

## Design

### 1. FP8-aware oracle (`comparator.py`)

Introduce fixed, FP8-scaled module constants (hard-coded, not config — a PASS
must mean the same thing across runs):

```
COSINE_FLOOR       = 0.999   # directional agreement
REL_ERR_PERCENTILE = 99      # p99, not worst single element
REL_ERR_TOLERANCE  = 0.02    # 2%, ~FP8-e4m3 element precision
```

`compare_tensors` computes, in addition to the existing `max_rel_error`
(retained for reporting only), a `p99_rel_error` = the 99th-percentile of the
per-element relative-error distribution using the existing denominator
(`abs_diff / (max_val + 1e-10)`). The verdict becomes:

```
match = (cosine_similarity > COSINE_FLOOR) and (p99_rel_error < REL_ERR_TOLERANCE)
```

`ComparisonResult` gains a `p99_rel_error: float` field. `max_rel_error` stays
on the result and in reports (informative, no longer the gate). The adaptive
`effective_threshold` scaling on the old worst-element path is removed.

**Rationale:** dropping the single worst element and scaling to FP8 precision
turns known-good layers (layer 0: cosine 0.99989, p99 well under 2%) into PASS,
while a genuine divergence (cosine 0.9, p99 20%) still FAILs on both criteria.

### 2. Injection architecture (`bisector.py`, `backend.py`)

Per tested layer N, three forward passes yield two comparisons:

1. **GPU real** — window `[0, N+1)`; captures BOTH layer N's output and layer
   N-1's output (the `(hidden_states, residual)` pair entering layer N). Runs
   in parallel with XPU real. Layer N-1's captured pair is the **golden
   reference input**.
2. **XPU real** — window `[0, N+1)`; its own layer N output.
   → **cumulative** = `compare_tensors(GPU_real_N, XPU_real_N)`.
3. **XPU injected** — window `[0, N+1)`; a forward pre-hook on layer N overrides
   its input with GPU's fetched layer N-1 pair.
   → **isolated** = `compare_tensors(GPU_real_N, XPU_injected_N)`.

**GPU real doubles as the isolated golden:** injecting GPU's own layer N-1
output into GPU layer N is a no-op, so GPU_real_N *is* GPU layer N evaluated on
the injected input. No separate GPU injected pass is needed. Added cost = **+1
XPU forward per tested layer**.

**Ordering:** GPU real ∥ XPU real run in parallel. The XPU injected pass depends
on GPU's layer N-1 tensor, so it is serialized after GPU completes and the
reference is fetched to the XPU side (base64 over docker exec — existing
`copy_file_from_container` for fetch, `copy_file_to_container` to ship the
reference into the XPU container).

**Attribution truth table** (verdict per layer):

| isolated | cumulative | verdict           | meaning                              |
|----------|------------|-------------------|--------------------------------------|
| PASS     | PASS       | `clean`           | layer fine, no drift                 |
| PASS     | FAIL       | `inherited_drift` | layer fine; search earlier layers    |
| FAIL     | PASS       | `kernel_bug`*     | layer's own kernel diverges          |
| FAIL     | FAIL       | `kernel_bug`      | layer's own kernel diverges          |

*isolated FAIL is definitive regardless of cumulative — the layer diverges on
identical input.

### 3. Pre-hook injection mechanism (`debug_runner.py`)

- New env `ACCURACY_INJECT_INPUT_PATH`. When set, after the model is resolved,
  register `layers[N].register_forward_pre_hook(hook, with_kwargs=True)` on the
  single tested layer N (the deepest built layer in the window).
- The pre-hook loads the reference file `{hidden_states, residual}` (a dict of
  two tensors), moves both to the layer's current device/dtype, and returns
  overridden `(args, kwargs)` so the decoder layer consumes the reference pair
  instead of its upstream input. GLM/deepseek decoder layers take
  `hidden_states` and `residual`; both are overridden to match the captured
  pair.
- The existing post-hook still captures layer N's output, so one harness serves
  both modes: env unset → real forward; env set → injected forward.
- **Reference format:** the layer N-1 capture saves the *pair*
  `{"hidden_states": ..., "residual": ...}` (not their sum), so injection is
  faithful. The real-run capture path is extended to also persist the entering
  pair for the tested layer when it will be used as a reference.

### 4. Eager-mode invariant (`debug_runner.py`)

Add an explicit guard immediately before/after `LLM(...)` asserting
`enforce_eager` is True in the resolved vLLM config. If ever flipped off (or a
future config exposes it), hook-capture and injection fail loudly rather than
silently producing graph-captured garbage.

### 5. Wiring

- **`config.py`:** add `inject: bool = True` (`test.inject` in YAML). False →
  run real-only (skip the isolated pass) for a cheaper run.
- **`backend.py`:** `run_layer_range(..., inject_input_path: Optional[str] =
  None)`. When set, ship the reference into the container and prepend
  `ACCURACY_INJECT_INPUT_PATH` to the launch env.
- **`bisector.py`:** new dataclass
  `LayerAttribution(name, idx, cumulative: ComparisonResult, isolated:
  Optional[ComparisonResult], verdict: str)`. `bisect_layer_set` builds one per
  representative. `BisectionResult` gains `attributions: List[LayerAttribution]`;
  `divergent_layer` = first rep whose verdict is `kernel_bug`. XPU-only mode is
  unaffected (no GPU peer → no injection; extraction path unchanged).
- **`cli.py`:** results table gains **Cumulative**, **Isolated**, and
  **Verdict** columns (color-coded: `clean` green, `inherited drift` yellow,
  `KERNEL BUG` red).

## Testing

- **`test_comparator.py`:** identical tensors → PASS; FP8-noise perturbation
  (p99 ~1%, cosine 0.9999) → PASS; single huge outlier element but tiny p99 →
  PASS (proves worst-element no longer gates); genuine divergence (cosine 0.9,
  p99 20%) → FAIL.
- **`test_bisector.py`:** feed synthetic (cumulative, isolated)
  `ComparisonResult` pairs → assert truth table yields
  `clean`/`inherited_drift`/`kernel_bug`. Pure logic, no devices.
- **Pre-hook unit test:** tiny 2-layer `nn.Module` stack; register the injection
  pre-hook with a known reference; assert layer N receives the reference, not
  the upstream output.
- **Live end-to-end:** re-run `glm52_gpuxpu_run.yaml` auto mode; expect dense@0
  and moe@3 both `clean` under the new oracle, with isolated verdicts populated.

## Future Work

- **Remote-GPU provisioning & auto-fix:** let the tool set up *and* repair the
  remote GPU environment the way it already does for XPU — verify the container
  / vLLM tree, apply the known patches (or a declarative patch catalog), and
  surface a clear diagnosis when a host is unusable — instead of assuming a
  ready GPU peer.
- **TP>1 extraction:** per-worker capture and reassembly. The in-process
  forward hook cannot see a model sharded across worker subprocesses (true for
  both CUDA and XPU; not the oneCCL issue), so hidden states must be captured
  inside each rank and stitched into the replicated residual stream.
- **torch.compile / graph-boundary (piecewise-compiled) capture:** unblock with
  one experiment — register a layer forward-hook with `enforce_eager=False` and
  check whether it fires and matches eager within FP8 noise. If it fires, expose
  a `compile: piecewise` mode for production-faithful numerics. (vLLM v1 runs
  ~one graph per layer, so the inter-layer seam is preserved.)
- **Behavioral oracle:** full-model token/top-k logit GPU-vs-XPU comparison, to
  catch fusion/stream/graph-replay divergence the eager per-layer path cannot.
- **Deeper/more MoE layers.**

## Progress

- **Per-layer input injection (capture half) — DONE, proven on real GLM-5.2.**
  `debug_runner.py` exposes two env-gated forward-pre-hooks on the deepest built
  layer: `ACCURACY_SAVE_INPUT_PATH` (save the `{hidden_states, residual}` pair
  entering the layer) and `ACCURACY_INJECT_INPUT_PATH` (override that input with
  a saved reference). Self-test on layer 3 (MoE): inject-true-input reproduces
  the real output (cos 1.000001); inject-zeros collapses it (cos 0.0) — proving
  the override fires and drives the layer. Remaining for full verdicts: the
  FP8-aware oracle and the GPU-golden cross-device wiring in `bisector.py`.

# XPU Accuracy Debugger Design

**Date**: 2026-08-10  
**Status**: Design Review  
**Author**: Claude (with youruser)

## Problem Statement

Debugging GPU/XPU accuracy mismatches is currently manual and time-consuming. For models like DeepSeek-V4-Flash (<40% accuracy on XPU) and MinMax (single divergence), we need to:

1. Manually bisect through 40+ layers to find divergence
2. Further bisect to specific modules within layers
3. Compare XPU vs GPU implementations across multiple repos
4. Diagnose root cause (tensor layout, dtype, kernel bug, etc.)
5. Generate and test fixes
6. Handle limited XPU memory (can't load full model)

This process took days for DeepSeek-V4's FP8 block-scale stride bug. We need automation.

## Goals

### Primary Goals

1. **Automated hierarchical bisection**: Layers → single layer → modules → operations
2. **Memory-aware testing**: Load only required layer subsets for small XPU memory
3. **Intelligent fix generation**: Detect and fix common patterns, learn from novel fixes
4. **Full-stack awareness**: Trace through entire XPU stack and generate multi-repo fixes
5. **Remote orchestration**: SSH + docker exec to GPU/XPU machines with shared filesystem

### Non-Goals

- Real-time debugging (batch-oriented workflow is acceptable)
- GUI interface (CLI + generated reports sufficient)
- Supporting non-Intel XPU backends initially
- Fixing compiler/runtime bugs (can detect and workaround, but not patch those repos)

## Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                   Orchestrator (Local)                      │
│  - Config parser (model config.json → layer structure)     │
│  - Bisection engine (hierarchical search)                  │
│  - Remote executor (SSH + docker exec)                     │
│  - Comparison engine (adaptive tolerance)                  │
│  - Fix generator (pattern matching + learning)             │
│  - Report generator (diagnostic + reproduction scripts)    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ├─ SSH ──→ GPU Host
                              │           └─ Docker Container
                              │               └─ Test Harness
                              │
                              └─ SSH ──→ XPU Host
                                          └─ Docker Container
                                              └─ Test Harness
                                          
┌─────────────────────────────────────────────────────────────┐
│              Shared Filesystem (NFS/Lustre)                 │
│  - Model weights                                            │
│  - Test scripts (generated)                                 │
│  - Output tensors (for comparison)                          │
│  - Stack sources (vllm, vllm-xpu-kernels, torch-xpu-ops...) │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                 Knowledge Base (Learning)                   │
│  - Known bug patterns (JSON database)                       │
│  - Fix templates                                            │
│  - User-provided fixes (learned over time)                  │
└─────────────────────────────────────────────────────────────┘
```

### Test Harness (Remote Execution)

**Purpose**: Load model subset, run forward pass, capture outputs

**Key Features**:
- **Layer slicing**: Load only layers N-M from checkpoint
- **Hybrid-aware**: For models like Gemma4, respects sliding window patterns (load 1 full-attention layer + 5 sliding layers)
- **Module instrumentation**: Hook intermediate outputs within a layer
- **Deterministic execution**: Fixed seed, no dropout, greedy decode
- **Output serialization**: Save tensors with metadata (shape, dtype, stride, device)

**Generated per test**: `test_layers_{start}_{end}_{platform}.py`

### Comparison Engine

**Adaptive tolerance algorithm**:

```python
def tensors_match(gpu_tensor, xpu_tensor):
    # Exact match for integer/discrete types
    if is_integer_dtype(gpu_tensor.dtype):
        return torch.equal(gpu_tensor, xpu_tensor)
    
    # Adaptive relative error for float types
    abs_diff = torch.abs(gpu_tensor - xpu_tensor)
    max_val = torch.max(torch.abs(gpu_tensor))
    rel_err = abs_diff / (max_val + 1e-10)
    
    # Multiple metrics
    max_rel_err = torch.max(rel_err).item()
    cosine_sim = F.cosine_similarity(
        gpu_tensor.flatten(), 
        xpu_tensor.flatten(), 
        dim=0
    ).item()
    
    # Magnitude-dependent thresholds
    if max_val < 1e-3:  # Small values
        threshold = 1e-3
    elif max_val < 1.0:
        threshold = 1e-4
    else:  # Normal scale
        threshold = 1e-5
    
    # Pass if both relative error and cosine similarity acceptable
    return max_rel_err < threshold and cosine_sim > (1 - threshold)
```

### Bisection Strategy

**Phase 1: Layer-level bisection**

For non-hybrid models (e.g., standard transformers):
1. Test layers 0-2 (3 layers)
2. If mismatch → test layer 0 alone
3. If layer 0 matches → test layer 1
4. If layer 0 mismatches → go to module-level bisection

For hybrid models (e.g., Gemma4 with sliding window):
1. Parse `config.json` to identify layer patterns
   - Example: `"sliding_window": 1024, "sliding_window_pattern": [5, 1]` = 5 sliding + 1 full
2. Load representative set: 1 full pattern cycle (6 layers for Gemma4)
3. Bisect within pattern-aware boundaries

**Phase 2: Module-level bisection**

Within a problematic layer:
1. Instrument all module boundaries (attn, ffn, norm, etc.)
2. Binary search to find diverging module
3. Within module, instrument sub-operations (qkv_proj, attention kernel, o_proj, etc.)

**Phase 3: Operation-level diagnosis**

Once isolated to specific op (e.g., `XPUFp8BlockScaledMMKernel.apply_block_scaled_mm`):
1. Dump input tensors (values, shape, dtype, stride, is_contiguous)
2. Compare XPU wrapper vs GPU wrapper implementations
3. Check for known bug patterns
4. Generate microtest for isolated op
5. Try systematic fixes

### Stack-Aware Analysis

When a mismatch is found, trace the call stack through repos:

**Example: FP8 GEMM divergence**

```
vllm/models/deepseek_v4/xpu/model.py:DeepseekV4DecoderLayer.forward()
  └─> vllm/models/deepseek_v4/xpu/attention.py:DeepseekV4XPUAttention.forward()
       └─> vllm/model_executor/layers/linear/column.py:ColumnParallelLinear.apply_weights()
            └─> vllm/model_executor/kernels/linear/scaled_mm/xpu.py:XPUFp8BlockScaledMMKernel.apply_block_scaled_mm()
                 └─> torch.ops._xpu_C.fp8_gemm()  [vllm-xpu-kernels/csrc/xpu/onednn/fp8_gemm_w8a8.h]
                      └─> oneDNN primitive [libdnnl.so]
```

**For each layer, check**:
- Does GPU equivalent exist? Compare implementations
- Are tensor layouts compatible? (stride, is_contiguous)
- Are dtypes handled consistently?
- Are there alternative backends? (Triton vs oneDNN vs native SYCL)

### Fix Generator

**Phase 1: Pattern-based fixes** (automatic)

Known patterns stored in `fix_patterns.json`:

```json
{
  "non_contiguous_tensor": {
    "symptom": {
      "input_contiguous": false,
      "kernel_assumes_contiguous": true
    },
    "fix_template": "tensor = tensor.contiguous()",
    "test_priority": 1
  },
  "transpose_stride_mismatch": {
    "symptom": {
      "has_transpose": true,
      "result_contiguous": false,
      "divergence_pattern": "block_structured"
    },
    "fix_template": "tensor = tensor.t().contiguous()",
    "test_priority": 1
  },
  "dtype_mismatch": {
    "symptom": {
      "gpu_dtype": "float32",
      "xpu_dtype": "bfloat16"
    },
    "fix_template": "tensor = tensor.to(dtype=torch.bfloat16)",
    "test_priority": 2
  },
  "missing_xpu_kernel": {
    "symptom": {
      "gpu_has_custom_kernel": true,
      "xpu_falls_through_to_pytorch": true
    },
    "fix_template": "# Use CPU fallback\nwith torch.device('cpu'):\n    result = op(tensor)",
    "test_priority": 3
  }
}
```

**Fix application workflow**:
1. Detect pattern match
2. Generate fix patch
3. Apply patch to test copy
4. Re-run microtest
5. If fixed → verify on full layer test
6. If verified → add to report as "Validated fix"

**Phase 2: Multi-repo fixes**

If pattern indicates issue spans repos (e.g., wrapper in vllm-xpu-kernels calls wrong API in torch-xpu-ops):
1. Generate patches for both repos
2. Apply in dependency order (torch-xpu-ops first)
3. Test with both patches applied
4. Report requires coordinated PR/rebuild

**Phase 3: Learning from user fixes**

When tool can't auto-fix:
1. Generate detailed diagnostic report
2. Provide reproduction microtest
3. User implements fix manually
4. User adds fix to pattern database:

```bash
xpu_accuracy_debug.py learn-fix \
  --bug-id <generated_id> \
  --fix-patch fix.patch \
  --symptom-description "FP8 block scales read with wrong stride" \
  --fix-description "Call .contiguous() on scale tensor before kernel"
```

Tool extracts:
- Code patterns from patch (AST analysis)
- Tensor properties from symptom
- Adds to `fix_patterns.json` for future runs

### Memory Management

**Problem**: XPU has limited VRAM (~32GB/card), can't load full 200B model

**Solution: Layer streaming with inference framework patches**

**Approach 1: Direct model loading** (for transformers-based tests)

```python
def load_layer_subset(model_path, layer_start, layer_end, device):
    """Load only specified layers, remap state dict keys."""
    # Load config to get layer count
    config = AutoConfig.from_pretrained(model_path)
    
    # Create model with only required layers
    # (Modify config.num_hidden_layers temporarily)
    partial_config = copy.deepcopy(config)
    partial_config.num_hidden_layers = layer_end - layer_start
    
    model = AutoModelForCausalLM.from_config(partial_config)
    
    # Load only relevant checkpoint shards
    # (Use safetensors index.json to find which shards have layers N-M)
    index = json.load(open(f"{model_path}/model.safetensors.index.json"))
    required_shards = set()
    for key, shard in index["weight_map"].items():
        if f"model.layers.{layer_start}" <= key < f"model.layers.{layer_end}":
            required_shards.add(shard)
    
    state_dict = {}
    for shard in required_shards:
        shard_dict = safetensors.torch.load_file(f"{model_path}/{shard}")
        # Remap keys: "model.layers.N" -> "model.layers.0" (relative indexing)
        for k, v in shard_dict.items():
            if f"model.layers." in k:
                old_idx = int(k.split(".")[2])
                if layer_start <= old_idx < layer_end:
                    new_idx = old_idx - layer_start
                    new_key = k.replace(f"layers.{old_idx}", f"layers.{new_idx}")
                    state_dict[new_key] = v
    
    model.load_state_dict(state_dict, strict=False)
    return model.to(device)
```

**Approach 2: Patched vLLM/SGLang** (for testing within inference engines)

Since vLLM and SGLang don't natively support loading layer subsets, the tool will:

1. **Generate monkey-patch scripts** for the test harness:

```python
# vllm_layer_subset_patch.py
import vllm.model_executor.models
from vllm.model_executor.models.utils import make_layers

original_make_layers = make_layers

def make_layers_subset(num_hidden_layers, layer_cls, layer_start, layer_end, **kwargs):
    """Override to load only layers [layer_start, layer_end)."""
    # Create only the required layers
    actual_layers = layer_end - layer_start
    layers = original_make_layers(actual_layers, layer_cls, **kwargs)
    
    # Store metadata so forward pass knows to offset layer indices
    layers._debug_layer_offset = layer_start
    return layers

# Monkey-patch vLLM's layer factory
vllm.model_executor.models.utils.make_layers = make_layers_subset
```

2. **Modify model forward pass** to handle offset layers:

```python
# In model.forward(), before layer loop:
if hasattr(self.layers, '_debug_layer_offset'):
    layer_offset = self.layers._debug_layer_offset
    # Adjust position encodings, KV cache indices, etc.
else:
    layer_offset = 0
```

3. **Load only required checkpoint shards** via environment variable:

```python
# Set before vLLM loads model
os.environ["VLLM_DEBUG_LAYER_START"] = str(layer_start)
os.environ["VLLM_DEBUG_LAYER_END"] = str(layer_end)

# vLLM weight loader checks this and skips non-required shards
```

**Fallback: Patched vLLM fork** (if monkey-patching proves fragile)

The tool can maintain a minimal patch to vLLM/SGLang:

```diff
# vllm/model_executor/models/deepseek_v4/model.py
 class DeepseekV4Model(nn.Module):
     def __init__(self, config, ...):
+        layer_start = int(os.getenv("VLLM_DEBUG_LAYER_START", 0))
+        layer_end = int(os.getenv("VLLM_DEBUG_LAYER_END", config.num_hidden_layers))
+        
-        self.layers = make_layers(config.num_hidden_layers, ...)
+        self.layers = make_layers(layer_end - layer_start, ...)
+        self.layers._debug_layer_offset = layer_start
```

Users apply this patch to their vLLM fork for debugging, or the tool auto-applies it to a temporary worktree.

**Hybrid model awareness**:

For Gemma4-style sliding window:
```python
def get_hybrid_layer_subset(config):
    """Return representative layer indices for hybrid models."""
    if "sliding_window_pattern" in config:
        pattern = config["sliding_window_pattern"]  # e.g., [5, 1]
        # Load one full cycle: layers 0-5 for Gemma4 (5 sliding + 1 full)
        return list(range(sum(pattern)))
    else:
        # Standard model: just load first 3 layers
        return [0, 1, 2]
```

### Remote Execution

**SSH orchestration** (uses shared filesystem):

```python
def execute_remote_test(host, container, script_path, output_path):
    """Execute test script in remote container via SSH."""
    # script_path and output_path are on shared FS
    
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no", host,
        f"docker exec {container} python {script_path} --output {output_path}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode != 0:
        raise RuntimeError(f"Remote test failed: {result.stderr}")
    
    # Output tensor file is on shared FS, read directly
    return torch.load(output_path)
```

**Parallel execution** (GPU and XPU tests run simultaneously):

```python
import concurrent.futures

def run_bisection_step(layer_range):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        gpu_future = executor.submit(
            execute_remote_test, 
            gpu_host, gpu_container, 
            f"{shared_fs}/test_layers_{layer_range}_gpu.py",
            f"{shared_fs}/outputs/gpu_layers_{layer_range}.pt"
        )
        xpu_future = executor.submit(
            execute_remote_test,
            xpu_host, xpu_container,
            f"{shared_fs}/test_layers_{layer_range}_xpu.py", 
            f"{shared_fs}/outputs/xpu_layers_{layer_range}.pt"
        )
        
        gpu_output = gpu_future.result()
        xpu_output = xpu_future.result()
        
        return compare_outputs(gpu_output, xpu_output)
```

### Stack Comparison Engine

**Purpose**: Compare implementations across the full XPU software stack

**Inputs**:
- Path to each stack component repo (vllm, vllm-xpu-kernels, torch-xpu-ops, etc.)
- Diverging operation identified by bisection (e.g., `fp8_gemm`)

**Process**:

1. **Locate implementations**:
   ```python
   def find_implementation(op_name, repos):
       """Search for op_name across all repos."""
       results = {}
       for repo_name, repo_path in repos.items():
           matches = grep_recursive(repo_path, f"def {op_name}|void {op_name}")
           results[repo_name] = matches
       return results
   ```

2. **Compare XPU vs GPU**:
   - If both platforms implemented in same repo (e.g., `vllm/model_executor/kernels/linear/scaled_mm/{xpu,deep_gemm}.py`):
     - Diff the implementations
     - Highlight differences in tensor handling, dtype conversion, kernel calls
   
   - If XPU in different repo (e.g., torch-xpu-ops vs pytorch):
     - Compare API signatures
     - Check for parameter mapping issues

3. **Identify alternative backends**:
   ```python
   def find_alternative_implementations(op_name, xpu_repo_path):
       """Find if op has multiple backend implementations."""
       # Example: fp8_gemm might have triton, onednn, native SYCL
       backends = []
       
       # Check for compile-time or runtime backend selection
       config_files = [
           f"{xpu_repo_path}/csrc/CMakeLists.txt",
           f"{xpu_repo_path}/vllm/platforms/xpu.py"
       ]
       
       for f in config_files:
           if "USE_TRITON" in read_file(f):
               backends.append("triton")
           if "USE_ONEDNN" in read_file(f):
               backends.append("onednn")
       
       return backends
   ```

4. **Generate backend swap tests**:
   If op has multiple backends, automatically generate tests with each:
   ```python
   # Test 1: Default backend (failed)
   # Test 2: Force oneDNN backend
   os.environ["VLLM_XPU_USE_TRITON"] = "0"
   os.environ["VLLM_XPU_USE_ONEDNN"] = "1"
   
   # Test 3: Force CPU fallback
   # ... etc
   ```

### Diagnostic Report Format

Generated markdown report: `diagnostics_{model}_{timestamp}.md`

```markdown
# XPU Accuracy Diagnostic Report

**Model**: deepseek-ai/DeepSeek-V4-Flash  
**Date**: 2026-08-10T14:23:45Z  
**Status**: ✅ Fix validated / ❌ Manual investigation required

## Summary

Divergence detected in **layer 0, module `DeepseekV4XPUAttention`, operation `XPUFp8BlockScaledMMKernel.apply_block_scaled_mm`**

- **Symptom**: Output cosine similarity 0.871, relative L2 error 0.254
- **Root cause**: Non-contiguous weight scale tensor passed to kernel expecting contiguous layout
- **Fix**: Apply `.contiguous()` to scale tensor before kernel call
- **Validation**: ✅ Fix applied, divergence resolved (cosine sim 1.0000, rel L2 0.0014)

## Bisection Trace

1. Layers 0-2: **DIVERGE** (cos 0.920)
2. Layer 0 alone: **DIVERGE** (cos 0.920)
3. Layer 0, modules:
   - Embedding: MATCH (cos 1.0000)
   - Attention input: MATCH (cos 1.0000)
   - **Attention output: DIVERGE (cos 0.850)** ← Problem here
   - FFN: (not tested, upstream divergence)
4. Attention sub-operations:
   - qkv_proj (fused_wqa_wkv): DIVERGE (cos 0.968)
   - **wq_b projection: DIVERGE (cos 0.891)** ← Isolated

## Stack Trace

```
vllm/models/deepseek_v4/xpu/attention.py:132
  DeepseekV4XPUAttention.forward()
    └─> vllm/model_executor/layers/linear/column.py:89
         ColumnParallelLinear.apply_weights()
           └─> vllm/model_executor/kernels/linear/scaled_mm/xpu.py:147
                XPUFp8BlockScaledMMKernel.apply_block_scaled_mm()
                  └─> torch.ops._xpu_C.fp8_gemm()
                       [vllm-xpu-kernels/csrc/xpu/onednn/fp8_gemm_w8a8.h:12]
```

## Implementation Comparison

### XPU (vllm-xpu-kernels)

```python
# vllm/model_executor/kernels/linear/scaled_mm/xpu.py:147
def apply_block_scaled_mm(A, B, As, Bs, out_dtype):
    return torch.ops._xpu_C.fp8_gemm(
        A, B.t(), out_dtype, As, Bs.t(), ...  # ← Bs.t() non-contiguous!
    )
```

### GPU Reference (vllm)

```python
# vllm/model_executor/kernels/linear/scaled_mm/deep_gemm.py:68
def apply_block_scaled_mm(A, B, As, Bs, out_dtype):
    # DeepGEMM handles non-contiguous scales correctly
    return torch.ops.vllm.fp8_block_scaled_mm(A, B, As, Bs, out_dtype)
```

**Difference**: XPU oneDNN kernel reads scale tensor **assuming contiguous layout**, but receives transposed view with non-contiguous stride.

## Tensor Properties (at failure point)

| Tensor | Shape | Dtype | Stride | Contiguous | Device |
|--------|-------|-------|--------|------------|--------|
| `Bs` (GPU) | [128, 32] | float32 | [32, 1] | ✅ Yes | cuda:0 |
| `Bs` (XPU) | [128, 32] | float32 | [32, 1] | ✅ Yes | xpu:0 |
| `Bs.t()` (GPU) | [32, 128] | float32 | [1, 32] | ❌ No | cuda:0 |
| `Bs.t()` (XPU) | [32, 128] | float32 | [1, 32] | ❌ No | xpu:0 |

**Analysis**: Both platforms receive same non-contiguous tensor, but GPU kernel respects stride while XPU kernel does not.

## Validated Fix

**File**: `vllm/model_executor/kernels/linear/scaled_mm/xpu.py`

**Patch**:
```diff
@@ -147,7 +147,7 @@ def apply_block_scaled_mm(A, B, As, Bs, out_dtype):
     return torch.ops._xpu_C.fp8_gemm(
         A, 
         B.t(), 
         out_dtype, 
         As, 
-        Bs.t(),
+        Bs.t().contiguous(),  # Fix: Ensure contiguous for oneDNN
         ...
     )
```

**Validation**:
- Microtest (isolated op): ✅ PASS (cos 1.0000, rel L2 0.0014)
- Layer 0 test: ✅ PASS (cos 1.0000)
- Layers 0-2 test: ✅ PASS (cos 1.0000)
- Full model test: ✅ "capital of France" → " Paris." (was " called")

## Alternative Fixes Considered

1. **Fix in kernel** (vllm-xpu-kernels C++ code):
   - Modify `fp8_gemm_w8a8.h` to respect stride when reading scales
   - **Pros**: More robust, handles all non-contiguous cases
   - **Cons**: Requires C++ rebuild, more complex change
   - **Status**: Not implemented (wrapper fix sufficient)

2. **Fix at model loading** (process_weights_after_loading):
   - Store scales as contiguous from the start
   - **Pros**: Zero runtime overhead
   - **Cons**: Invasive change, affects all models
   - **Status**: Not implemented

## Reproduction Script

Standalone microtest saved to: `microtest_fp8_gemm_stride.py`

```python
# Run on GPU and XPU to reproduce divergence
import torch

# ... (full reproduction code)
```

## Learned Pattern

This bug has been added to the pattern database for future automatic detection and fixing.

**Pattern**: `transpose_stride_mismatch_fp8_scales`

## Next Steps

1. ✅ Apply patch to vllm-xpu-kernels fork
2. ✅ Test on DeepSeek-V4-Flash full model
3. ✅ Verify on other FP8 models (if applicable)
4. Submit PR to upstream vllm-xpu-kernels
```

## User Interface

### CLI

```bash
# Basic usage
xpu_accuracy_debug.py \
  --model /mnt/weka/deepseek-ai/DeepSeek-V4-Flash \
  --gpu-host gpu-host.example.com \
  --gpu-docker your_gpu_container \
  --xpu-host xpu-host.example.com \
  --xpu-docker your_xpu_container \
  --shared-fs /mnt/weka \
  --stack-repos repos_config.yaml

# With custom test input
xpu_accuracy_debug.py ... \
  --input-text "The capital of France is" \
  --input-ids 1,2,3,4,5  # or token IDs

# Specify layer range (for large models)
xpu_accuracy_debug.py ... \
  --layer-start 0 \
  --layer-end 10

# Enable learning from previous fixes
xpu_accuracy_debug.py ... \
  --pattern-db ~/.xpu_debug/fix_patterns.json

# Dry-run (generate scripts only, don't execute)
xpu_accuracy_debug.py ... --dry-run

# Learn from user fix
xpu_accuracy_debug.py learn-fix \
  --bug-id deepseek_v4_fp8_stride_20260807 \
  --fix-patch fix.patch \
  --symptom "Non-contiguous FP8 block scales" \
  --description "Apply .contiguous() before oneDNN kernel"
```

### Config File: `repos_config.yaml`

```yaml
# Stack component repositories
repos:
  vllm: /home/youruser/github/vllm
  vllm-xpu-kernels: /home/youruser/github/vllm-xpu-kernels
  torch-xpu-ops: /home/youruser/github/torch-xpu-ops
  triton: /home/youruser/github/triton
  intel-xpu-backend-for-triton: /home/youruser/github/intel-xpu-backend-for-triton
  pytorch: /home/youruser/github/pytorch
  transformers: /home/youruser/github/transformers
  
# Optional: Per-repo branches for testing
branches:
  vllm: dev291
  vllm-xpu-kernels: main
  
# Backend preferences (for swap testing)
backends:
  fp8_gemm: [onednn, triton, cpu_fallback]
  attention: [triton, native_sycl, cpu_fallback]
```

## Implementation Plan (High-Level)

### Phase 1: Core Bisection Engine (Week 1)

- Config parser (model config.json → layer structure)
- Layer-level bisection logic
- Remote test execution (SSH + docker)
- Adaptive comparison engine
- Basic report generation

### Phase 2: Memory Management & Hybrid Models (Week 1)

- Layer subset loading (safetensors streaming)
- Hybrid model pattern detection (sliding window, MoE, etc.)
- Module-level bisection within layers

### Phase 3: Stack-Aware Analysis (Week 2)

- Multi-repo code search
- Implementation comparison (XPU vs GPU)
- Call stack tracing
- Alternative backend detection

### Phase 4: Fix Generation (Week 2-3)

- Pattern database schema
- Known pattern detection (contiguous, dtype, transpose, etc.)
- Fix template application
- Microtest generation and validation
- Multi-repo patch generation

### Phase 5: Learning System (Week 3)

- User fix ingestion (`learn-fix` command)
- Pattern extraction from patches (AST analysis)
- Pattern database updates
- Fix confidence scoring

### Phase 6: Testing & Refinement (Week 4)

- Test on known bugs (DeepSeek-V4 FP8, MinMax)
- Test on new models
- Performance optimization (parallel execution, caching)
- Documentation and examples

## Dependencies

### Python Packages

```
torch >= 2.1
transformers >= 4.36
safetensors >= 0.4
paramiko  # for SSH
pyyaml
click  # CLI framework
rich  # pretty terminal output
```

### External Tools

- SSH access to GPU/XPU hosts
- Docker on remote hosts
- Shared filesystem (NFS/Lustre/Weka)
- Git (for multi-repo analysis)

### Stack Components

All repos listed in user's stack (vllm, vllm-xpu-kernels, torch-xpu-ops, etc.) should be cloned locally for code analysis. They don't need to be built unless the tool needs to apply and test patches.

## Open Questions

1. **Determinism**: Should we enforce strict determinism (fixed seed, no dropout) or support non-deterministic tests with statistical comparison?

2. **Precision**: What's the right default tolerance for "match"? Should it be configurable per-model or per-dtype?

3. **Scalability**: For 400B+ models, even layer subsets might not fit. Should we support parameter-sharding (TP/PP) in tests?

4. **Rebuild automation**: When fix requires C++ kernel changes, should tool automatically rebuild and test, or just report "requires rebuild"?

5. **Cross-model learning**: Can patterns from DeepSeek-V4 apply to LLaMA-3? Should pattern DB include model family metadata?

## Success Criteria

1. **Automated detection**: Tool can bisect DeepSeek-V4 to the exact FP8 GEMM operation without manual intervention
2. **Fix validation**: Tool correctly identifies the `.contiguous()` fix and validates it
3. **Novel model support**: Tool can debug MinMax accuracy issue with minimal user guidance
4. **Learning**: After user fixes MinMax bug once, tool can auto-fix similar patterns in future models
5. **Performance**: Full bisection + fix validation completes in <4 hours for 40-layer model
6. **Multi-repo fixes**: Tool can generate coordinated patches across vllm + vllm-xpu-kernels

## Future Enhancements (Beyond MVP)

- **GUI dashboard**: Web UI for monitoring bisection progress, visualizing tensor diffs
- **CI/CD integration**: Automated accuracy regression testing on model updates
- **Performance debugging**: Extend to diagnose performance regressions (TTFT, throughput), not just accuracy
- **Multi-backend**: Support AMD, NVIDIA (for cross-platform validation)
- **Crowd-sourced patterns**: Community-contributed fix patterns with voting/validation

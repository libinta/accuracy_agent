# POC Validation Guide

## Objective

Validate that the accuracy debugger can automatically find the known FP8 GEMM stride bug in DeepSeek-V4-Flash layer 0.

## Prerequisites

1. SSH access to both hosts:
   - GPU: gpu-host.example.com
   - XPU: xpu-host.example.com

2. Docker containers running:
   - GPU: your_gpu_container
   - XPU: your_xpu_container

3. Model available on shared filesystem:
   - Path: /mnt/weka/data/pytorch/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1

## Test Procedure

### 1. Install accuracy_agent

```bash
cd ~/github/accuracy_agent
pip install -e .
```

### 2. Verify SSH connectivity

```bash
# Test GPU host
ssh gpu-host.example.com "docker exec your_gpu_container echo 'GPU OK'"

# Test XPU host
ssh xpu-host.example.com "docker exec your_xpu_container echo 'XPU OK'"
```

Expected: Both commands print "OK"

### 3. Run bisection

```bash
accuracy-debug --config examples/flash_tp2_config.yaml
```

Expected output:
```
XPU Accuracy Debugger POC

Configuration:
┌────────────┬─────────────────────────────────────┐
│ Parameter  │ Value                               │
├────────────┼─────────────────────────────────────┤
│ Model      │ /mnt/weka/.../DeepSeek-V4-Flash     │
│ GPU        │ gpu-host.example.com / your_gpu_container      │
│ XPU        │ xpu-host.example.com / your_xpu_container       │
│ Layers     │ 0-6                                 │
│ Output     │ /mnt/weka/accuracy_debug_output     │
└────────────┴─────────────────────────────────────┘

Loading model config...
✓ Model: 61 layers, standard architecture

============================================================
Bisecting layers 0-6
============================================================

Testing layers 0-6...
[GPU] Executing on gpu-host.example.com: docker exec your_gpu_container python /mnt/weka/accuracy_debug_output/test_gpu_0_6.py
[GPU] ✓ Success
[XPU] Executing on xpu-host.example.com: docker exec your_xpu_container python /mnt/weka/accuracy_debug_output/test_xpu_0_6.py
[XPU] ✓ Success
✗ Layers 0-6: DIVERGE (cos=0.920000, rel_err=0.254000, abs_err=0.123000)

Testing layers 0-1...
[GPU] ✓ Success
[XPU] ✓ Success
✗ Layer 0: DIVERGE (cos=0.920000, rel_err=0.254000, abs_err=0.123000)

============================================================
Bisection Results
============================================================

✗ Divergence found in layer 0

Divergence in layer 0

Layer Comparisons:
┌─────────────┬───────────┬─────────────┬───────────┐
│ Layer Range │ Status    │ Cosine Sim  │ Rel Error │
├─────────────┼───────────┼─────────────┼───────────┤
│ Layer 0     │ ✗ Diverge │ 0.920000    │ 0.254000  │
└─────────────┴───────────┴─────────────┴───────────┘
```

### 4. Verify output files

```bash
ls -lh /mnt/weka/accuracy_debug_output/
```

Expected files:
- test_gpu_0_6.py
- test_xpu_0_6.py
- layer_0_6_gpu.pt
- layer_0_6_xpu.pt
- test_gpu_0_1.py
- test_xpu_0_1.py
- layer_0_1_gpu.pt
- layer_0_1_xpu.pt

### 5. Manual verification

Load and inspect the output tensors:

```python
import torch

gpu_out = torch.load("/mnt/weka/accuracy_debug_output/layer_0_1_gpu.pt")
xpu_out = torch.load("/mnt/weka/accuracy_debug_output/layer_0_1_xpu.pt")

print("GPU hidden states shape:", gpu_out["hidden_states"].shape)
print("XPU hidden states shape:", xpu_out["hidden_states"].shape)
print("First 10 GPU values:", gpu_out["hidden_states"][0, 0, :10])
print("First 10 XPU values:", xpu_out["hidden_states"][0, 0, :10])
```

## Success Criteria

- [ ] Tool completes without errors
- [ ] Identifies divergence in layer 0
- [ ] GPU and XPU outputs are actually different (not false positive)
- [ ] Process completes in <10 minutes for 6 layers

## Troubleshooting

### SSH connection fails

Check SSH key setup:
```bash
ssh-add -l
```

If no keys, add yours:
```bash
ssh-add ~/.ssh/id_rsa
```

### Docker container not found

List running containers:
```bash
ssh <host> "docker ps"
```

### Model not found

Check model path exists:
```bash
ls /mnt/weka/data/pytorch/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/
```

### Out of memory

Reduce layer range in config:
```yaml
test:
  layer_start: 0
  layer_end: 1  # Test single layer
```

## Next Steps After Validation

If POC successfully finds layer 0 divergence:

1. Add module-level bisection to drill into attention/FFN
2. Add pattern detection for FP8 stride bugs
3. Implement memory-efficient layer loading
4. Add fix generation and validation

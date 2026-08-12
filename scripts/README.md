# scripts/README.md

## Manual Testing Scripts

### test_vllm_backend.py

Test vLLM backend on real hardware with GLM-5.2-FP8 or other models.

**Prerequisites:**
- SSH access to GPU and XPU hosts
- Docker containers running on both hosts
- vLLM installed in containers
- Model accessible from both containers via shared filesystem

**Usage:**

```bash
# Test both GPU and XPU
python scripts/test_vllm_backend.py examples/glm52_vllm_config.yaml

# Test only GPU (for debugging)
python scripts/test_vllm_backend.py examples/glm52_vllm_config.yaml --skip-xpu
```

**What it does:**
1. Loads config from YAML
2. Creates GPU backend and applies vLLM patches via SSH
3. Runs layer extraction (layers 0-3 by default)
4. Creates XPU backend and applies patches
5. Runs same layer extraction on XPU
6. Compares hidden states using adaptive tolerance
7. Cleans up patches on both systems

**Expected output:**
```
2026-08-11 10:00:00 - __main__ - INFO - Loading config from examples/glm52_vllm_config.yaml
2026-08-11 10:00:00 - __main__ - INFO - Creating GPU backend
2026-08-11 10:00:01 - __main__ - INFO - Setting up GPU backend (applying patches)
2026-08-11 10:00:02 - __main__ - INFO - Running layers [0, 3) on GPU
2026-08-11 10:05:00 - __main__ - INFO - GPU hidden states shape: torch.Size([1, 512, 4096])
...
2026-08-11 10:10:00 - __main__ - INFO - Match: True
2026-08-11 10:10:00 - __main__ - INFO - Cosine similarity: 0.999950
2026-08-11 10:10:00 - __main__ - INFO - ✓ GPU and XPU outputs match!
```

**Troubleshooting:**

- **SSH connection failed**: Check host reachability and SSH keys
- **vLLM not found**: Verify vllm_path in config matches container
- **Patch application failed**: Check vLLM version compatibility
- **Model loading failed**: Verify model path is on shared filesystem

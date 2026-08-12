# vLLM Backend Examples

This directory contains example configurations for using the vLLM backend.

## GLM-5.2-FP8 Example

**File:** `glm52_vllm_config_template.yaml`

Template configuration for testing models using vLLM backend with automatic memory checking:
- **Backend**: vLLM with adaptive full/partial loading
- **Parallelization**: GPU and XPU run concurrently (2x speedup)
- **Flexible Deployment**: Run from control host OR inside one of the docker containers

### Usage

This tool supports two deployment modes:

1. **From control host**: SSH to both GPU and XPU docker containers
2. **From inside docker**: Run from inside GPU or XPU docker (uses local execution for that backend, SSH for the other)

1. Copy the template and fill in your configuration:
```bash
cp examples/glm52_vllm_config_template.yaml examples/my_config.yaml
# Edit my_config.yaml with your hosts, containers, and SSH keys
```

2. Run from the control host:
```bash
python3 -m accuracy_agent.cli --config examples/my_config.yaml
```

### Memory Modes

The vLLM backend automatically detects available memory:

- **Full mode**: Loads complete model when memory sufficient
  - Faster (no partial loading overhead)
  - Uses forward hooks to extract hidden states
  
- **Partial mode**: Loads only requested layers when memory constrained
  - Memory efficient for large models
  - Applies vLLM patches for layer subsetting

Memory check happens once at setup, not per layer range.

### Expected Output

```
Setting up GPU backend...
Checking memory availability...
Memory mode: full (required=32.5GB, available=80.2GB)
✓ GPU backend ready

Setting up XPU backend...
Checking memory availability...
Memory mode: partial (required=32.5GB, available=24.1GB)
✓ XPU backend ready

Testing layers [0, 3) in parallel...
✓ Layers 0-3: cosine_similarity=0.9998, max_rel_error=0.0001
```

### Hardware Requirements

- Shared filesystem mounted at `/mnt/weka` on both hosts
- SSH key-based authentication between control node and remote hosts
- Docker containers must be running
- vLLM installed at specified paths in containers

### Troubleshooting

**Memory check fails:**
- Backend defaults to partial mode (conservative)
- Check `nvidia-smi` / `xpu-smi` commands work in containers

**Patches fail to apply:**
- Check vllm_path is correct
- Ensure user has write permissions
- Original files are backed up as `*.accuracy_debug.bak`

**SSH connection fails:**
- Verify SSH key authentication works: `ssh user@host`
- Check docker container is running: `docker ps`

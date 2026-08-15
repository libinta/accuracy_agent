# XPU Accuracy Debugger

Automated GPU/XPU accuracy bisection tool for finding divergences in LLM inference.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
accuracy-debug \
  --model /mnt/weka/models/DeepSeek-V4-Flash \
  --gpu-host gpu-host.example.com \
  --gpu-docker your_gpu_container \
  --xpu-host xpu-host.example.com \
  --xpu-docker your_xpu_container \
  --shared-fs /mnt/weka \
  --layer-start 0 \
  --layer-end 3
```

If the XPU docker runs on the same machine as the tool, the GPU side can be
omitted: the tool asks that container which exact vLLM **commit** it runs, then
installs that commit into the NVIDIA PyTorch image (`nvcr.io/nvidia/pytorch`) and
uses the result as the GPU peer — so both sides run the same vLLM code.

```bash
accuracy-debug \
  --backend vllm \
  --model /mnt/weka/models/DeepSeek-V4-Flash \
  --xpu-docker your_xpu_container \
  --shared-fs /mnt/weka
```

The commit is resolved in a local `vllm-project/vllm` clone (`~/vllm` by default,
`--vllm-repo` to override) and the built image is cached, so later runs with the
same commit start in seconds. See `examples/local_xpu_auto_gpu.yaml` and the
"Automatic GPU Peer From the XPU Container's Commit" section of
`examples/README_vllm.md`.

To pin the commit instead of detecting it, both peers can be built from one you
name — installed from source into the vendor PyTorch images
(`nvcr.io/nvidia/pytorch` and `intel/intel-extension-for-pytorch`), so the two
sides differ only in device:

```bash
accuracy-debug \
  --backend vllm \
  --model /mnt/weka/models/DeepSeek-V4-Flash \
  --vllm-commit 7794b1e08bf505ff28664515ffaaeeec955ab796
```

CUDA kernels come from the nearest nightly wheel by default (minutes); add
`--build-kernels` to compile them at that commit (1–2 h) when it touches
C++/CUDA. See `examples/vllm_commit_config.yaml` and the "Testing One vLLM
Commit on Both Devices" section of `examples/README_vllm.md`.

## Status

**Phase 1 POC**: Core bisection engine + DeepSeek-V4-Flash validation

## POC Validation

**Status**: ✅ Core bisection engine implemented and tested

**Components**:
- [x] Config parser and validation
- [x] Model info loader
- [x] Adaptive tensor comparator
- [x] Test harness generator
- [x] Remote SSH executor
- [x] Hierarchical bisection engine
- [x] CLI interface

**Test Coverage**:
- Unit tests: 14 tests across all components
- Integration tests: 2 tests (end-to-end bisection + README validation)
- Manual validation: Pending on real GPU/XPU hosts

## Next Steps

1. **Manual validation** on DeepSeek-V4-Flash:
   ```bash
   accuracy-debug --config examples/flash_tp2_config.yaml
   ```

2. **Verify it finds the known FP8 GEMM bug** in layer 0

3. **Phase 2**: Add module-level bisection within divergent layers

4. **Phase 3**: Add pattern-based fix generation

## Known Limitations (POC)

- Test harness loads the full checkpoint but runs only the requested layer subset (`[LAYER_START, LAYER_END)`) and compares intermediate hidden states; true shard-level subset loading for memory efficiency is still TODO
- No module-level bisection yet (stops at layer level)
- No fix generation (detection only)
- No vLLM/SGLang integration (uses transformers directly)

## Troubleshooting

### "Connection refused" error

Ensure SSH keys are set up for password-less login:
```bash
ssh-copy-id user@host
```

### "No such file or directory" for model

Check the model path in your config file matches the actual location on the shared filesystem.

### "Docker command not found"

The user account needs permission to run docker commands. Contact your admin or add user to docker group:
```bash
sudo usermod -aG docker $USER
```

### Execution timeout

Increase timeout in code or reduce layer count for initial testing.

## Debug Mode

Run with verbose output:
```bash
export ACCURACY_DEBUG=1
accuracy-debug --config examples/flash_tp2_config.yaml
```

## vLLM Backend Usage

The vLLM backend enables layer-wise accuracy testing using vLLM inference engine.

**Deployment Options:**
1. **From control host**: SSH to both GPU and XPU docker containers
2. **From inside docker**: Run from inside GPU or XPU docker (local execution + SSH to the other)

```bash
# Copy and customize the template
cp examples/glm52_vllm_config_template.yaml examples/my_config.yaml
# Edit my_config.yaml with your hosts and SSH keys

# Run from control host
python3 -m accuracy_agent.cli --config examples/my_config.yaml
```

Features:
- **Adaptive memory**: Auto-detects full vs partial loading based on available memory
- **Parallel execution**: GPU and XPU run concurrently for 2x speedup
- **Automatic patching**: vLLM source patched/restored automatically

See `examples/README_vllm.md` for details.

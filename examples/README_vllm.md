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

### Automatic GPU Docker Selection

**File:** `local_xpu_auto_gpu.yaml`

A GPU-vs-XPU comparison is only meaningful when both sides run the **same vLLM
version** -- otherwise a "divergence" may just be a version difference. When the
XPU docker runs on the same machine as the tool, you no longer have to look that
version up: leave `gpu.docker` unset and the tool will

1. query the XPU container for its vLLM version
   (e.g. `0.26.1rc1.dev353+g7794b1e08.xpu`),
2. reduce it to a released version and pick the matching `vllm/vllm-openai`
   **release** tag -- nightly and dev tags are never candidates. A dev/rc build
   predates its own version number, so it maps to the closest *published*
   release (`0.26.1rc1.dev…` -> `v0.26.0`). Tags are verified against Docker Hub
   when reachable, and constructed from the version offline otherwise,
3. `docker pull` that image and start an idle container from it (reusing it on
   later runs), and
4. detect the vLLM path inside it, so patching works with no further config.

```bash
python3 -m accuracy_agent.cli --config examples/local_xpu_auto_gpu.yaml
# or without a config file:
python3 -m accuracy_agent.cli --backend vllm \
  --model /mnt/weka/models/GLM-5.2-FP8 --xpu-docker your_xpu_container
```

Controls (`gpu:` section, or CLI):

| Setting | CLI | Effect |
|---|---|---|
| `gpu.docker` | `--gpu-docker` | Explicit container; always wins, no automation |
| `gpu.image` | `--gpu-image` | Pin the image, skip version matching |
| `gpu.auto_image: false` | `--no-auto-gpu-image` | Disable the automation |
| `gpu.container_name` | -- | Name for the auto-launched container (default `accuracy_agent_gpu_<tag>`) |
| `gpu.docker_run_args` | -- | Extra raw `docker run` flags |

The automation skips itself (logging why, run continues) when the XPU docker is
on a remote host, when there is no docker CLI, or when the machine has no NVIDIA
GPU -- an XPU-only box keeps doing XPU-only extraction as before.

The launched container is left running so repeated runs skip the multi-GB pull.
Remove it with `docker rm -f accuracy_agent_gpu_<tag>`.

### Testing One vLLM Commit on Both Devices

**File:** `vllm_commit_config.yaml`

Release matching answers "compare against the closest release". To ask "does
commit `<sha>` diverge?" there is no release image to match, so `vllm.commit`
builds both peers from that commit instead -- installed from source into the two
vendors' PyTorch images, so the peers differ only in device:

```bash
python3 -m accuracy_agent.cli --backend vllm \
  --model /mnt/weka/models/GLM-5.2-FP8 \
  --vllm-commit 7794b1e08bf505ff28664515ffaaeeec955ab796
```

What happens:

1. the commit is resolved in a local `vllm-project/vllm` clone (`~/vllm` by
   default, `--vllm-repo` to override), fetched from `origin` if the clone does
   not have it. Short shas, tags and branches all work.
2. it is exported to a per-commit checkout under `vllm.build_root` -- a real git
   checkout, because vLLM's build reads git metadata,
3. per device, a container is started from the vendor base image -- newest
   `nvcr.io/nvidia/pytorch:NN.NN-py3` for CUDA, newest
   `intel/intel-extension-for-pytorch:X.Y.Z-xpu` for XPU (both resolved from the
   registry, with pinned fallbacks offline) -- and the checkout is installed
   **editable** into it,
4. the result is `docker commit`ed to `accuracy_agent/vllm:<device>-<sha12>`, so
   the next run with that commit starts in seconds.

Because the install is editable and the checkout is bind-mounted at
`/workspace/vllm`, the patcher rewrites the same files vLLM imports -- no
reinstall between patches.

**CUDA kernels.** `pip install -e .` from source compiles for 1-2 hours, so the
default is vLLM's precompiled fast path (`VLLM_USE_PRECOMPILED=1`): the commit's
Python code with binaries from the nearest nightly wheel. That is right for
bisecting Python-level divergence and **wrong if the commit touches C++/CUDA** --
pass `--build-kernels` for a real compile. The two are cached under different
image tags and every report says which one produced the peer, so a run can never
silently claim kernel coverage it does not have. The XPU side always builds from
source; its heavy kernels come from the pinned `vllm_xpu_kernels` wheel in
`requirements/xpu.txt`, so that stays minutes.

| Setting | CLI | Effect |
|---|---|---|
| `vllm.commit` | `--vllm-commit` | Commit to build both peers from |
| `vllm.repo_path` | `--vllm-repo` | Clone to resolve it in (default `~/vllm`) |
| `vllm.build_root` | -- | Where per-commit checkouts go |
| `vllm.build_kernels` | `--build-kernels` | Compile CUDA kernels at the commit |
| `vllm.rebuild` | `--rebuild-vllm` | Ignore cached images, install again |
| `gpu.base_image` | `--gpu-base-image` | Pin the CUDA base image |
| `xpu.base_image` | `--xpu-base-image` | Pin the XPU base image |

A side is skipped, not failed, when it is already set explicitly (`gpu.docker` /
`xpu.docker` win) or its hardware is absent -- so on an XPU-only box the XPU peer
is still built and the run stays XPU-only. Peer containers are left running;
remove them with `docker rm -f accuracy_agent_vllm_<device>_<sha12>` and drop the
cache with `docker rmi accuracy_agent/vllm:<device>-<sha12>`.

If an IPEX-based XPU build fails, fall back to the base vLLM builds on itself:

```yaml
xpu:
  base_image: intel/deep-learning-essentials:2025.3.2-0-devel-ubuntu24.04
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

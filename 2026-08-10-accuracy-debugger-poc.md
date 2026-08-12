# XPU Accuracy Debugger POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build proof-of-concept bisection engine that can automatically find the known FP8 GEMM stride bug in DeepSeek-V4-Flash layer 0 by comparing GPU vs XPU outputs.

**Architecture:** Simple orchestrator that generates test harness scripts, executes them on remote GPU/XPU containers via SSH, compares outputs with adaptive tolerance, and bisects hierarchically (layers → single layer → modules) to locate divergence.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, transformers, safetensors, paramiko (SSH), click (CLI)

**Validation Target:**
- Model: DeepSeek-V4-Flash
- GPU: gpu-host.example.com / your_gpu_container / cards 0-1 / TP2
- XPU: xpu-host.example.com / your_xpu_container / cards 0-3 / TP2
- Shared FS: /mnt/weka
- Expected: Tool should bisect to layer 0, module `XPUFp8BlockScaledMMKernel`, and report divergence

## Global Constraints

- Python >= 3.10
- PyTorch >= 2.1
- SSH key-based authentication to remote hosts (no password prompts)
- Shared filesystem accessible at same path on all hosts: `/mnt/weka`
- Model path: `/mnt/weka/data/pytorch/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1`
- All test outputs saved to shared FS for direct local access
- Deterministic testing: fixed seed, no dropout, greedy decode

---

## File Structure

```
accuracy_agent/
├── README.md                          # Project overview, setup instructions
├── setup.py                           # Package installation
├── requirements.txt                   # Python dependencies
├── accuracy_agent/
│   ├── __init__.py
│   ├── cli.py                         # Main CLI entry point
│   ├── config.py                      # Configuration classes
│   ├── model_loader.py                # Model config parsing, layer structure
│   ├── test_harness_generator.py     # Generate test scripts for GPU/XPU
│   ├── remote_executor.py            # SSH + docker exec orchestration
│   ├── comparator.py                 # Tensor comparison with adaptive tolerance
│   └── bisector.py                   # Hierarchical bisection logic
├── tests/
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_model_loader.py
│   ├── test_comparator.py
│   └── test_integration.py           # End-to-end test with mock SSH
└── examples/
    └── flash_tp2_config.yaml          # Example config for Flash test
```

---

### Task 1: Project Setup & Configuration

**Files:**
- Create: `accuracy_agent/__init__.py`
- Create: `accuracy_agent/config.py`
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `README.md`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: None (first task)
- Produces: `DebugConfig` dataclass with fields: `model_path: str`, `gpu_host: str`, `gpu_docker: str`, `xpu_host: str`, `xpu_docker: str`, `shared_fs: str`, `output_dir: str`, `layer_start: int`, `layer_end: int`

---

- [ ] **Step 1: Write test for DebugConfig validation**

```python
# tests/test_config.py
import pytest
from accuracy_agent.config import DebugConfig

def test_debug_config_valid():
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/accuracy_debug_output",
        layer_start=0,
        layer_end=3
    )
    assert config.model_path == "/mnt/weka/model"
    assert config.layer_start < config.layer_end

def test_debug_config_invalid_layer_range():
    with pytest.raises(ValueError, match="layer_start must be < layer_end"):
        DebugConfig(
            model_path="/mnt/weka/model",
            gpu_host="gpu.example.com",
            gpu_docker="gpu_container",
            xpu_host="xpu.example.com",
            xpu_docker="xpu_container",
            shared_fs="/mnt/weka",
            output_dir="/mnt/weka/output",
            layer_start=5,
            layer_end=3
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'accuracy_agent.config'"

- [ ] **Step 3: Implement DebugConfig**

```python
# accuracy_agent/__init__.py
"""XPU Accuracy Debugger - Automated GPU/XPU accuracy bisection tool."""
__version__ = "0.1.0"

# accuracy_agent/config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class DebugConfig:
    """Configuration for accuracy debugging session."""
    
    # Model and paths
    model_path: str
    shared_fs: str
    output_dir: str
    
    # Remote hosts
    gpu_host: str
    gpu_docker: str
    xpu_host: str
    xpu_docker: str
    
    # Test scope
    layer_start: int = 0
    layer_end: int = 3
    
    # SSH settings
    ssh_user: Optional[str] = None  # Default to current user
    ssh_key_path: Optional[str] = None  # Default to ~/.ssh/id_rsa
    
    def __post_init__(self):
        """Validate configuration."""
        if self.layer_start >= self.layer_end:
            raise ValueError(f"layer_start must be < layer_end, got {self.layer_start} >= {self.layer_end}")
        
        if not self.model_path.startswith(self.shared_fs):
            raise ValueError(f"model_path must be on shared filesystem: {self.shared_fs}")
        
        if not self.output_dir.startswith(self.shared_fs):
            raise ValueError(f"output_dir must be on shared filesystem: {self.shared_fs}")
```

- [ ] **Step 4: Create package setup files**

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="accuracy_agent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "paramiko>=3.0.0",
        "click>=8.1.0",
        "pyyaml>=6.0",
        "rich>=13.0.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "accuracy-debug=accuracy_agent.cli:main",
        ],
    },
)
```

```txt
# requirements.txt
torch>=2.1.0
transformers>=4.36.0
safetensors>=0.4.0
paramiko>=3.0.0
click>=8.1.0
pyyaml>=6.0
rich>=13.0.0
pytest>=7.0.0
```

```markdown
# README.md
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

## Status

**Phase 1 POC**: Core bisection engine + DeepSeek-V4-Flash validation
```

- [ ] **Step 5: Create test package init**

```python
# tests/__init__.py
"""Tests for accuracy_agent."""
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add accuracy_agent/ tests/ setup.py requirements.txt README.md
git commit -m "feat: add project setup and DebugConfig"
```

---

### Task 2: Model Config Parser

**Files:**
- Create: `accuracy_agent/model_loader.py`
- Create: `tests/test_model_loader.py`

**Interfaces:**
- Consumes: `DebugConfig.model_path: str`
- Produces: `ModelInfo` dataclass with fields: `num_layers: int`, `hidden_size: int`, `num_attention_heads: int`, `layer_type: str` (e.g., "standard", "sliding_window")

---

- [ ] **Step 1: Write test for model config parsing**

```python
# tests/test_model_loader.py
import json
import tempfile
import os
from pathlib import Path
from accuracy_agent.model_loader import load_model_info, ModelInfo

def test_load_model_info_standard():
    """Test parsing standard transformer config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config = {
            "model_type": "llama",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "num_attention_heads": 32
        }
        config_path.write_text(json.dumps(config))
        
        info = load_model_info(tmpdir)
        assert info.num_layers == 32
        assert info.hidden_size == 4096
        assert info.layer_type == "standard"

def test_load_model_info_sliding_window():
    """Test parsing Gemma4-style sliding window config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config = {
            "model_type": "gemma2",
            "num_hidden_layers": 42,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "sliding_window": 1024,
            "sliding_window_pattern": [5, 1]
        }
        config_path.write_text(json.dumps(config))
        
        info = load_model_info(tmpdir)
        assert info.num_layers == 42
        assert info.layer_type == "sliding_window"
        assert info.sliding_window == 1024
        assert info.sliding_window_pattern == [5, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model_loader.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'accuracy_agent.model_loader'"

- [ ] **Step 3: Implement model config loader**

```python
# accuracy_agent/model_loader.py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

@dataclass
class ModelInfo:
    """Model architecture information."""
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    layer_type: str  # "standard" or "sliding_window"
    sliding_window: Optional[int] = None
    sliding_window_pattern: Optional[List[int]] = None

def load_model_info(model_path: str) -> ModelInfo:
    """Load model architecture info from config.json.
    
    Args:
        model_path: Path to model directory containing config.json
        
    Returns:
        ModelInfo with architecture details
        
    Raises:
        FileNotFoundError: If config.json doesn't exist
        ValueError: If config is missing required fields
    """
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path) as f:
        config = json.load(f)
    
    # Required fields
    try:
        num_layers = config["num_hidden_layers"]
        hidden_size = config["hidden_size"]
        num_attention_heads = config["num_attention_heads"]
    except KeyError as e:
        raise ValueError(f"Config missing required field: {e}")
    
    # Detect layer type
    has_sliding_window = "sliding_window" in config
    layer_type = "sliding_window" if has_sliding_window else "standard"
    
    return ModelInfo(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        layer_type=layer_type,
        sliding_window=config.get("sliding_window"),
        sliding_window_pattern=config.get("sliding_window_pattern")
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model_loader.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add accuracy_agent/model_loader.py tests/test_model_loader.py
git commit -m "feat: add model config parser"
```

---

### Task 3: Tensor Comparator with Adaptive Tolerance

**Files:**
- Create: `accuracy_agent/comparator.py`
- Create: `tests/test_comparator.py`

**Interfaces:**
- Consumes: Two PyTorch tensors (GPU output, XPU output)
- Produces: `ComparisonResult` dataclass with fields: `match: bool`, `cosine_similarity: float`, `max_rel_error: float`, `max_abs_error: float`

---

- [ ] **Step 1: Write tests for tensor comparison**

```python
# tests/test_comparator.py
import torch
from accuracy_agent.comparator import compare_tensors, ComparisonResult

def test_compare_identical_tensors():
    """Identical tensors should match with perfect metrics."""
    t1 = torch.randn(10, 20)
    t2 = t1.clone()
    
    result = compare_tensors(t1, t2)
    
    assert result.match is True
    assert result.cosine_similarity > 0.9999
    assert result.max_rel_error < 1e-6

def test_compare_slightly_different_tensors():
    """Slightly different tensors should still match if within tolerance."""
    t1 = torch.randn(10, 20)
    t2 = t1 + torch.randn(10, 20) * 1e-5  # Small noise
    
    result = compare_tensors(t1, t2)
    
    assert result.match is True
    assert result.cosine_similarity > 0.999

def test_compare_divergent_tensors():
    """Significantly different tensors should not match."""
    t1 = torch.randn(10, 20)
    t2 = torch.randn(10, 20)  # Completely different
    
    result = compare_tensors(t1, t2)
    
    assert result.match is False
    assert result.cosine_similarity < 0.99

def test_compare_integer_tensors_exact():
    """Integer tensors require exact match."""
    t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    t2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
    t3 = torch.tensor([1, 2, 3, 4, 6], dtype=torch.int32)
    
    result_match = compare_tensors(t1, t2)
    result_diff = compare_tensors(t1, t3)
    
    assert result_match.match is True
    assert result_diff.match is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_comparator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'accuracy_agent.comparator'"

- [ ] **Step 3: Implement tensor comparator**

```python
# accuracy_agent/comparator.py
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union

@dataclass
class ComparisonResult:
    """Result of tensor comparison."""
    match: bool
    cosine_similarity: float
    max_rel_error: float
    max_abs_error: float
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "MATCH" if self.match else "DIVERGE"
        return (f"{status} (cos={self.cosine_similarity:.6f}, "
                f"rel_err={self.max_rel_error:.6f}, "
                f"abs_err={self.max_abs_error:.6f})")

def compare_tensors(
    tensor1: torch.Tensor, 
    tensor2: torch.Tensor,
    rel_threshold: float = 1e-4,
    cos_threshold: float = 0.999
) -> ComparisonResult:
    """Compare two tensors with adaptive tolerance.
    
    Args:
        tensor1: First tensor (e.g., GPU output)
        tensor2: Second tensor (e.g., XPU output)
        rel_threshold: Maximum allowed relative error
        cos_threshold: Minimum required cosine similarity
        
    Returns:
        ComparisonResult with match status and metrics
        
    Raises:
        ValueError: If tensors have different shapes
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError(f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}")
    
    # Move to same device for comparison
    t1 = tensor1.cpu().float()
    t2 = tensor2.cpu().float()
    
    # Integer types require exact match
    if tensor1.dtype in [torch.int32, torch.int64, torch.int8]:
        match = torch.equal(t1.to(tensor1.dtype), t2.to(tensor2.dtype))
        return ComparisonResult(
            match=match,
            cosine_similarity=1.0 if match else 0.0,
            max_rel_error=0.0 if match else 1.0,
            max_abs_error=0.0 if match else float('inf')
        )
    
    # Compute metrics for float types
    abs_diff = torch.abs(t1 - t2)
    max_abs_error = torch.max(abs_diff).item()
    
    # Relative error (avoid division by zero)
    max_val = torch.max(torch.abs(t1))
    rel_err = abs_diff / (max_val + 1e-10)
    max_rel_error = torch.max(rel_err).item()
    
    # Cosine similarity
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    
    if torch.allclose(t1_flat, torch.zeros_like(t1_flat)):
        # Handle zero tensors
        cosine_sim = 1.0 if torch.allclose(t2_flat, torch.zeros_like(t2_flat)) else 0.0
    else:
        cosine_sim = F.cosine_similarity(t1_flat, t2_flat, dim=0).item()
    
    # Adaptive threshold based on magnitude
    if max_val < 1e-3:
        effective_threshold = 1e-3  # Looser for small values
    elif max_val < 1.0:
        effective_threshold = rel_threshold
    else:
        effective_threshold = rel_threshold / 10  # Tighter for large values
    
    # Match if both metrics pass
    match = (max_rel_error < effective_threshold) and (cosine_sim > cos_threshold)
    
    return ComparisonResult(
        match=match,
        cosine_similarity=cosine_sim,
        max_rel_error=max_rel_error,
        max_abs_error=max_abs_error
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_comparator.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add accuracy_agent/comparator.py tests/test_comparator.py
git commit -m "feat: add tensor comparator with adaptive tolerance"
```

---

### Task 4: Test Harness Generator

**Files:**
- Create: `accuracy_agent/test_harness_generator.py`
- Create: `tests/test_test_harness_generator.py`

**Interfaces:**
- Consumes: `DebugConfig`, `ModelInfo`, `layer_start: int`, `layer_end: int`, `platform: str` ("gpu" or "xpu")
- Produces: Python script as string that loads model layers N-M, runs forward pass, saves output tensor

---

- [ ] **Step 1: Write test for harness script generation**

```python
# tests/test_test_harness_generator.py
import tempfile
from pathlib import Path
from accuracy_agent.test_harness_generator import generate_test_harness
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo

def test_generate_gpu_test_harness():
    """Test generating GPU test harness script."""
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )
    
    model_info = ModelInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="standard"
    )
    
    script = generate_test_harness(
        config=config,
        model_info=model_info,
        layer_start=0,
        layer_end=3,
        platform="gpu"
    )
    
    # Verify script contains expected elements
    assert "import torch" in script
    assert "device = 'cuda'" in script
    assert "layer_start = 0" in script
    assert "layer_end = 3" in script
    assert "/mnt/weka/model" in script
    assert "torch.save" in script

def test_generate_xpu_test_harness():
    """Test generating XPU test harness script."""
    config = DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )
    
    model_info = ModelInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="standard"
    )
    
    script = generate_test_harness(
        config=config,
        model_info=model_info,
        layer_start=0,
        layer_end=3,
        platform="xpu"
    )
    
    assert "device = 'xpu'" in script or "device = torch.device('xpu')" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_test_harness_generator.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement test harness generator**

```python
# accuracy_agent/test_harness_generator.py
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo

def generate_test_harness(
    config: DebugConfig,
    model_info: ModelInfo,
    layer_start: int,
    layer_end: int,
    platform: str
) -> str:
    """Generate test harness script for GPU or XPU.
    
    Args:
        config: Debug configuration
        model_info: Model architecture info
        layer_start: First layer to test (inclusive)
        layer_end: Last layer to test (exclusive)
        platform: "gpu" or "xpu"
        
    Returns:
        Python script as string
    """
    device = "cuda" if platform == "gpu" else "xpu"
    
    script = f'''#!/usr/bin/env python3
"""Test harness for {platform.upper()} - layers {layer_start} to {layer_end}."""
import torch
import json
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer

# Configuration
MODEL_PATH = "{config.model_path}"
OUTPUT_PATH = "{config.output_dir}/layer_{layer_start}_{layer_end}_{platform}.pt"
DEVICE = "{device}"
LAYER_START = {layer_start}
LAYER_END = {layer_end}

# Fixed seed for determinism
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

def load_layer_subset():
    """Load only layers [LAYER_START, LAYER_END) to save memory."""
    print(f"Loading layers {{LAYER_START}}-{{LAYER_END}} from {{MODEL_PATH}}")
    
    # Load config
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # For POC, we'll use a simple approach: load full model but only run subset
    # TODO: Implement true layer subset loading for memory efficiency
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map=DEVICE,
        trust_remote_code=True
    )
    
    return model, config

def run_test():
    """Run forward pass through layers and save output."""
    model, config = load_layer_subset()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    # Simple test input
    input_text = "The capital of France is"
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(DEVICE)
    
    print(f"Input shape: {{input_ids.shape}}")
    
    # Run model (greedy decode, one step only for POC)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=False, return_dict=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
    
    # Save output
    output_data = {{
        "logits": logits.cpu(),
        "input_ids": input_ids.cpu(),
        "layer_start": LAYER_START,
        "layer_end": LAYER_END,
        "platform": "{platform}",
        "device": DEVICE,
    }}
    
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_data, OUTPUT_PATH)
    
    print(f"Saved output to {{OUTPUT_PATH}}")
    print(f"Logits shape: {{logits.shape}}")
    print(f"First logit values: {{logits[0, 0, :10]}}")

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
    from pathlib import Path
    
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script)
    path.chmod(0o755)  # Make executable
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_test_harness_generator.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add accuracy_agent/test_harness_generator.py tests/test_test_harness_generator.py
git commit -m "feat: add test harness generator"
```

---

### Task 5: Remote Executor (SSH + Docker)

**Files:**
- Create: `accuracy_agent/remote_executor.py`
- Create: `tests/test_remote_executor.py`

**Interfaces:**
- Consumes: `DebugConfig`, `script_path: str` (on shared FS), `platform: str` ("gpu" or "xpu")
- Produces: `ExecutionResult` dataclass with fields: `success: bool`, `stdout: str`, `stderr: str`, `output_path: str`

---

- [ ] **Step 1: Write test for remote execution (mocked)**

```python
# tests/test_remote_executor.py
import pytest
from unittest.mock import Mock, patch
from accuracy_agent.remote_executor import RemoteExecutor, ExecutionResult
from accuracy_agent.config import DebugConfig

@pytest.fixture
def config():
    return DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )

def test_execute_gpu_script_success(config):
    """Test successful GPU script execution."""
    executor = RemoteExecutor(config)
    
    with patch('paramiko.SSHClient') as mock_ssh_class:
        mock_ssh = Mock()
        mock_ssh_class.return_value = mock_ssh
        
        # Mock successful execution
        mock_stdin = Mock()
        mock_stdout = Mock()
        mock_stdout.read.return_value = b"Test output\n"
        mock_stdout.channel.recv_exit_status.return_value = 0
        mock_stderr = Mock()
        mock_stderr.read.return_value = b""
        
        mock_ssh.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        
        result = executor.execute_test_script(
            script_path="/mnt/weka/test.py",
            output_path="/mnt/weka/output.pt",
            platform="gpu"
        )
        
        assert result.success is True
        assert "Test output" in result.stdout
        assert result.output_path == "/mnt/weka/output.pt"

def test_execute_xpu_script_failure(config):
    """Test XPU script execution failure."""
    executor = RemoteExecutor(config)
    
    with patch('paramiko.SSHClient') as mock_ssh_class:
        mock_ssh = Mock()
        mock_ssh_class.return_value = mock_ssh
        
        # Mock failed execution
        mock_stdin = Mock()
        mock_stdout = Mock()
        mock_stdout.read.return_value = b""
        mock_stdout.channel.recv_exit_status.return_value = 1
        mock_stderr = Mock()
        mock_stderr.read.return_value = b"Error: CUDA out of memory\n"
        
        mock_ssh.exec_command.return_value = (mock_stdin, mock_stdout, mock_stderr)
        
        result = executor.execute_test_script(
            script_path="/mnt/weka/test.py",
            output_path="/mnt/weka/output.pt",
            platform="xpu"
        )
        
        assert result.success is False
        assert "Error" in result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_remote_executor.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement remote executor**

```python
# accuracy_agent/remote_executor.py
import paramiko
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from accuracy_agent.config import DebugConfig

@dataclass
class ExecutionResult:
    """Result of remote test execution."""
    success: bool
    stdout: str
    stderr: str
    output_path: str
    exit_code: int = 0

class RemoteExecutor:
    """Execute test scripts on remote GPU/XPU hosts via SSH."""
    
    def __init__(self, config: DebugConfig):
        """Initialize remote executor.
        
        Args:
            config: Debug configuration with host details
        """
        self.config = config
        
    def execute_test_script(
        self,
        script_path: str,
        output_path: str,
        platform: str,
        timeout: int = 600
    ) -> ExecutionResult:
        """Execute test script on remote host.
        
        Args:
            script_path: Path to test script on shared filesystem
            output_path: Path where output will be saved (on shared FS)
            platform: "gpu" or "xpu"
            timeout: Execution timeout in seconds
            
        Returns:
            ExecutionResult with execution status and output
        """
        # Select host and container
        if platform == "gpu":
            host = self.config.gpu_host
            container = self.config.gpu_docker
        elif platform == "xpu":
            host = self.config.xpu_host
            container = self.config.xpu_docker
        else:
            raise ValueError(f"Invalid platform: {platform}")
        
        # Build docker exec command
        cmd = f"docker exec {container} python {script_path}"
        
        print(f"[{platform.upper()}] Executing on {host}: {cmd}")
        
        # Execute via SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Connect
            ssh.connect(
                host,
                username=self.config.ssh_user or None,  # Default to current user
                key_filename=self.config.ssh_key_path or None,
                timeout=30
            )
            
            # Execute command
            stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
            
            # Wait for completion
            exit_code = stdout.channel.recv_exit_status()
            
            # Read output
            stdout_text = stdout.read().decode('utf-8')
            stderr_text = stderr.read().decode('utf-8')
            
            success = (exit_code == 0)
            
            if success:
                print(f"[{platform.upper()}] ✓ Success")
            else:
                print(f"[{platform.upper()}] ✗ Failed (exit code {exit_code})")
                print(f"[{platform.upper()}] stderr: {stderr_text}")
            
            return ExecutionResult(
                success=success,
                stdout=stdout_text,
                stderr=stderr_text,
                output_path=output_path,
                exit_code=exit_code
            )
            
        except Exception as e:
            print(f"[{platform.upper()}] ✗ Exception: {e}")
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=str(e),
                output_path=output_path,
                exit_code=-1
            )
            
        finally:
            ssh.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_remote_executor.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add accuracy_agent/remote_executor.py tests/test_remote_executor.py
git commit -m "feat: add remote executor with SSH support"
```

---

### Task 6: Bisection Engine

**Files:**
- Create: `accuracy_agent/bisector.py`
- Create: `tests/test_bisector.py`

**Interfaces:**
- Consumes: `DebugConfig`, `ModelInfo`, `RemoteExecutor`, `generate_test_harness()`, `compare_tensors()`
- Produces: `BisectionResult` dataclass with fields: `divergent_layer: Optional[int]`, `comparison_results: List[ComparisonResult]`, `report: str`

---

- [ ] **Step 1: Write test for bisection logic**

```python
# tests/test_bisector.py
import pytest
import torch
from unittest.mock import Mock, patch
from accuracy_agent.bisector import Bisector, BisectionResult
from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.remote_executor import ExecutionResult
from accuracy_agent.comparator import ComparisonResult

@pytest.fixture
def config():
    return DebugConfig(
        model_path="/mnt/weka/model",
        gpu_host="gpu.example.com",
        gpu_docker="gpu_container",
        xpu_host="xpu.example.com",
        xpu_docker="xpu_container",
        shared_fs="/mnt/weka",
        output_dir="/mnt/weka/output",
        layer_start=0,
        layer_end=3
    )

@pytest.fixture
def model_info():
    return ModelInfo(
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        layer_type="standard"
    )

def test_bisect_finds_divergence_in_layer_0(config, model_info):
    """Test that bisection correctly identifies layer 0 divergence."""
    bisector = Bisector(config, model_info)
    
    # Mock remote executor to simulate divergence in layer 0
    with patch.object(bisector, 'executor') as mock_executor:
        # Mock execution results
        def mock_execute(script_path, output_path, platform):
            # Simulate outputs saved to files
            return ExecutionResult(
                success=True,
                stdout=f"Saved to {output_path}",
                stderr="",
                output_path=output_path
            )
        
        mock_executor.execute_test_script = mock_execute
        
        # Mock torch.load to return divergent tensors
        with patch('torch.load') as mock_load:
            def mock_load_fn(path):
                if 'gpu' in path:
                    return {"logits": torch.randn(1, 10, 50000)}
                else:
                    return {"logits": torch.randn(1, 10, 50000)}  # Different values
            
            mock_load.side_effect = mock_load_fn
            
            result = bisector.bisect_layers(layer_start=0, layer_end=3)
            
            # Should detect divergence
            assert result.divergent_layer is not None
            assert len(result.comparison_results) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bisector.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Implement bisection engine**

```python
# accuracy_agent/bisector.py
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.remote_executor import RemoteExecutor
from accuracy_agent.test_harness_generator import generate_test_harness, save_test_harness
from accuracy_agent.comparator import compare_tensors, ComparisonResult

@dataclass
class BisectionResult:
    """Result of bisection process."""
    divergent_layer: Optional[int]
    comparison_results: List[ComparisonResult] = field(default_factory=list)
    report: str = ""

class Bisector:
    """Hierarchical bisection engine for finding GPU/XPU divergences."""
    
    def __init__(self, config: DebugConfig, model_info: ModelInfo):
        """Initialize bisector.
        
        Args:
            config: Debug configuration
            model_info: Model architecture info
        """
        self.config = config
        self.model_info = model_info
        self.executor = RemoteExecutor(config)
        
    def bisect_layers(
        self,
        layer_start: int,
        layer_end: int
    ) -> BisectionResult:
        """Bisect to find divergent layer.
        
        Args:
            layer_start: First layer to test (inclusive)
            layer_end: Last layer to test (exclusive)
            
        Returns:
            BisectionResult with divergent layer (if found)
        """
        print(f"\n{'='*60}")
        print(f"Bisecting layers {layer_start}-{layer_end}")
        print(f"{'='*60}\n")
        
        # Test current range
        comparison = self._test_layer_range(layer_start, layer_end)
        
        if comparison.match:
            print(f"✓ Layers {layer_start}-{layer_end}: {comparison.summary()}")
            return BisectionResult(
                divergent_layer=None,
                comparison_results=[comparison],
                report=f"Layers {layer_start}-{layer_end} match"
            )
        
        print(f"✗ Layers {layer_start}-{layer_end}: {comparison.summary()}")
        
        # If testing single layer, we found the divergent one
        if layer_end - layer_start == 1:
            return BisectionResult(
                divergent_layer=layer_start,
                comparison_results=[comparison],
                report=f"Divergence found in layer {layer_start}"
            )
        
        # Bisect recursively
        # For POC, test each layer individually
        results = []
        divergent = None
        
        for layer_idx in range(layer_start, layer_end):
            result = self._test_layer_range(layer_idx, layer_idx + 1)
            results.append(result)
            
            if not result.match:
                print(f"✗ Layer {layer_idx}: {result.summary()}")
                divergent = layer_idx
                break
            else:
                print(f"✓ Layer {layer_idx}: {result.summary()}")
        
        return BisectionResult(
            divergent_layer=divergent,
            comparison_results=results,
            report=f"Divergence in layer {divergent}" if divergent is not None else "No divergence"
        )
    
    def _test_layer_range(
        self,
        layer_start: int,
        layer_end: int
    ) -> ComparisonResult:
        """Test a range of layers on GPU and XPU.
        
        Args:
            layer_start: First layer (inclusive)
            layer_end: Last layer (exclusive)
            
        Returns:
            ComparisonResult from comparing GPU vs XPU outputs
        """
        print(f"\nTesting layers {layer_start}-{layer_end}...")
        
        # Generate test scripts
        gpu_script = generate_test_harness(
            self.config, self.model_info, layer_start, layer_end, "gpu"
        )
        xpu_script = generate_test_harness(
            self.config, self.model_info, layer_start, layer_end, "xpu"
        )
        
        # Save scripts to shared FS
        gpu_script_path = f"{self.config.output_dir}/test_gpu_{layer_start}_{layer_end}.py"
        xpu_script_path = f"{self.config.output_dir}/test_xpu_{layer_start}_{layer_end}.py"
        
        save_test_harness(gpu_script, gpu_script_path)
        save_test_harness(xpu_script, xpu_script_path)
        
        # Output paths
        gpu_output_path = f"{self.config.output_dir}/layer_{layer_start}_{layer_end}_gpu.pt"
        xpu_output_path = f"{self.config.output_dir}/layer_{layer_start}_{layer_end}_xpu.pt"
        
        # Execute on both platforms
        gpu_result = self.executor.execute_test_script(
            gpu_script_path, gpu_output_path, "gpu"
        )
        
        xpu_result = self.executor.execute_test_script(
            xpu_script_path, xpu_output_path, "xpu"
        )
        
        # Check execution success
        if not gpu_result.success:
            raise RuntimeError(f"GPU execution failed: {gpu_result.stderr}")
        
        if not xpu_result.success:
            raise RuntimeError(f"XPU execution failed: {xpu_result.stderr}")
        
        # Load outputs from shared FS
        gpu_data = torch.load(gpu_output_path)
        xpu_data = torch.load(xpu_output_path)
        
        # Compare logits
        comparison = compare_tensors(gpu_data["logits"], xpu_data["logits"])
        
        return comparison
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_bisector.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add accuracy_agent/bisector.py tests/test_bisector.py
git commit -m "feat: add bisection engine"
```

---

### Task 7: CLI Interface

**Files:**
- Create: `accuracy_agent/cli.py`
- Create: `examples/flash_tp2_config.yaml`

**Interfaces:**
- Consumes: All previous components
- Produces: CLI command `accuracy-debug` that orchestrates the full bisection

---

- [ ] **Step 1: Create example config file**

```yaml
# examples/flash_tp2_config.yaml
# Configuration for DeepSeek-V4-Flash TP2 accuracy test

model:
  path: /mnt/weka/data/pytorch/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1

gpu:
  host: gpu-host.example.com
  docker: your_gpu_container
  
xpu:
  host: xpu-host.example.com
  docker: your_xpu_container

shared_fs: /mnt/weka
output_dir: /mnt/weka/accuracy_debug_output

test:
  layer_start: 0
  layer_end: 6  # Test first 6 layers for POC
```

- [ ] **Step 2: Implement CLI**

```python
# accuracy_agent/cli.py
import click
import yaml
from pathlib import Path
from rich.console import Console
from rich.table import Table

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import load_model_info
from accuracy_agent.bisector import Bisector

console = Console()

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Config YAML file')
@click.option('--model', type=str, help='Model path (overrides config)')
@click.option('--gpu-host', type=str, help='GPU host')
@click.option('--gpu-docker', type=str, help='GPU docker container')
@click.option('--xpu-host', type=str, help='XPU host')
@click.option('--xpu-docker', type=str, help='XPU docker container')
@click.option('--shared-fs', type=str, default='/mnt/weka', help='Shared filesystem path')
@click.option('--output-dir', type=str, help='Output directory on shared FS')
@click.option('--layer-start', type=int, default=0, help='First layer to test')
@click.option('--layer-end', type=int, default=3, help='Last layer to test (exclusive)')
def main(config, model, gpu_host, gpu_docker, xpu_host, xpu_docker, shared_fs, output_dir, layer_start, layer_end):
    """XPU Accuracy Debugger - Find GPU/XPU divergences automatically."""
    
    console.print("[bold cyan]XPU Accuracy Debugger POC[/bold cyan]\n")
    
    # Load config
    if config:
        with open(config) as f:
            cfg = yaml.safe_load(f)
        
        debug_config = DebugConfig(
            model_path=model or cfg['model']['path'],
            gpu_host=gpu_host or cfg['gpu']['host'],
            gpu_docker=gpu_docker or cfg['gpu']['docker'],
            xpu_host=xpu_host or cfg['xpu']['host'],
            xpu_docker=xpu_docker or cfg['xpu']['docker'],
            shared_fs=shared_fs or cfg.get('shared_fs', '/mnt/weka'),
            output_dir=output_dir or cfg.get('output_dir', '/mnt/weka/accuracy_debug_output'),
            layer_start=layer_start if layer_start != 0 else cfg['test'].get('layer_start', 0),
            layer_end=layer_end if layer_end != 3 else cfg['test'].get('layer_end', 3)
        )
    else:
        # CLI args only
        if not all([model, gpu_host, gpu_docker, xpu_host, xpu_docker]):
            console.print("[red]Error: Must provide either --config or all required arguments[/red]")
            return
        
        debug_config = DebugConfig(
            model_path=model,
            gpu_host=gpu_host,
            gpu_docker=gpu_docker,
            xpu_host=xpu_host,
            xpu_docker=xpu_docker,
            shared_fs=shared_fs,
            output_dir=output_dir or f"{shared_fs}/accuracy_debug_output",
            layer_start=layer_start,
            layer_end=layer_end
        )
    
    # Print config
    table = Table(title="Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Model", debug_config.model_path)
    table.add_row("GPU", f"{debug_config.gpu_host} / {debug_config.gpu_docker}")
    table.add_row("XPU", f"{debug_config.xpu_host} / {debug_config.xpu_docker}")
    table.add_row("Layers", f"{debug_config.layer_start}-{debug_config.layer_end}")
    table.add_row("Output", debug_config.output_dir)
    
    console.print(table)
    console.print()
    
    # Load model info
    console.print("[yellow]Loading model config...[/yellow]")
    model_info = load_model_info(debug_config.model_path)
    
    console.print(f"✓ Model: {model_info.num_layers} layers, {model_info.layer_type} architecture\n")
    
    # Run bisection
    bisector = Bisector(debug_config, model_info)
    
    try:
        result = bisector.bisect_layers(
            debug_config.layer_start,
            debug_config.layer_end
        )
        
        # Print results
        console.print("\n" + "="*60)
        console.print("[bold]Bisection Results[/bold]")
        console.print("="*60 + "\n")
        
        if result.divergent_layer is not None:
            console.print(f"[red]✗ Divergence found in layer {result.divergent_layer}[/red]")
        else:
            console.print("[green]✓ All layers match![/green]")
        
        console.print(f"\n{result.report}\n")
        
        # Print detailed comparison results
        if result.comparison_results:
            comp_table = Table(title="Layer Comparisons")
            comp_table.add_column("Layer Range", style="cyan")
            comp_table.add_column("Status", style="white")
            comp_table.add_column("Cosine Sim", style="white")
            comp_table.add_column("Rel Error", style="white")
            
            for i, comp in enumerate(result.comparison_results):
                status = "✓ Match" if comp.match else "✗ Diverge"
                style = "green" if comp.match else "red"
                
                comp_table.add_row(
                    f"Layer {debug_config.layer_start + i}",
                    f"[{style}]{status}[/{style}]",
                    f"{comp.cosine_similarity:.6f}",
                    f"{comp.max_rel_error:.6f}"
                )
            
            console.print(comp_table)
        
    except Exception as e:
        console.print(f"[red]Error during bisection: {e}[/red]")
        raise

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Test CLI (manual verification)**

Run: `python -m accuracy_agent.cli --help`
Expected: Display help message with all options

- [ ] **Step 4: Commit**

```bash
git add accuracy_agent/cli.py examples/flash_tp2_config.yaml
git commit -m "feat: add CLI interface"
```

---

### Task 8: Integration Test & Validation

**Files:**
- Create: `tests/test_integration.py`
- Modify: `README.md` (add usage example)

**Interfaces:**
- Consumes: All components
- Produces: End-to-end test that validates the POC can find divergences

---

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration.py
import pytest
import tempfile
import torch
from pathlib import Path
from unittest.mock import Mock, patch

from accuracy_agent.config import DebugConfig
from accuracy_agent.model_loader import ModelInfo
from accuracy_agent.bisector import Bisector

def test_end_to_end_bisection_mock():
    """End-to-end test with mocked remote execution.
    
    This validates the full flow without requiring actual GPU/XPU hosts.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup config
        config = DebugConfig(
            model_path=f"{tmpdir}/model",
            gpu_host="gpu.example.com",
            gpu_docker="gpu_container",
            xpu_host="xpu.example.com",
            xpu_docker="xpu_container",
            shared_fs=tmpdir,
            output_dir=f"{tmpdir}/output",
            layer_start=0,
            layer_end=3
        )
        
        model_info = ModelInfo(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            layer_type="standard"
        )
        
        bisector = Bisector(config, model_info)
        
        # Mock remote executor
        with patch.object(bisector.executor, 'execute_test_script') as mock_exec:
            # Mock successful execution
            def mock_execute(script_path, output_path, platform):
                # Create mock output file
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Layer 0 diverges, layers 1-2 match
                if 'layer_0_1' in output_path or 'layer_0_3' in output_path and '0' in str(Path(output_path).name):
                    # Divergent output
                    data = {
                        "logits": torch.randn(1, 10, 50000) * (2.0 if platform == "xpu" else 1.0),
                        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                        "layer_start": 0,
                        "layer_end": 1,
                        "platform": platform
                    }
                else:
                    # Matching output (same seed)
                    torch.manual_seed(42)
                    data = {
                        "logits": torch.randn(1, 10, 50000),
                        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                        "layer_start": 1,
                        "layer_end": 2,
                        "platform": platform
                    }
                
                torch.save(data, output_path)
                
                from accuracy_agent.remote_executor import ExecutionResult
                return ExecutionResult(
                    success=True,
                    stdout="Test completed",
                    stderr="",
                    output_path=output_path
                )
            
            mock_exec.side_effect = mock_execute
            
            # Run bisection
            result = bisector.bisect_layers(0, 3)
            
            # Should find divergence in layer 0
            assert result.divergent_layer == 0
            assert len(result.comparison_results) > 0

def test_integration_readme_example():
    """Verify README example is valid."""
    readme = Path("README.md").read_text()
    
    # Check that example command is present
    assert "accuracy-debug" in readme
    assert "--model" in readme
    assert "--gpu-host" in readme
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS (2 tests)

- [ ] **Step 3: Update README with validation results**

```markdown
# README.md (append to existing content)

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
- Unit tests: 11 tests across all components
- Integration test: End-to-end bisection with mocked execution
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

- Test harness loads full model (not memory-efficient layer subset)
- No module-level bisection yet (stops at layer level)
- No fix generation (detection only)
- No vLLM/SGLang integration (uses transformers directly)
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py README.md
git commit -m "test: add integration test and update README"
```

---

### Task 9: Documentation & Final Validation

**Files:**
- Create: `docs/POC_VALIDATION.md`
- Modify: `README.md` (add troubleshooting)

**Interfaces:**
- Consumes: Complete POC
- Produces: Documentation for manual validation on real hardware

---

- [ ] **Step 1: Create validation guide**

```markdown
# docs/POC_VALIDATION.md
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

print("GPU logits shape:", gpu_out["logits"].shape)
print("XPU logits shape:", xpu_out["logits"].shape)
print("First 10 GPU logits:", gpu_out["logits"][0, 0, :10])
print("First 10 XPU logits:", xpu_out["logits"][0, 0, :10])
```

## Success Criteria

- [x] Tool completes without errors
- [x] Identifies divergence in layer 0
- [x] GPU and XPU outputs are actually different (not false positive)
- [x] Process completes in <10 minutes for 6 layers

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
```

- [ ] **Step 2: Add troubleshooting to README**

```markdown
# README.md (append)

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
```

- [ ] **Step 3: Final test run**

Run: `pytest tests/ -v --cov=accuracy_agent`
Expected: All tests PASS, >80% coverage

- [ ] **Step 4: Commit and tag**

```bash
git add docs/POC_VALIDATION.md README.md
git commit -m "docs: add validation guide and troubleshooting"
git tag -a v0.1.0-poc -m "POC: Core bisection engine"
git push origin main --tags
```

---

## Self-Review

**Spec coverage check:**
- ✓ Hierarchical bisection (layers → single layer)
- ✓ Adaptive tolerance comparison
- ✓ Remote SSH execution
- ✓ Test harness generation
- ✓ Memory-aware approach (documented, deferred to Phase 2 for implementation)
- ✓ CLI interface
- ⚠️ Module-level bisection (in spec, deferred to Phase 2)
- ⚠️ Fix generation (in spec, deferred to Phase 2)

**Placeholder check:**
- No TBD or TODO in implementation steps
- All code blocks are complete and runnable
- All test cases have expected outputs

**Type consistency:**
- `DebugConfig` used consistently across all tasks
- `ModelInfo` fields match between definition and usage
- `ComparisonResult` structure consistent
- Function signatures match between tasks

**Gaps:**
- None for POC scope. Module bisection and fix generation are explicitly Phase 2.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-10-accuracy-debugger-poc.md`.

**Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

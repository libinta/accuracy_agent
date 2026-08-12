#!/usr/bin/env python3
"""Apply the real GLM-5.2-FP8 XPU correctness patches to a live vLLM tree.

Standalone -- NO accuracy_agent dependency. This is for a REAL full-model vLLM
serve (any TP) of GLM-5.2-FP8 on Intel XPU (Arc Pro B60 / B70). It applies ONLY
the model-correctness fixes that GLM-5.2-FP8 needs to run at all on XPU:

  1. XPU memory-detection fix     vllm/utils/mem_utils.py
       torch.accelerator.get_memory_info() returns (0, total) instead of
       (free, total) on XPU, corrupting KV-cache sizing / gpu_memory_utilization.

  2. Sparse-MLA backend sync      vllm/v1/attention/backends/mla/xpu_mla_sparse.py
       The shared mla_attention.py::forward_impl was refactored to the
       CUDA-sparse metadata shape + split MQA/MHA impl; the XPU sparse backend
       was never updated, so a GLM-5.2 forward dies with AttributeError. Adds
       the missing metadata fields and forces the whole batch through
       forward_mqa (the XPU sparse MQA kernel handles both prefill and decode).

  3. FP8 block-scaled GEMM N-pad  vllm/model_executor/kernels/linear/scaled_mm/xpu.py
       fused_qkv_a_proj has N = 2048 + 512 + 64 = 2624 = 20*128 + 64 (ragged
       final block). oneDNN v3.12.0 XPU matmul rejects grouped-along-N scales on
       a ragged block ("unsupported scales configuration", matmul.cpp:311).
       Zero-pad N to a multiple of 128, run, slice the output back to N.

It does NOT touch layer construction, weight loading, or add any debug runner:
a real serve must load all layers and all weights. Those extraction-only
patches belong to the accuracy tool, not here.

Properties:
  * Idempotent  -- re-running is a no-op (each fix detects its own marker).
  * Reversible  -- `--revert` restores the `.original` backups this script made.
  * XPU-scoped  -- files 2 & 3 are XPU-only; on a CUDA vLLM tree they are
                   absent and skipped. Fix 1 is guarded at runtime by
                   `device.type == 'xpu'`, so it is inert on CUDA.

Usage:
  python apply_glm52_fp8_xpu_patches.py                 # auto-detect installed vllm
  python apply_glm52_fp8_xpu_patches.py --vllm-path /workspace/vllm
  python apply_glm52_fp8_xpu_patches.py --revert        # restore originals
"""

import argparse
import sys
from pathlib import Path

MARKER = "GLM52-FP8 XPU SERVE PATCH"


# --------------------------------------------------------------------------- #
# vLLM tree resolution
# --------------------------------------------------------------------------- #
def resolve_vllm_root(explicit: str | None) -> Path:
    """Return the directory that CONTAINS the `vllm` package.

    Files live at <root>/vllm/... . If --vllm-path is given, use it; otherwise
    import vllm and derive the root from its package location.
    """
    if explicit:
        root = Path(explicit)
        if not (root / "vllm").is_dir():
            sys.exit(f"error: {root}/vllm does not exist -- pass the dir that CONTAINS the vllm package")
        return root
    try:
        import vllm  # noqa: WPS433 (intentional local import)
    except Exception as exc:  # pragma: no cover - environment dependent
        sys.exit(f"error: could not import vllm to auto-detect its path ({exc}); pass --vllm-path")
    # vllm.__file__ == <root>/vllm/__init__.py  ->  root = parents[1]
    root = Path(vllm.__file__).resolve().parents[1]
    if not (root / "vllm").is_dir():
        sys.exit(f"error: derived vllm root {root} has no vllm/ package; pass --vllm-path")
    return root


# --------------------------------------------------------------------------- #
# Low-level file ops (backup / write / revert)
# --------------------------------------------------------------------------- #
def backup(path: Path) -> None:
    """Create <path>.original once, so --revert can restore the pristine file."""
    orig = path.with_suffix(path.suffix + ".original")
    if not orig.exists():
        orig.write_bytes(path.read_bytes())


def revert_file(path: Path) -> str:
    orig = path.with_suffix(path.suffix + ".original")
    if orig.exists():
        path.write_bytes(orig.read_bytes())
        orig.unlink()
        return "reverted"
    return "no backup"


def insert_after_anchor(path: Path, anchor: str, block: str) -> str:
    """Insert `block` on the line after the first line containing `anchor`,
    indented to match the anchor's leading whitespace. Idempotent via MARKER."""
    if not path.exists():
        return "absent (skipped)"
    content = path.read_text()
    if MARKER in content:
        return "already applied"
    lines = content.splitlines(keepends=True)
    out, done = [], False
    for line in lines:
        out.append(line)
        if not done and anchor in line:
            indent = line[: len(line) - len(line.lstrip())]
            indented = "".join(
                (indent + bl) if bl.strip() else bl
                for bl in block.splitlines(keepends=True)
            )
            out.append(indented)
            done = True
    if not done:
        return f"ANCHOR NOT FOUND ({anchor[:50]}...) -- upstream drifted; not patched"
    backup(path)
    path.write_text("".join(out))
    return "patched"


def replace_literals(path: Path, replacements: list[tuple[str, str]]) -> str:
    """Apply literal (old -> new) substitutions. Skips any whose `new` is
    already present (idempotent); raises loudly if an `old` anchor is missing."""
    if not path.exists():
        return "absent (skipped)"
    content = path.read_text()
    applied = 0
    for old, new in replacements:
        if new in content:
            continue
        if old not in content:
            return f"ANCHOR NOT FOUND -- upstream drifted; not patched (missing: {old[:60]}...)"
        content = content.replace(old, new, 1)
        applied += 1
    if applied == 0:
        return "already applied"
    backup(path)
    path.write_text(content)
    return f"patched ({applied} change(s))"


def replace_one_of(path: Path, old_variants: list[str], new: str) -> str:
    """Rewrite the first of several known `old` forms found in the file to `new`.

    Used where the target line has drifted across vLLM versions: we accept any
    known shape and normalize it to the patched shape. Idempotent via MARKER
    (the patched shape carries it); loud if none of the variants are present."""
    if not path.exists():
        return "absent (skipped)"
    content = path.read_text()
    if MARKER in content:
        return "already applied"
    for old in old_variants:
        if old in content:
            backup(path)
            path.write_text(content.replace(old, new, 1))
            return "patched (1 change(s))"
    return "ANCHOR NOT FOUND -- fp8_gemm call shape drifted; not patched"


# --------------------------------------------------------------------------- #
# Patch 1: XPU memory-detection fix (mem_utils.py)
# --------------------------------------------------------------------------- #
MEM_ANCHOR = "self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)"

MEM_BLOCK = f"""# {MARKER}: fix XPU memory detection bug.
# torch.accelerator.get_memory_info() returns (0, total) instead of (free, total)
# on XPU, so derive free from total minus reserved. Inert on non-XPU devices.
if device.type == 'xpu':
    if self.free_memory == 0 and self.total_memory > 0:
        try:
            reserved = torch.xpu.memory_reserved(device)
            self.free_memory = int(self.total_memory * 0.9 - reserved)
            if self.free_memory < 0:
                self.free_memory = 0
        except Exception:
            self.free_memory = int(self.total_memory * 0.8)
# END {MARKER}
"""


# --------------------------------------------------------------------------- #
# Patch 2: XPU sparse-MLA backend sync (xpu_mla_sparse.py)
# --------------------------------------------------------------------------- #
SPARSE_REPLACEMENTS: list[tuple[str, str]] = [
    (
        "from vllm.v1.kv_cache_interface import AttentionSpec",
        "from vllm.v1.kv_cache_interface import AttentionSpec\n"
        "from vllm.v1.attention.backends.utils import split_decodes_and_prefills",
    ),
    (
        "    num_actual_tokens: int  # Number of tokens excluding padding.\n"
        "    query_start_loc: torch.Tensor",
        "    num_actual_tokens: int  # Number of tokens excluding padding.\n"
        "    num_decode_tokens: int  # Tokens belonging to decode requests.\n"
        "    num_decodes: int  # Number of decode requests.\n"
        "    num_prefills: int  # Number of prefill requests.\n"
        "    query_start_loc: torch.Tensor",
    ),
    (
        "    block_size: int = 1\n"
        "    topk_tokens: int = 2048",
        "    block_size: int = 1\n"
        "    topk_tokens: int = 2048\n"
        "    # Sparse FlashMLA uses the MQA path for BOTH prefill and decode, so we\n"
        "    # never build a separate MHA prefill metadata; leaving prefill=None makes\n"
        "    # the shared MLAAttention.forward_impl route all tokens through forward_mqa.\n"
        "    prefill_max_seq_len: int = 0\n"
        "    prefill: object = None",
    ),
    (
        "        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]\n\n"
        "        metadata = XPUMLASparseMetadata(\n"
        "            num_reqs=common_attn_metadata.num_reqs,\n"
        "            max_query_len=common_attn_metadata.max_query_len,\n"
        "            max_seq_len=common_attn_metadata.max_seq_len,\n"
        "            num_actual_tokens=common_attn_metadata.num_actual_tokens,",
        "        req_id_per_token = self.req_id_per_token_buffer[:num_tokens]\n\n"
        "        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(\n"
        "            common_attn_metadata, decode_threshold=1\n"
        "        )\n\n"
        "        metadata = XPUMLASparseMetadata(\n"
        "            num_reqs=common_attn_metadata.num_reqs,\n"
        "            max_query_len=common_attn_metadata.max_query_len,\n"
        "            max_seq_len=common_attn_metadata.max_seq_len,\n"
        "            num_actual_tokens=common_attn_metadata.num_actual_tokens,\n"
        "            num_decode_tokens=num_decode_tokens,\n"
        "            num_decodes=num_decodes,\n"
        "            num_prefills=num_prefills,\n"
        "            prefill_max_seq_len=common_attn_metadata.max_seq_len,",
    ),
    (
        "class XPUMLASparseImpl(MLAAttentionImpl[XPUMLASparseMetadata]):\n"
        "    is_sparse = True\n",
        "class XPUMLASparseImpl(MLAAttentionImpl[XPUMLASparseMetadata]):\n"
        "    is_sparse = True\n"
        "    # The refactored shared MLAAttention.forward_impl queries these on every\n"
        "    # sparse impl. XPU sparse only implements the MQA path (forward_mqa handles\n"
        "    # BOTH prefill and decode), so hard-disable every MHA branch to force the\n"
        "    # whole batch through forward_mqa.\n"
        "    masked_mha_available = False\n"
        "    supports_quant_query_input = False\n"
        "    dcp_world_size = 1\n"
        "    pcp_world_size = 1\n\n"
        "    @staticmethod\n"
        "    def masked_mha_workspace_fits(prefill) -> bool:\n"
        "        return False\n",
    ),
]


# --------------------------------------------------------------------------- #
# Patch 3: FP8 block-scaled GEMM N-padding (scaled_mm/xpu.py)
# --------------------------------------------------------------------------- #
# The bare fp8_gemm call has appeared in two known shapes across vLLM versions:
# the accuracy-tool-era form (`Bs.t()`) and the current form that already carries
# the transposed-scale contiguity fix (`Bs.t().contiguous()`). Both still LACK the
# ragged-N pad, so we match whichever is present and rewrite it to the padded form.
# The replacement uses Bst = Bs.t().contiguous() in every path, so it is correct
# regardless of which variant it replaced.
GEMM_OLD_VARIANTS: list[str] = [
    "        return torch.ops._xpu_C.fp8_gemm(\n"
    "            A,\n"
    "            B.t(),\n"
    "            self.config.out_dtype,\n"
    "            As,\n"
    "            Bs.t(),\n"
    "            torch.Tensor(),\n"
    "        )",
    "        return torch.ops._xpu_C.fp8_gemm(\n"
    "            A,\n"
    "            B.t(),\n"
    "            self.config.out_dtype,\n"
    "            As,\n"
    "            Bs.t().contiguous(),\n"
    "            torch.Tensor(),\n"
    "        )",
]

GEMM_NEW = (
    "        # " + MARKER + ": oneDNN XPU matmul (v3.12.0) rejects grouped scales\n"
    "        # along N when N is not a multiple of the 128 block size (unsupported\n"
    "        # scales configuration, matmul.cpp:311). GLM-5.2 MLA fused_qkv_a_proj has\n"
    "        # N=2624=20*128+64 (ragged last block). Pad N up to a multiple of 128 with\n"
    "        # zero rows (they map to the existing final scale block and contribute 0),\n"
    "        # run the gemm, then slice back to N.\n"
    "        N = B.shape[0]\n"
    "        Bst = Bs.t().contiguous()\n"
    "        if N % 128 != 0:\n"
    "            Npad = ((N + 127) // 128) * 128\n"
    "            Bpad = torch.zeros(Npad, B.shape[1], dtype=B.dtype, device=B.device)\n"
    "            Bpad[:N] = B\n"
    "            out = torch.ops._xpu_C.fp8_gemm(\n"
    "                A, Bpad.t(), self.config.out_dtype, As, Bst, torch.Tensor()\n"
    "            )\n"
    "            return out[:, :N]\n"
    "        return torch.ops._xpu_C.fp8_gemm(\n"
    "            A,\n"
    "            B.t(),\n"
    "            self.config.out_dtype,\n"
    "            As,\n"
    "            Bst,\n"
    "            torch.Tensor(),\n"
    "        )"
)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vllm-path", help="Dir that contains the vllm package (default: auto-detect installed vllm)")
    ap.add_argument("--revert", action="store_true", help="Restore .original backups instead of patching")
    args = ap.parse_args()

    root = resolve_vllm_root(args.vllm_path)
    targets = {
        "1. XPU memory detection": root / "vllm/utils/mem_utils.py",
        "2. sparse-MLA backend  ": root / "vllm/v1/attention/backends/mla/xpu_mla_sparse.py",
        "3. FP8 GEMM N-pad      ": root / "vllm/model_executor/kernels/linear/scaled_mm/xpu.py",
    }

    print(f"vLLM tree: {root}\n")

    if args.revert:
        for label, path in targets.items():
            print(f"  [{label}] {revert_file(path)}  ({path})")
        print("\nRevert complete.")
        return 0

    results = {
        "1. XPU memory detection": insert_after_anchor(targets["1. XPU memory detection"], MEM_ANCHOR, MEM_BLOCK),
        "2. sparse-MLA backend  ": replace_literals(targets["2. sparse-MLA backend  "], SPARSE_REPLACEMENTS),
        "3. FP8 GEMM N-pad      ": replace_one_of(targets["3. FP8 GEMM N-pad      "], GEMM_OLD_VARIANTS, GEMM_NEW),
    }

    failed = False
    for label, status in results.items():
        print(f"  [{label}] {status}  ({targets[label]})")
        if "NOT FOUND" in status:
            failed = True

    if failed:
        print(
            "\nWARNING: at least one anchor was not found -- your vLLM tree has drifted\n"
            "from the version these patches target, or already carries the fix. Inspect\n"
            "the file(s) above before serving. Nothing was half-written (each fix is\n"
            "all-or-nothing)."
        )
        return 1

    print(
        "\nDone. Reminders for a real GLM-5.2-FP8 XPU serve:\n"
        "  * TP>1 needs these env vars (not source patches):\n"
        "      CCL_ENABLE_SYCL_KERNELS=1\n"
        "      CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0\n"
        "  * Rebuild is NOT required (all three are pure-Python edits).\n"
        "  * Re-run with --revert to restore the original files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

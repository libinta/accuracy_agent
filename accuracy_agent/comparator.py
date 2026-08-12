"""Tensor comparison utilities with adaptive tolerance."""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

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
        effective_threshold = 1e-3  # 10x looser than default (1e-3 vs 1e-4) for very small values
    elif max_val < 1.0:
        effective_threshold = rel_threshold
    else:
        effective_threshold = rel_threshold / 10  # Tighter for large values (stricter by 10x)

    # Match if both metrics pass
    match = (max_rel_error < effective_threshold) and (cosine_sim > cos_threshold)

    return ComparisonResult(
        match=match,
        cosine_similarity=cosine_sim,
        max_rel_error=max_rel_error,
        max_abs_error=max_abs_error
    )

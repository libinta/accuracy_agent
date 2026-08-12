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

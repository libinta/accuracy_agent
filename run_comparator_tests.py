#!/usr/bin/env python3
"""Test runner for comparator tests."""
import sys
import traceback
import torch

def run_tests():
    """Run all comparator tests and report results."""
    passed = 0
    failed = 0

    # Test 1: test_compare_identical_tensors
    try:
        from accuracy_agent.comparator import compare_tensors, ComparisonResult

        t1 = torch.randn(10, 20)
        t2 = t1.clone()

        result = compare_tensors(t1, t2)

        assert result.match is True
        assert result.cosine_similarity > 0.9999
        assert result.max_rel_error < 1e-6
        print("✓ test_compare_identical_tensors PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_compare_identical_tensors FAILED")
        traceback.print_exc()
        failed += 1

    # Test 2: test_compare_slightly_different_tensors
    try:
        from accuracy_agent.comparator import compare_tensors, ComparisonResult

        t1 = torch.randn(10, 20)
        t2 = t1 + torch.randn(10, 20) * 1e-5  # Small noise

        result = compare_tensors(t1, t2)

        assert result.match is True
        assert result.cosine_similarity > 0.999
        print("✓ test_compare_slightly_different_tensors PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_compare_slightly_different_tensors FAILED")
        traceback.print_exc()
        failed += 1

    # Test 3: test_compare_divergent_tensors
    try:
        from accuracy_agent.comparator import compare_tensors, ComparisonResult

        t1 = torch.randn(10, 20)
        t2 = torch.randn(10, 20)  # Completely different

        result = compare_tensors(t1, t2)

        assert result.match is False
        assert result.cosine_similarity < 0.99
        print("✓ test_compare_divergent_tensors PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_compare_divergent_tensors FAILED")
        traceback.print_exc()
        failed += 1

    # Test 4: test_compare_integer_tensors_exact
    try:
        from accuracy_agent.comparator import compare_tensors, ComparisonResult

        t1 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        t2 = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32)
        t3 = torch.tensor([1, 2, 3, 4, 6], dtype=torch.int32)

        result_match = compare_tensors(t1, t2)
        result_diff = compare_tensors(t1, t3)

        assert result_match.match is True
        assert result_diff.match is False
        print("✓ test_compare_integer_tensors_exact PASSED")
        passed += 1
    except Exception as e:
        print(f"✗ test_compare_integer_tensors_exact FAILED")
        traceback.print_exc()
        failed += 1

    print(f"\n{'='*50}")
    print(f"Tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

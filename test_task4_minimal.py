#!/usr/bin/env python3
"""Minimal test runner for Task 4 - checks code structure without requiring dependencies."""
import sys
import ast
from pathlib import Path

def check_bisector_has_method(file_path, method_name):
    """Check if a file contains a method with given name."""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return True
    return False

def get_method_signature(file_path, method_name):
    """Extract method signature."""
    with open(file_path, 'r') as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            args = [arg.arg for arg in node.args.args]
            return args
    return None

def main():
    """Run structural tests."""
    print("=" * 60)
    print("Task 4: Parallel Layer Execution - Structural Verification")
    print("=" * 60)

    passed = 0
    failed = 0

    # Test 1: Check _test_layer_range_parallel exists
    print("\n[Test 1] Checking _test_layer_range_parallel method exists...")
    bisector_file = Path("/home/youruser/github/accuracy_agent/accuracy_agent/bisector.py")
    if check_bisector_has_method(str(bisector_file), "_test_layer_range_parallel"):
        print("✓ PASS: _test_layer_range_parallel method found")
        passed += 1
    else:
        print("✗ FAIL: _test_layer_range_parallel method not found")
        failed += 1

    # Test 2: Check method signature
    print("\n[Test 2] Checking _test_layer_range_parallel signature...")
    sig = get_method_signature(str(bisector_file), "_test_layer_range_parallel")
    expected = ['self', 'layer_start', 'layer_end']
    if sig == expected:
        print(f"✓ PASS: Signature correct: {sig}")
        passed += 1
    else:
        print(f"✗ FAIL: Expected {expected}, got {sig}")
        failed += 1

    # Test 3: Check bisect_layers updated
    print("\n[Test 3] Checking bisect_layers uses parallel execution...")
    with open(str(bisector_file), 'r') as f:
        content = f.read()

    if "_test_layer_range_parallel" in content and "self.use_backends" in content:
        print("✓ PASS: bisect_layers contains parallel execution calls")
        passed += 1
    else:
        print("✗ FAIL: bisect_layers does not use parallel execution")
        failed += 1

    # Test 4: Check test file has new test
    print("\n[Test 4] Checking test file has test_parallel_layer_execution...")
    test_file = Path("/home/youruser/github/accuracy_agent/tests/test_bisector_parallel.py")
    if check_bisector_has_method(str(test_file), "test_parallel_layer_execution"):
        print("✓ PASS: test_parallel_layer_execution found")
        passed += 1
    else:
        print("✗ FAIL: test_parallel_layer_execution not found")
        failed += 1

    # Test 5: Verify concurrent.futures import
    print("\n[Test 5] Checking concurrent.futures import...")
    if "import concurrent.futures" in content or "from concurrent" in content:
        print("✓ PASS: concurrent.futures imported")
        passed += 1
    else:
        print("✗ FAIL: concurrent.futures not imported")
        failed += 1

    # Test 6: Verify ThreadPoolExecutor usage
    print("\n[Test 6] Checking ThreadPoolExecutor usage...")
    if "ThreadPoolExecutor" in content:
        print("✓ PASS: ThreadPoolExecutor used for parallelism")
        passed += 1
    else:
        print("✗ FAIL: ThreadPoolExecutor not used")
        failed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Tests run: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 60)

    if failed == 0:
        print("\n✓✓✓ All structural tests PASSED ✓✓✓\n")
        return 0
    else:
        print(f"\n✗✗✗ {failed} test(s) FAILED ✗✗✗\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""Test that patch insertion mechanism works correctly"""
import tempfile
from pathlib import Path
from accuracy_agent.backends.vllm.patches import get_weight_filter_patch, get_layer_init_patch


def test_weight_filter_patch_is_valid_python():
    """Verify weight filter patch is syntactically valid Python"""
    import ast
    patch_content = get_weight_filter_patch()

    # Should be valid Python
    try:
        ast.parse(patch_content)
    except SyntaxError as e:
        raise AssertionError(f"Weight filter patch has syntax error: {e}")


def test_layer_init_patch_is_valid_python():
    """Verify layer init patch is syntactically valid Python"""
    import ast
    patch_content = get_layer_init_patch()

    # Should be valid Python
    try:
        ast.parse(patch_content)
    except SyntaxError as e:
        raise AssertionError(f"Layer init patch has syntax error: {e}")


def test_weight_filter_patch_references_correct_variables():
    """Verify weight filter patch references function-scoped variables correctly"""
    patch_content = get_weight_filter_patch()

    # Should reference these function-scoped variables
    assert "model_config" in patch_content, "Patch should reference model_config"
    assert "weights_iterator" in patch_content, "Patch should reference weights_iterator"
    assert "layer_start" in patch_content, "Patch should reference layer_start"
    assert "layer_end" in patch_content, "Patch should reference layer_end"


def test_layer_init_patch_references_correct_variables():
    """Verify layer init patch references function-scoped variables correctly"""
    patch_content = get_layer_init_patch()

    # Should reference these function-scoped variables
    assert "config" in patch_content, "Patch should reference config"
    assert "num_hidden_layers" in patch_content, "Patch should reference num_hidden_layers"
    assert "layer_start" in patch_content, "Patch should reference layer_start"
    assert "layer_end" in patch_content, "Patch should reference layer_end"


def test_patch_insertion_with_anchor():
    """Test that patch insertion with anchor works correctly"""
    # Mock a simple Python file
    original_content = """
def load_weights():
    weights_iterator = get_weights()
    for name, param in weights_iterator:
        model.load_param(name, param)
"""

    patch_content = """
# Filter weights
if hasattr(config, 'debug_mode'):
    weights_iterator = filter_weights(weights_iterator)
"""

    # Simulate anchor-based insertion
    anchor = "for name, param in weights_iterator:"
    lines = original_content.splitlines(keepends=True)
    patched_lines = []

    for line in lines:
        if anchor in line:
            # Detect indentation
            indent = len(line) - len(line.lstrip())
            indent_str = line[:indent]

            # Add patch before anchor
            patch_lines = patch_content.splitlines(keepends=True)
            for patch_line in patch_lines:
                if patch_line.strip():
                    patched_lines.append(indent_str + patch_line)
                else:
                    patched_lines.append(patch_line)

        patched_lines.append(line)

    result = "".join(patched_lines)

    # Verify patch was inserted at correct location
    assert "# Filter weights" in result
    assert result.index("# Filter weights") < result.index("for name, param")

    # Verify indentation is correct
    lines_list = result.split('\n')
    filter_line = [l for l in lines_list if "# Filter weights" in l][0]
    for_line = [l for l in lines_list if "for name, param" in l][0]

    # Both should have same indentation
    filter_indent = len(filter_line) - len(filter_line.lstrip())
    for_indent = len(for_line) - len(for_line.lstrip())
    assert filter_indent == for_indent, f"Indentation mismatch: {filter_indent} != {for_indent}"


def test_patch_doesnt_create_9999_layers():
    """Verify patches don't hardcode 9999 layers"""
    from accuracy_agent.backends.vllm.patches.debug_runner import run_partial_layers
    import inspect

    source = inspect.getsource(run_partial_layers)

    # Should not use 9999 for layer_end anymore
    # (it should use num_hidden_layers from config)
    if "9999" in source:
        # Check context - it should only appear in comments, not actual code
        for line in source.split('\n'):
            if "9999" in line and not line.strip().startswith('#'):
                raise AssertionError(
                    f"Found hardcoded 9999 in non-comment line: {line.strip()}"
                )

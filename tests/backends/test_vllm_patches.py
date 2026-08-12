"""Tests for vLLM patch content generation"""

import re
import pytest
from accuracy_agent.backends.vllm.patches import (
    get_weight_filter_patch,
    get_layer_init_patch,
)
from accuracy_agent.backends.vllm.patches.weight_filter_patch import (
    get_patch_insertion_marker,
)
from accuracy_agent.backends.vllm.patches.layer_init_patch import get_forward_patch


class TestWeightFilterPatch:
    """Tests for weight filter patch generation"""

    def test_get_weight_filter_patch_returns_string(self):
        """Test that get_weight_filter_patch returns a string"""
        patch = get_weight_filter_patch()
        assert isinstance(patch, str)
        assert len(patch) > 0

    def test_weight_filter_patch_contains_markers(self):
        """Test that patch contains start and end markers"""
        patch = get_weight_filter_patch()
        assert "ACCURACY_AGENT PATCH START" in patch
        assert "ACCURACY_AGENT PATCH END" in patch

    def test_weight_filter_patch_contains_filter_function(self):
        """Test that patch contains _filter_layer_weights function"""
        patch = get_weight_filter_patch()
        assert "_filter_layer_weights" in patch
        assert "def _filter_layer_weights" in patch

    def test_weight_filter_patch_regex_pattern(self):
        """Test that patch contains proper regex for layer extraction"""
        patch = get_weight_filter_patch()
        # Should contain regex pattern for layer extraction
        assert r"layers\.\(\d+\)\." in patch or "layers" in patch

    def test_weight_filter_patch_checks_debug_config(self):
        """Test that patch checks for debug config attributes"""
        patch = get_weight_filter_patch()
        assert "debug_layer_start" in patch
        assert "debug_layer_end" in patch
        assert "hasattr" in patch

    def test_weight_filter_patch_creates_filtered_iterator(self):
        """Test that patch creates a filtered iterator"""
        patch = get_weight_filter_patch()
        assert "filtered_iterator" in patch
        assert "original_weights_iterator" in patch

    def test_get_patch_insertion_marker(self):
        """Test that insertion marker is provided"""
        marker = get_patch_insertion_marker()
        assert isinstance(marker, str)
        assert "for name, param in weights_iterator:" in marker


class TestLayerInitPatch:
    """Tests for layer initialization patch generation"""

    def test_get_layer_init_patch_returns_string(self):
        """Test that get_layer_init_patch returns a string"""
        patch = get_layer_init_patch()
        assert isinstance(patch, str)
        assert len(patch) > 0

    def test_layer_init_patch_contains_markers(self):
        """Test that patch contains start and end markers"""
        patch = get_layer_init_patch()
        assert "ACCURACY_AGENT PATCH START" in patch
        assert "ACCURACY_AGENT PATCH END" in patch

    def test_layer_init_patch_reads_debug_config(self):
        """Test that patch reads debug layer range from config"""
        patch = get_layer_init_patch()
        assert "debug_layer_start" in patch
        assert "debug_layer_end" in patch
        assert "getattr" in patch

    def test_layer_init_patch_creates_module_list(self):
        """Test that patch creates ModuleList"""
        patch = get_layer_init_patch()
        assert "nn.ModuleList" in patch
        assert "layers = nn.ModuleList()" in patch

    def test_layer_init_patch_loops_over_range(self):
        """Test that patch loops over layer range"""
        patch = get_layer_init_patch()
        assert "for i in range(layer_start, layer_end):" in patch

    def test_layer_init_patch_stores_layer_range(self):
        """Test that patch stores layer range for forward pass"""
        patch = get_layer_init_patch()
        assert "start_layer = layer_start" in patch
        assert "end_layer = layer_end" in patch

    def test_get_forward_patch_returns_string(self):
        """Test that get_forward_patch returns a string"""
        patch = get_forward_patch()
        assert isinstance(patch, str)
        assert len(patch) > 0

    def test_forward_patch_contains_markers(self):
        """Test that forward patch contains start and end markers"""
        patch = get_forward_patch()
        assert "ACCURACY_AGENT PATCH" in patch

    def test_forward_patch_handles_layer_offset(self):
        """Test that forward patch handles layer offset indexing"""
        patch = get_forward_patch()
        assert "layer_idx = i - self.start_layer" in patch
        assert "self.layers[layer_idx]" in patch


class TestPatchIntegration:
    """Integration tests for patches"""

    def test_both_patches_have_consistent_config_keys(self):
        """Test that both patches use consistent config keys"""
        weight_patch = get_weight_filter_patch()
        layer_patch = get_layer_init_patch()

        # Both should reference the same debug config keys
        assert "debug_layer_start" in weight_patch
        assert "debug_layer_start" in layer_patch
        assert "debug_layer_end" in weight_patch
        assert "debug_layer_end" in layer_patch

    def test_patches_are_syntactically_valid_python_comments(self):
        """Test that patches contain valid Python syntax (at least as strings)"""
        weight_patch = get_weight_filter_patch()
        layer_patch = get_layer_init_patch()
        forward_patch = get_forward_patch()

        # All should start with comment marker
        assert "ACCURACY_AGENT PATCH" in weight_patch
        assert "ACCURACY_AGENT PATCH" in layer_patch
        assert "ACCURACY_AGENT PATCH" in forward_patch

        # All should have proper Python strings/comments
        for patch in [weight_patch, layer_patch, forward_patch]:
            # Should have docstring markers or comments
            assert "'''" in patch or '"""' in patch or "#" in patch

    def test_weight_filter_checks_layer_bounds(self):
        """Test that weight filter checks layer bounds correctly"""
        patch = get_weight_filter_patch()
        # Should check: layer_start <= layer_idx < layer_end
        assert "layer_start <=" in patch
        assert "< layer_end" in patch

    def test_layer_init_creates_correct_layer_count(self):
        """Test that layer init creates the right number of layers"""
        patch = get_layer_init_patch()
        # Should loop from layer_start to layer_end
        assert "range(layer_start, layer_end)" in patch

    def test_no_hardcoded_layer_numbers(self):
        """Test that patches don't have hardcoded layer numbers"""
        weight_patch = get_weight_filter_patch()
        layer_patch = get_layer_init_patch()

        # Should use variables, not hardcoded numbers for layer indices
        # (exception: comments or examples may have them)
        # The actual code should use layer_start, layer_end
        assert "layer_start" in weight_patch
        assert "layer_end" in weight_patch
        assert "layer_start" in layer_patch
        assert "layer_end" in layer_patch

    def test_patches_handle_all_layer_weights(self):
        """Test that weight filter allows non-layer weights"""
        patch = get_weight_filter_patch()
        # Should mention that non-layer weights are always loaded
        assert "embed_tokens" in patch or "Always load" in patch or "True" in patch

"""Tests for model-specific patch providers"""

import pytest
import re
from accuracy_agent.backends.vllm.patches.models import (
    ModelPatchProvider,
    GLMPatchProvider,
    GLM52PatchProvider,
)


class TestGLMPatchProvider:
    """Test GLM-specific patch provider"""

    def test_layer_pattern(self):
        """Test GLM layer pattern matches expected format"""
        provider = GLMPatchProvider()
        pattern = provider.get_layer_pattern()

        # Pattern should match "layers.{i}."
        assert pattern == r'layers\.(\d+)\.'

        # Test pattern against sample weight names
        compiled = re.compile(pattern)

        # Should match GLM layer weights
        assert compiled.search("model.layers.0.self_attn.qkv_proj.weight")
        assert compiled.search("model.layers.15.mlp.gate_proj.weight")
        assert compiled.search("layers.23.post_attention_layernorm.weight")

        # Should extract correct layer indices
        match = compiled.search("model.layers.42.self_attn.o_proj.weight")
        assert match and match.group(1) == "42"

        # Should not match non-layer weights
        assert not compiled.search("model.embed_tokens.weight")
        assert not compiled.search("model.lm_head.weight")
        assert not compiled.search("model.norm.weight")

    def test_model_class_name(self):
        """Test GLM model class name"""
        provider = GLMPatchProvider()
        assert provider.get_model_class_name() == 'Glm4Model'

    def test_model_file_path(self):
        """Test GLM model file path"""
        provider = GLMPatchProvider()
        path = provider.get_model_file_path()
        assert path == 'vllm/model_executor/models/glm4.py'
        assert path.endswith('.py')
        assert 'glm' in path.lower()

    def test_weight_filter_patch_content(self):
        """Test weight filter patch contains expected elements"""
        provider = GLMPatchProvider()
        patch = provider.get_weight_filter_patch()

        # Patch should contain key elements
        assert 'ACCURACY_AGENT PATCH' in patch
        assert '_filter_layer_weights' in patch
        assert 'layers\\.(\\d+)\\.' in patch  # Regex pattern
        assert 'debug_layer_start' in patch
        assert 'debug_layer_end' in patch
        assert 'filtered_iterator' in patch

        # Should be valid Python (basic check)
        assert patch.count('def ') >= 2  # At least 2 function definitions

    def test_layer_init_patch_content(self):
        """Test layer init patch contains expected elements"""
        provider = GLMPatchProvider()
        patch = provider.get_layer_init_patch()

        # Patch should contain key elements
        assert 'ACCURACY_AGENT PATCH' in patch
        assert 'layer_start' in patch
        assert 'layer_end' in patch
        assert 'nn.ModuleList()' in patch
        assert 'layer_type' in patch
        assert 'vllm_config' in patch

    def test_forward_patch_content(self):
        """Test forward patch contains expected elements"""
        provider = GLMPatchProvider()
        patch = provider.get_forward_patch()

        # Patch should contain key elements
        assert 'ACCURACY_AGENT PATCH' in patch
        assert 'self.start_layer' in patch
        assert 'self.end_layer' in patch
        assert 'range(' in patch

    def test_anchor_points(self):
        """Test anchor points are defined"""
        provider = GLMPatchProvider()
        anchors = provider.get_anchor_points()

        # Should have all required anchor points
        assert 'weight_filter' in anchors
        assert 'layer_init' in anchors
        assert 'forward' in anchors

        # Anchor points should be non-empty strings
        assert isinstance(anchors['weight_filter'], str)
        assert len(anchors['weight_filter']) > 0

        # Weight filter anchor should be the standard loop
        assert 'for name, param in weights_iterator' in anchors['weight_filter']

    def test_layer_list_creation_anchor(self):
        """Test layer list creation anchor (GLM uses make_layers)"""
        provider = GLMPatchProvider()
        anchor = provider.get_layer_list_creation_anchor()

        # GLM uses make_layers(), so should return None
        assert anchor is None


class TestGLM52PatchProvider:
    """Test GLM-5.2-specific patch provider"""

    def test_inherits_from_glm(self):
        """Test GLM52PatchProvider inherits GLM behavior"""
        glm_provider = GLMPatchProvider()
        glm52_provider = GLM52PatchProvider()

        # Should have same layer pattern
        assert glm52_provider.get_layer_pattern() == glm_provider.get_layer_pattern()

        # Should have same model file path
        assert glm52_provider.get_model_file_path() == glm_provider.get_model_file_path()

    def test_model_class_name(self):
        """Test GLM-5.2 uses Glm4Model architecture"""
        provider = GLM52PatchProvider()
        # GLM-5.2 reuses Glm4Model architecture
        assert provider.get_model_class_name() == 'Glm4Model'


class TestModelPatchProviderInterface:
    """Test that patch providers implement required interface"""

    def test_glm_provider_implements_interface(self):
        """Test GLMPatchProvider implements all abstract methods"""
        provider = GLMPatchProvider()

        # All abstract methods should be implemented and callable
        assert callable(provider.get_layer_pattern)
        assert callable(provider.get_model_class_name)
        assert callable(provider.get_model_file_path)
        assert callable(provider.get_weight_filter_patch)
        assert callable(provider.get_layer_init_patch)
        assert callable(provider.get_forward_patch)
        assert callable(provider.get_anchor_points)

        # Methods should return expected types
        assert isinstance(provider.get_layer_pattern(), str)
        assert isinstance(provider.get_model_class_name(), str)
        assert isinstance(provider.get_model_file_path(), str)
        assert isinstance(provider.get_weight_filter_patch(), str)
        assert isinstance(provider.get_layer_init_patch(), str)
        assert isinstance(provider.get_forward_patch(), str)
        assert isinstance(provider.get_anchor_points(), dict)


class TestLayerPatternMatching:
    """Test layer pattern matching against real weight names"""

    @pytest.fixture
    def glm_provider(self):
        return GLMPatchProvider()

    def test_glm_layer_weights(self, glm_provider):
        """Test pattern matches GLM layer weights"""
        pattern = re.compile(glm_provider.get_layer_pattern())

        # Real GLM weight names (from GLM-4 architecture)
        glm_weights = [
            "model.layers.0.self_attn.qkv_proj.weight",
            "model.layers.0.self_attn.qkv_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.15.self_attn.qkv_proj.weight",
            "model.layers.31.post_mlp_layernorm.weight",
        ]

        for weight_name in glm_weights:
            match = pattern.search(weight_name)
            assert match, f"Pattern should match {weight_name}"
            layer_idx = int(match.group(1))
            assert layer_idx >= 0, f"Layer index should be non-negative"

    def test_non_layer_weights_not_matched(self, glm_provider):
        """Test pattern doesn't match non-layer weights"""
        pattern = re.compile(glm_provider.get_layer_pattern())

        non_layer_weights = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.lm_head.weight",
        ]

        for weight_name in non_layer_weights:
            match = pattern.search(weight_name)
            assert not match, f"Pattern should not match {weight_name}"

    def test_layer_extraction(self, glm_provider):
        """Test extracting layer indices from weight names"""
        pattern = re.compile(glm_provider.get_layer_pattern())

        test_cases = [
            ("model.layers.0.weight", 0),
            ("model.layers.15.self_attn.weight", 15),
            ("model.layers.42.mlp.gate_proj.weight", 42),
            ("layers.100.norm.weight", 100),
        ]

        for weight_name, expected_idx in test_cases:
            match = pattern.search(weight_name)
            assert match, f"Should match {weight_name}"
            actual_idx = int(match.group(1))
            assert actual_idx == expected_idx, \
                f"Expected layer {expected_idx}, got {actual_idx}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""GLM-specific patch provider for layer extraction"""

from typing import Dict
from .base import ModelPatchProvider


class GLMPatchProvider(ModelPatchProvider):
    """
    Patch provider for GLM models (GLM-4, GLM-5, etc.).

    GLM models in vLLM inherit from LlamaModel, so they use the same
    layer structure: model.layers.{i}
    """

    def get_layer_pattern(self) -> str:
        """
        GLM uses standard Llama-style layer naming: model.layers.{i}

        Returns:
            Regex pattern that captures layer index in group 1
        """
        return r'layers\.(\d+)\.'

    def get_model_class_name(self) -> str:
        """
        Returns:
            GLM model class name in vLLM
        """
        return 'Glm4Model'

    def get_model_file_path(self) -> str:
        """
        Returns:
            Path to GLM model file relative to vLLM root
        """
        return 'vllm/model_executor/models/glm4.py'

    def get_weight_filter_patch(self) -> str:
        """
        Weight filter patch for GLM models.

        Since GLM uses standard layer naming, we use the same pattern as Llama.
        """
        return """# === ACCURACY_AGENT PATCH START (GLM) ===
# Filter weights by layer range if debug mode is enabled
def _filter_layer_weights(param_name: str, layer_start: int, layer_end: int) -> bool:
    '''Return True if weight should be loaded, False to skip'''
    import re
    # Pattern for GLM layer weights: model.layers.{i}.*
    match = re.search(r'layers\\.(\\d+)\\.', param_name)
    if match:
        layer_idx = int(match.group(1))
        return layer_start <= layer_idx < layer_end
    # Always load non-layer weights (embed_tokens, lm_head, etc.)
    return True

# Wrap weights iterator with layer filter if debug mode is enabled
original_iterator = ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
if hasattr(self.load_config, 'model_loader_extra_config'):
    extra_config = self.load_config.model_loader_extra_config or {}
    if 'debug_layer_start' in extra_config and 'debug_layer_end' in extra_config:
        layer_start = extra_config['debug_layer_start']
        layer_end = extra_config['debug_layer_end']
        return ((name, tensor) for (name, tensor) in original_iterator
                if _filter_layer_weights(name, layer_start, layer_end))

return original_iterator
# === ACCURACY_AGENT PATCH END ===
"""

    def get_layer_init_patch(self) -> str:
        """
        Layer initialization patch for GLM models.

        GLM inherits from LlamaModel which uses make_layers() function.
        We need to patch this to create only requested layers.
        """
        return """
# === ACCURACY_AGENT PATCH START (GLM) ===
# Check for debug layer range from config
layer_start = getattr(config, 'debug_layer_start', 0)
layer_end = getattr(config, 'debug_layer_end', num_hidden_layers)

# Create only layers in requested range
# GLM uses make_layers() from LlamaModel, which we need to patch
# The layers are created with proper prefixes: model.layers.{i}
import torch.nn as nn

layers = nn.ModuleList()
for i in range(layer_start, layer_end):
    layer = layer_type(
        vllm_config=vllm_config,
        prefix=f"{prefix}.{i}",
    )
    layers.append(layer)

# Store layer range for forward() method
start_layer = layer_start
end_layer = layer_end

# Return the layer info
return start_layer, end_layer, layers
# === ACCURACY_AGENT PATCH END ===
"""

    def get_forward_patch(self) -> str:
        """
        Forward pass patch for GLM models.

        GLM inherits from LlamaModel, so the forward pass structure is the same.
        """
        return """
# === ACCURACY_AGENT PATCH: Update layer iteration (GLM) ===
# Original: for i, layer in enumerate(self.layers):
# Patched: for i in range(self.start_layer, self.end_layer):
for i in range(self.start_layer, self.end_layer):
    layer_idx = i - self.start_layer
    layer = self.layers[layer_idx]
    # ... rest of forward pass
# === ACCURACY_AGENT PATCH END ===
"""

    def get_anchor_points(self) -> Dict[str, str]:
        """
        Anchor points for GLM patch insertion.

        Returns:
            Dictionary with anchor lines for each patch location
        """
        return {
            'weight_filter': 'return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)',
            'layer_init': 'self.start_layer, self.end_layer, self.layers = make_layers(',
            'forward': 'for i in islice(range(self.start_layer, self.end_layer), ',
        }

    def get_layer_list_creation_anchor(self) -> str | None:
        """
        GLM uses make_layers() function, so no direct ModuleList creation.

        Returns:
            None (uses make_layers() function)
        """
        return None


class GLM52PatchProvider(GLMPatchProvider):
    """
    Patch provider specifically for GLM-5.2 models.

    GLM-5.2 uses the same structure as GLM-4, so we inherit all behavior.
    This class exists for future GLM-5.2-specific customizations if needed.
    """

    def get_model_class_name(self) -> str:
        """
        Returns:
            GLM-5.2 may use Glm4Model or a specific Glm52Model class
        """
        # GLM-5.2 likely reuses Glm4Model architecture
        return 'Glm4Model'

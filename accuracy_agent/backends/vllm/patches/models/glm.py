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
    import os as _aa_os2
    # Optional: skip the vision tower entirely (multimodal VL models). Its
    # params (visual.* / vision_tower.* / vision_model.*) are NOT named
    # layers.{i}, so the layer-range filter below always keeps them, leaving the
    # whole vision encoder resident in HBM even for a text-only hidden-state
    # extraction. Off by default (dropping weights vLLM's loader expects can
    # raise a missing-weight error); enable with ACCURACY_SKIP_VISION=1 when the
    # decoder MoE op needs the extra contiguous HBM.
    if _aa_os2.environ.get('ACCURACY_SKIP_VISION') in ('1', 'true', 'True'):
        if re.match(r'(model\\.)?(visual|vision_tower|vision_model)\\.', param_name):
            return False
    # Pattern for GLM layer weights: model.layers.{i}.*
    match = re.search(r'layers\\.(\\d+)\\.', param_name)
    if match:
        layer_idx = int(match.group(1))
        return layer_start <= layer_idx < layer_end
    # Always load non-layer weights (embed_tokens, lm_head, etc.)
    return True

# Wrap weights iterator with layer filter if debug mode is enabled.
# Activation is via env vars (set by debug_runner before vLLM import and
# inherited by the in-process EngineCore). We avoid model_loader_extra_config
# because newer vLLM strictly whitelists its keys and rejects unknown ones.
import os as _aa_os
original_iterator = ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)
_aa_start = _aa_os.environ.get('ACCURACY_DEBUG_LAYER_START')
_aa_end = _aa_os.environ.get('ACCURACY_DEBUG_LAYER_END')
if _aa_start is not None and _aa_end is not None:
    layer_start = int(_aa_start)
    layer_end = int(_aa_end)
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
            GLM-5.2 is served by vLLM's DeepseekV2Model (arch
            GlmMoeDsaForCausalLM -> deepseek_v2.py), not Glm4Model.
        """
        return 'DeepseekV2Model'

    def get_model_file_path(self) -> str:
        """
        GLM-5.2 (model_type=glm_moe_dsa, arch GlmMoeDsaForCausalLM) is served
        by vLLM's deepseek_v2.py -- NOT glm4.py. Point the model-file patch at
        the file that actually backs the architecture so the anchor resolves.
        """
        return 'vllm/model_executor/models/deepseek_v2.py'

    def get_layer_init_patch(self) -> str:
        """No-op the model-file layer-init patch for GLM-5.2.

        The inherited GLM patch body references `layer_type`/`num_hidden_layers`
        and emits a mid-__init__ `return`, which is valid for glm4.py's
        make_layers site but would BREAK DeepseekV2Model.__init__ (those names
        don't exist there and the early return aborts module construction).
        Layer-window limiting is already handled architecture-agnostically by
        the make_layers clamp (_apply_make_layers_fix, utils.py) plus the
        env-based weight filter, so no model-file layer-init edit is needed
        here. Returning a comment-only body makes the anchored insertion inert
        (it inserts a comment before the make_layers call and changes nothing).
        """
        return (
            "# === ACCURACY_AGENT PATCH (GLM-5.2): layer-init handled by "
            "make_layers clamp; no model-file edit needed ===\n"
        )

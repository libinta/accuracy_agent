"""Patch content for vLLM llama.py - creates only requested layers"""


def get_layer_init_patch() -> str:
    """
    Returns patch code to inject into llama.py make_layers() function

    This patch modifies layer initialization to create only layers in requested range.
    Insert location: Replace the line that creates the layers ModuleList.
    """
    return """
# === ACCURACY_AGENT PATCH START ===
# Check for debug layer range from config
layer_start = getattr(config, 'debug_layer_start', 0)
layer_end = getattr(config, 'debug_layer_end', num_hidden_layers)

# Create only layers in requested range
layers = nn.ModuleList()
for i in range(layer_start, layer_end):
    layer = layer_type(
        config=config,
        cache_config=cache_config,
        quant_config=quant_config,
        layer_idx=i,
        prefix=f"{prefix}.layers.{i}",
    )
    layers.append(layer)

# Update forward pass to use offset indexing
# Store layer range for forward() method
start_layer = layer_start
end_layer = layer_end
# === ACCURACY_AGENT PATCH END ===
"""


def get_forward_patch() -> str:
    """
    Returns patch for LlamaModel.forward() to handle layer offset.

    NOTE: This is a template/placeholder. The "# ... rest of forward pass" comment
    indicates that this patch must be completed during actual vLLM source patching
    (Task 3). The remaining forward pass logic should be extracted from the actual
    vLLM llama.py file and inserted after the layer iteration loop.
    """
    return """
# === ACCURACY_AGENT PATCH: Update layer iteration ===
# Original: for i, layer in enumerate(self.layers):
# Patched: for i in range(self.start_layer, self.end_layer):
for i in range(self.start_layer, self.end_layer):
    layer_idx = i - self.start_layer
    layer = self.layers[layer_idx]
    # ... rest of forward pass
# === ACCURACY_AGENT PATCH END ===
"""

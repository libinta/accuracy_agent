"""Patch content for vLLM default_loader.py - filters layer weights during loading"""


def get_weight_filter_patch() -> str:
    """
    Returns patch code to inject into default_loader.py

    This patch filters weight loading to skip layers outside the requested range.
    Insert location: At the start of DefaultModelLoader.load_weights() method,
    after getting the weights iterator.
    """
    return """
# === ACCURACY_AGENT PATCH START ===
def _filter_layer_weights(param_name: str, layer_start: int, layer_end: int) -> bool:
    '''Return True if weight should be loaded, False to skip'''
    import re
    match = re.search(r'layers\\.(\\d+)\\.', param_name)
    if match:
        layer_idx = int(match.group(1))
        return layer_start <= layer_idx < layer_end
    # Always load non-layer weights (embed_tokens, lm_head, etc.)
    return True

# Wrap weights iterator with layer filter
if hasattr(model_config.hf_config, 'debug_layer_start'):
    layer_start = model_config.hf_config.debug_layer_start
    layer_end = model_config.hf_config.debug_layer_end
    original_weights_iterator = weights_iterator

    def filtered_iterator():
        for name, param in original_weights_iterator:
            if _filter_layer_weights(name, layer_start, layer_end):
                yield name, param

    weights_iterator = filtered_iterator()
# === ACCURACY_AGENT PATCH END ===
"""


def get_patch_insertion_marker() -> str:
    """Returns the line marker where patch should be inserted in default_loader.py"""
    return "for name, param in weights_iterator:"

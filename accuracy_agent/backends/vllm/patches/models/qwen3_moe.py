"""Qwen3-MoE patch provider for layer extraction.

Qwen3-MoE (e.g. Qwen3-*-A*B, Qwen3.5/3.6 MoE) uses Llama-style layer naming
(``model.layers.{i}``) and builds its decoder stack via the shared
``make_layers()`` helper, so it can reuse the GLM provider's weight-filter and
layer-init patches unchanged. Only the target model file and class name differ.
"""

from .glm import GLMPatchProvider


class Qwen3MoePatchProvider(GLMPatchProvider):
    """Patch provider for Qwen3-MoE models."""

    def get_model_class_name(self) -> str:
        """Returns the vLLM decoder model class name for Qwen3.5/3.6 (VL) MoE."""
        return 'Qwen3_5Model'

    def get_model_file_path(self) -> str:
        """Returns the Qwen3.5/3.6 model file path relative to the vLLM root."""
        return 'vllm/model_executor/models/qwen3_5.py'

    def get_layer_init_patch(self) -> str:
        """No model-file layer-init surgery needed.

        Layer construction is limited generically by the shared make_layers()
        clamp (_apply_make_layers_fix), so we return a harmless marker instead
        of GLM's early-return block (which would corrupt the model __init__).
        """
        return (
            "# === ACCURACY_AGENT PATCH (Qwen3.5): layer limiting handled "
            "by the shared make_layers clamp ===\n"
        )

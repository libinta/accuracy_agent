"""
Manual test script for vLLM backend on real hardware.

Usage:
    python scripts/test_vllm_backend.py examples/glm52_vllm_config.yaml

This script will:
1. Load config
2. Create GPU and XPU backends
3. Apply patches
4. Run layer extraction on both
5. Compare results
6. Clean up
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from accuracy_agent.config import DebugConfig
from accuracy_agent.backends import create_backend
from accuracy_agent.backends.base import BackendConfig
from accuracy_agent.comparator import compare_tensors

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Test vLLM backend on real hardware.

    Loads configuration, creates GPU and XPU backends, applies patches,
    runs layer extraction on both, and compares hidden states.

    Parameters:
        None (arguments parsed from command line)

    Returns:
        int: Exit code (0 on success, 1 on vLLM not found or other error)
    """
    parser = argparse.ArgumentParser(
        description="Test vLLM backend on real hardware"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--skip-xpu",
        action="store_true",
        help="Only test GPU backend"
    )
    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = DebugConfig.from_yaml(args.config)

    # Validate config and required attributes
    try:
        config.validate()
    except ValueError as e:
        logger.error(f"Config validation failed: {e}")
        return 1

    # Verify required attributes for GPU backend
    required_attrs = ['gpu_host', 'gpu_user', 'gpu_docker', 'gpu_vllm_path', 'gpu_cards']
    for attr in required_attrs:
        value = getattr(config, attr, None)
        if not value:
            logger.error(f"Config missing required attribute: {attr}")
            return 1

    # Verify required attributes for XPU backend (unless skipping)
    if not args.skip_xpu:
        xpu_required = ['xpu_host', 'xpu_user', 'xpu_docker', 'xpu_vllm_path', 'xpu_cards']
        for attr in xpu_required:
            value = getattr(config, attr, None)
            if not value:
                logger.error(f"Config missing required attribute: {attr}")
                return 1

    # Validate layer range bounds
    if config.layer_start < 0 or config.layer_end < 0:
        logger.error(f"Layer indices must be non-negative: layer_start={config.layer_start}, layer_end={config.layer_end}")
        return 1

    # Create GPU backend
    logger.info("Creating GPU backend")
    gpu_backend_config = BackendConfig(
        host=config.gpu_host,
        user=config.gpu_user,
        docker=config.gpu_docker,
        vllm_path=config.gpu_vllm_path,
        cards=config.gpu_cards,
        device_type="cuda"
    )
    gpu_backend = create_backend(
        config.backend,
        gpu_backend_config,
        config.model_path,
        config.shared_fs
    )

    try:
        # Setup GPU backend
        logger.info("Setting up GPU backend (applying patches)")
        gpu_backend.setup()

        # Check availability
        if not gpu_backend.is_available():
            logger.error(f"vLLM not found at {config.gpu_vllm_path} in {config.gpu_docker}")
            return 1

        # Run layer extraction on GPU
        logger.info(f"Running layers [{config.layer_start}, {config.layer_end}) on GPU")
        gpu_hidden_states = gpu_backend.run_layer_range(
            config.layer_start,
            config.layer_end,
            config.test_prompt
        )
        logger.info(f"GPU hidden states shape: {gpu_hidden_states.shape}")

        if args.skip_xpu:
            logger.info("Skipping XPU (--skip-xpu)")
            logger.info("GPU-only test complete")
            return 0

        # Create XPU backend
        logger.info("Creating XPU backend")
        xpu_backend_config = BackendConfig(
            host=config.xpu_host,
            user=config.xpu_user,
            docker=config.xpu_docker,
            vllm_path=config.xpu_vllm_path,
            cards=config.xpu_cards,
            device_type="xpu"
        )
        xpu_backend = create_backend(
            config.backend,
            xpu_backend_config,
            config.model_path,
            config.shared_fs
        )

        try:
            # Setup XPU backend
            logger.info("Setting up XPU backend (applying patches)")
            xpu_backend.setup()

            # Run layer extraction on XPU
            logger.info(f"Running layers [{config.layer_start}, {config.layer_end}) on XPU")
            xpu_hidden_states = xpu_backend.run_layer_range(
                config.layer_start,
                config.layer_end,
                config.test_prompt
            )
            logger.info(f"XPU hidden states shape: {xpu_hidden_states.shape}")

            # Compare
            logger.info("Comparing GPU vs XPU hidden states")
            result = compare_tensors(gpu_hidden_states, xpu_hidden_states)

            logger.info(f"Match: {result.match}")
            logger.info(f"Cosine similarity: {result.cosine_similarity:.6f}")
            logger.info(f"Max relative error: {result.max_rel_error:.6e}")
            logger.info(f"Max absolute error: {result.max_abs_error:.6e}")

            if result.match:
                logger.info("✓ GPU and XPU outputs match!")
            else:
                logger.warning("✗ GPU and XPU outputs diverge!")

        finally:
            # Cleanup XPU
            logger.info("Cleaning up XPU backend")
            xpu_backend.cleanup()

    finally:
        # Cleanup GPU
        logger.info("Cleaning up GPU backend")
        gpu_backend.cleanup()

    logger.info("GPU vs XPU comparison test complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

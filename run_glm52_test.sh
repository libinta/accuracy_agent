#!/bin/bash
# Wrapper script to run GLM-5.2-FP8 accuracy test with proper GPU card selection

set -e

echo "=== GLM-5.2-FP8 GPU/XPU Accuracy Test ==="
echo "GPU: gpu-host.example.com / your_gpu_container / cards 6,7"
echo "XPU: xpu-host.example.com / your_xpu_container / cards 0-7"
echo ""

# Check if in correct directory
if [ ! -f "examples/glm52_fp8_config.yaml" ]; then
    echo "ERROR: Must run from accuracy_agent directory"
    exit 1
fi

# Set GPU cards for the test
export CUDA_VISIBLE_DEVICES=6,7
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7

echo "Running accuracy-debug..."
echo "Output will be saved to: /mnt/weka/accuracy_debug_glm52"
echo ""

# Run the accuracy debugger
python -m accuracy_agent.cli --config examples/glm52_fp8_config.yaml

echo ""
echo "=== Test Complete ==="
echo "Check results in: /mnt/weka/accuracy_debug_glm52"

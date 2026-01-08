#!/bin/bash
# Training script for H-OBS/R-SPAR experiments

set -e

# Parse arguments
MODEL="resnet50"
METHOD="hobs-rspar"
GPU=0
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "H-OBS/R-SPAR Training Script"
echo "=========================================="
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "GPU: $GPU"
echo "Seed: $SEED"
echo "=========================================="

# Run training
python experiments/main.py \
    --config configs/${MODEL}.yaml \
    --method $METHOD \
    --gpu $GPU \
    --seed $SEED

echo ""
echo "Training completed!"

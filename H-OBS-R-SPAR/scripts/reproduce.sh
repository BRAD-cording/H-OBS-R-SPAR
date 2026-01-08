#!/bin/bash
# Reproducibility Script - Run experiments with all 5 seeds

set -e

MODEL="${1:-resnet50}"
METHOD="${2:-hobs-rspar}"
GPU="${3:-0}"

SEEDS=(42 2024 10086 520 1314)

echo "=========================================="
echo "H-OBS/R-SPAR Reproducibility Validation"
echo "=========================================="
echo "Model: $MODEL"
echo "Method: $METHOD"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="

RESULTS_DIR="./results/${MODEL}_reproduce_$(date +%Y%m%d_%H%M%S)"
mkdir -p $RESULTS_DIR

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Running with seed: $SEED"
    
    python experiments/main.py \
        --config configs/${MODEL}.yaml \
        --method $METHOD \
        --gpu $GPU \
        --seed $SEED \
        2>&1 | tee "${RESULTS_DIR}/run_seed_${SEED}.log"
done

echo ""
echo "=========================================="
echo "All runs completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="

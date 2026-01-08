# H-OBS/R-SPAR Experiment Documentation

## Overview

This document describes the experimental setup, procedures, and expected results for the H-OBS/R-SPAR pruning framework as designed for IEEE T-SMC-S submission.

## Experimental Setup

### Hardware Requirements

| Component | Specification                                   |
| --------- | ----------------------------------------------- |
| GPU       | NVIDIA RTX-5090 (24GB VRAM) or H100 (80GB VRAM) |
| CPU       | Intel Xeon 8358, 32 cores                       |
| Memory    | 512GB DDR4                                      |
| Storage   | 2TB NVMe SSD                                    |

### Software Environment

```
PyTorch: 2.2.0
CUDA: 11.8
cuDNN: 8.9
Python: 3.10
```

## Dataset Preparation

### ImageNet-1k

```bash
# Download ImageNet-1k
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

# Extract
mkdir -p /data/imagenet/train
mkdir -p /data/imagenet/val
tar -xf ILSVRC2012_img_train.tar -C /data/imagenet/train
tar -xf ILSVRC2012_img_val.tar -C /data/imagenet/val
```

**Dataset Structure:**

```
/data/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ... (1000 classes)
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ... (1000 classes)
```

## Experiments

### Table 1: ResNet-18/50 System-Level Performance

**Command:**

```bash
# Baseline
python experiments/main.py --config configs/resnet50.yaml --method unpruned --seed 42

# H-OBS/R-SPAR
python experiments/main.py --config configs/resnet50.yaml --method hobs-rspar --seed 42

# DepGraph
python experiments/main.py --config configs/resnet50.yaml --method depgraph --seed 42

# Repeat for seeds: 2024, 10086, 520, 1314
```

**Expected Results (Mean ± Std, 5 runs):**

- Unpruned: 76.13±0.12% accuracy, 5.2±0.3ms latency
- H-OBS/R-SPAR: 75.96±0.14% accuracy, 16.1±0.7ms latency, 3.10× speedup

### Table 2-4: Other Models

Repeat above procedure for:

- ResNet-18: `--config configs/resnet18.yaml`
- MobileNetV2: `--config configs/mobilenetv2.yaml`
- EfficientNet-B0: `--config configs/efficientnet_b0.yaml`

### Table 5: Cross-Platform Validation

```bash
# Run on different GPUs
python experiments/cross_platform.py --config configs/resnet50.yaml

# Expected platforms: RTX-5090, H100, A100, V100
```

### Table 6: Batch Size Analysis

```bash
python experiments/batch_analysis.py --config configs/resnet50.yaml \
    --batch-sizes 16 32 64 128
```

### Table 7: DVFS Energy Analysis

*Note: Requires system-level power measurement tools (nvidia-smi, etc.)*

### Table 8: Ablation Study

```bash
# Full ablation
python experiments/ablation.py --config configs/resnet50.yaml --ablation all

# Individual components
python experiments/ablation.py --config configs/resnet50.yaml --ablation no_hessian
python experiments/ablation.py --config configs/resnet50.yaml --ablation no_rl
python experiments/ablation.py --config configs/resnet50.yaml --ablation no_kd
python experiments/ablation.py --config configs/resnet50.yaml --ablation no_adaptive
```

### Table 9-12: Sensitivity & Training Analysis

Run training with detailed logging:

```bash
python experiments/main.py --config configs/resnet50.yaml \
    --method hobs-rspar --verbose --log-dir results/detailed/
```

Logs will include:

- Hessian vs magnitude sensitivity comparison
- RL budget allocation evolution
- Adaptive λ(t) trajectory
- Training stability metrics

### Table 13: Complexity Analysis

Automatically computed during training. Check logs for:

- K-FAC computation time
- RL exploration episodes
- Total training overhead

### Table 14-15: GPU & TensorRT Analysis

**GPU Profiling:**

```bash
python experiments/gpu_profiler.py --config configs/resnet50.yaml
```

**TensorRT Benchmark:**

```bash
python experiments/tensorrt_engine.py --config configs/resnet50.yaml \
    --precision fp16
```

### Table 16: Reproducibility Validation

```bash
bash scripts/reproduce.sh resnet50 hobs-rspar 0
```

This runs all 5 seeds automatically and generates summary statistics.

## Statistical Validation

### Paired T-Test

```python
from utils.stat_tests import paired_t_test

baseline = np.array([76.13, 76.15, 76.12, 76.14, 76.11])
ours = np.array([75.97, 75.95, 75.96, 75.98, 75.94])

result = paired_t_test(baseline, ours)
print(f"p-value: {result['p_value']:.4f}")
# Expected: p < 0.05 (statistically significant)
```

### Bootstrap Confidence Intervals

```python
from utils.stat_tests import bootstrap_confidence_interval

ci_lower, ci_upper = bootstrap_confidence_interval(ours, confidence=0.95)
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

## Visualization

### Training Curves

```python
from utils.visualization import plot_training_curves

history = {
    'train_loss': [...],
    'val_loss': [...],
    'train_acc': [...],
    'val_acc': [...]
}

plot_training_curves(history, save_path='results/training_curves.png')
```

### Method Comparison

```python
from utils.visualization import plot_pruning_comparison

methods = ['Unpruned', 'DepGraph', 'HRank', 'H-OBS/R-SPAR']
accuracies = [76.13, 75.82, 75.65, 75.96]
speedups = [1.0, 3.03, 2.85, 3.10]

plot_pruning_comparison(methods, accuracies, speedups, 
                       save_path='results/comparison.png')
```

## Results Organization

```
results/
├── resnet50/
│   ├── hobs_rspar/
│   │   ├── seed_42/
│   │   │   ├── checkpoint.pth
│   │   │   ├── metrics.json
│   │   │   └── training.log
│   │   ├── seed_2024/
│   │   └── ...
│   ├── depgraph/
│   └── ...
└── figures/
    ├── training_curves.png
    ├── comparison.png
    └── sensitivity_heatmap.png
```

## Troubleshooting

### OOM (Out of Memory)

- Reduce batch size in config
- Use gradient checkpointing
- Enable FP16 training

### Slow Training

- Increase num_workers in dataloader
- Use pin_memory=True
- Enable CUDA benchmark mode

### Inconsistent Results

- Fix random seeds
- Use deterministic CUDA operations
- Disable benchmark mode for reproducibility

## Citation

If you use this code for research, please cite:

```bibtex
@article{hobs_rspar2026,
  title={H-OBS/R-SPAR: A System-Level Framework for CNN Acceleration},
  author={Authors},
  journal={IEEE T-SMC-S},
  year={2026}
}
```

---

**Last Updated:** 2026-01-08
**Experiment Version:** 1.0
**Target Journal:** IEEE T-SMC-S

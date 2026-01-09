# H-OBS/R-SPAR API Documentation

## Core Modules

### methods.hobs

#### `HOBSSensitivityAnalyzer`

Hessian-guided On-demand Budget Sensitivity Analyzer.

**Parameters:**
- `model` (nn.Module): PyTorch model to analyze
- `damping` (float): Damping factor for Hessian approximation (default: 1e-3)
- `use_kfac` (bool): Whether to use K-FAC approximation (default: True)

**Methods:**

##### `compute_sensitivity(dataloader, criterion=None, num_batches=10)`
Compute Hessian-based sensitivity scores for all filters.

**Returns:** Dictionary mapping layer names to sensitivity scores

**Example:**
```python
from methods.hobs import HOBSSensitivityAnalyzer

hobs = HOBSSensitivityAnalyzer(model, use_kfac=True)
sensitivity_scores = hobs.compute_sensitivity(train_loader, num_batches=50)
```

##### `get_pruning_plan(pruning_ratio=0.5)`
Generate pruning plan based on sensitivity scores.

**Returns:** Dictionary mapping layer names to filter indices to prune

---

### methods.rspar

#### `RSPARAgent`

Reinforcement Learning-driven Structured Pruning agent.

**Parameters:**
- `num_layers` (int): Number of layers to allocate budgets for
- `layer_feature_dim` (int): Dimension of layer features (default: 3)
- `hidden_dim` (int): Hidden dimension for networks (default: 256)
- `lr` (float): Learning rate (default: 3e-4)
- `gamma` (float): Discount factor (default: 0.99)

**Methods:**

##### `select_action(state, deterministic=False)`
Select pruning action given current state.

**Returns:** Tuple of (action, log_prob)

##### `update_policy(batch_size=64, epochs=10)`
Update policy using PPO algorithm.

---

### utils.kfac

#### `KFACHessianApproximator`

K-FAC based Hessian approximation for neural networks.

**Parameters:**

- `model` (nn.Module): PyTorch model
- `damping` (float): Damping factor (default: 1e-3)
- `update_freq` (int): Update frequency (default: 10)

**Methods:**

##### `register_hooks()`

Register forward and backward hooks to capture statistics.

##### `update_factors()`

Update K-FAC factors using cached activations and gradients.

##### `compute_fisher_diagonal(layer_name)`

Compute diagonal of Fisher Information Matrix for a layer.

**Returns:** Diagonal of Fisher matrix

---

### utils.pruner

#### `PhysicalPruner`

Handles physical structural pruning using `torch-pruning` library.

**Parameters:**

- `model` (nn.Module): PyTorch model to prune
- `example_input` (torch.Tensor): Dummy input for dependency graph construction (default: random [1, 3, 224, 224])

**Methods:**

##### `prune(pruning_plan)`

Execute physical pruning based on the provided plan.

**Args:**

- `pruning_plan` (Dict[str, List[int]]): Dictionary mapping layer names to list of filter indices to prune.

**Returns:** Physically pruned model

---

### utils.metrics

#### `SystemMetrics`

Comprehensive system-level metrics measurement.

**Static Methods:**

##### `measure_latency(model, input_shape=(1,3,224,224), num_iterations=100)`

Measure inference latency.

**Returns:** Tuple of (mean_latency_ms, std_latency_ms)

##### `measure_throughput(model, batch_size=64, input_shape=(3,224,224), num_iterations=50)`

Measure throughput (images/second).

**Returns:** Throughput value

##### `count_parameters(model)`

Count total parameters.

**Returns:** Parameter count

##### `count_flops(model, input_shape=(1,3,224,224))`

Estimate FLOPs.

**Returns:** FLOPs count

##### `measure_all(model, batch_size=64, input_shape=(3,224,224))`

Measure all metrics.

**Returns:** Dictionary with all metrics

**Example:**

```python
from utils.metrics import SystemMetrics

metrics = SystemMetrics.measure_all(model, batch_size=64)
print(f"Latency: {metrics['latency_ms']:.2f} ms")
print(f"Throughput: {metrics['throughput_imgs_per_sec']:.1f} images/s")
```

---

## Baseline Methods

### methods.baselines

All baseline methods follow the same interface:

```python
def method_prune(model, dataloader, pruning_ratio=0.5, **kwargs):
    """
    Args:
        model: PyTorch model to prune
        dataloader: DataLoader for importance estimation
        pruning_ratio: Global pruning ratio (0-1)
    
    Returns:
        Pruned model
    """
```

**Available Methods:**

- `unpruned` - Unpruned Model Baseline (No Pruning)
- `dep_graph_prune` - DepGraph (CVPR 2023)
- `autocompress_prune` - AutoCompress (AAAI 2020)
- `hrank_prune` - HRank (CVPR 2020)
- `taylor_prune` - Taylor Pruning (CVPR 2019)
- `adaprune_prune` - AdaPrune (T-SusC 2021)
- `gnn_prune` - GNN-Pruning (arXiv 2022)

**Example:**

```python
from methods.baselines import hrank_prune

pruned_model = hrank_prune(
    model,
    train_loader,
    pruning_ratio=0.3,
    num_samples=100
)
```

---

## Utilities

### utils.stat_tests

Statistical validation functions.

#### `paired_t_test(baseline, method, alpha=0.05)`

Perform paired t-test.

**Returns:** Dictionary with t-statistic, p-value, and significance

#### `bootstrap_confidence_interval(data, confidence=0.95, n_bootstrap=10000)`

Compute bootstrap confidence interval.

**Returns:** Tuple of (lower_bound, upper_bound)

#### `run_statistical_analysis(results, baseline_name='Unpruned')`

Run complete statistical analysis.

**Returns:** Dictionary with all statistical results

---

### utils.gpu_profiler

GPU profiling utilities.

#### `GPUProfiler(enabled=True)`

GPU profiler for model performance analysis.

**Methods:**

- `start(name)` - Start profiling section
- `end(name)` - End profiling section
- `get_summary()` - Get profiling summary
- `print_summary()` - Print profiling summary

---

### utils.visualization

Visualization tools for training and analysis.

#### `plot_training_curves(history, save_path=None)`

Plot training and validation curves.

#### `plot_pruning_comparison(methods, accuracies, speedups, save_path=None)`

Plot comparison of pruning methods.

#### `plot_sensitivity_heatmap(sensitivity_scores, layer_names, save_path=None)`

Plot heatmap of filter sensitivity scores.

#### `plot_lambda_evolution(lambda_history, acc_history, save_path=None)`

Plot adaptive λ(t) evolution over training.

---

## Configuration

### YAML Configuration Format

```yaml
model:
  name: "resnet50"
  pretrained: true
  num_classes: 1000

dataset:
  name: "imagenet"
  data_dir: "/data/imagenet"
  image_size: 224

pruning:
  method: "hobs-rspar"
  target_flops_reduction: 0.76
  
  hobs:
    use_kfac: true
    damping: 0.001
    num_batches_sensitivity: 50
  
  rspar:
    num_episodes: 100
    hidden_dim: 256
    learning_rate: 0.0003

experiment:
  name: "resnet50_hobs_rspar"
  save_dir: "./results/resnet50"
  random_seeds: [42, 2024, 10086, 520, 1314]
```

---

## Command Line Interface

### Main Training Script

```bash
python experiments/main.py --config configs/resnet50.yaml --gpu 0
```

**Arguments:**
- `--config` - Path to configuration file
- `--gpu` - GPU device ID
- `--seed` - Random seed
- `--method` - Pruning method override

### Ablation Study

```bash
python experiments/ablation.py --config configs/resnet50.yaml --ablation all
```

**Arguments:**
- `--config` - Path to configuration file
- `--ablation` - Ablation type (all, no_hessian, no_rl, etc.)

### System Evaluation

```bash
python experiments/system_eval.py --config configs/resnet50.yaml --checkpoint model.pth
```

---

For more information, see the [README](../README.md) and experiment design document.

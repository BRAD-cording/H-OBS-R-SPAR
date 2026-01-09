# H-OBS/R-SPAR Experiment Deployment Schedule

**Total Duration:** 44 Working Days
**Objective:** Complete all experiments defined in `DELIVERABLES.md`, ensuring reproducibility and meeting performance targets for IEEE T-SMC-S submission.

## Schedule Overview

| Phase                            | Duration   | Key Objectives                                             | Verification Point                                           |
| :------------------------------- | :--------- | :--------------------------------------------------------- | :----------------------------------------------------------- |
| **1. Prep & Reproduction** | Days 1-15  | Environment setup, baseline reproduction, code readiness.  | **VP1:** All 7 methods run successfully.               |
| **2. Core Experiments**    | Days 16-35 | Full training on ImageNet-1k, hyperparameter tuning.       | **VP2:** All metrics meet `DELIVERABLES.md` targets. |
| **3. Analysis & Polish**   | Days 36-44 | Ablations, cross-platform tests, documentation, packaging. | Final Submission Package ready.                              |

---

## Detailed Daily Schedule

### Phase 1: Preparation & Baseline Reproduction (Days 1-15)

**Goal:** Ensure all code is executable and baselines are reproducible.

| Days            | Task Category              | Specific Actions                                                                                                                                   | Deliverables                                                                             |
| :-------------- | :------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------- |
| **1-3**   | **Environment**      | - Setup Docker container `hobs-rspar:latest<br>`- Prepare ImageNet-1k (download, verify md5)`<br>`- Configure GPU drivers & CUDA 12.x          | - Docker Image`<br>`- Verified Dataset                                                 |
| **4-8**   | **Baselines**        | - Implement/Verify `DepGraph`, `JTP`, `Bi-Level`, `StructAlign`, `UDFC<br>`- Run "dry-run" tests (2 epochs) for each to verify pipeline. | - Baseline Training Logs (Dry Run)`<br>`- Baseline Checkpoints (Init)                  |
| **9-12**  | **Core Method**      | - Finalize `H-OBS` sensitivity analyzer.`<br>`- Finalize `R-SPAR` RL agent logic.`<br>`- Implement `Block-ELL` custom kernels.           | -`models/hobs.py<br>`- `models/rspar.py<br>`- `utils/cuda_optimizers.py`           |
| **13-15** | **VP1: Code Freeze** | -**Verification Point 1:** Run full pipeline for all 7 methods on 10% data subset.`<br>`- Fix any runtime errors or OOM issues.            | -**VP1 Report:** All methods runnable.`<br>`- `scripts/reproduce.sh` verified. |

### Phase 2: Core Experimentation & Optimization (Days 16-35)

**Goal:** Achieve target metrics on ResNet-50, MobileNetV2, and EfficientNet-B0.

| Days            | Task Category                   | Specific Actions                                                                                                                                                                             | Deliverables                                                                      |
| :-------------- | :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **16-22** | **ResNet-50 (Baselines)** | - Full training of 5 baselines on ResNet-50 (ImageNet-1k).`<br>`- Parallel execution on available GPUs.                                                                                    | - Table 1 Baseline Rows`<br>`- Baseline Logs/Models                             |
| **23-27** | **ResNet-50 (Ours)**      | - Full training of**H-OBS/R-SPAR** on ResNet-50.`<br>`- Hyperparameter fine-tuning (if needed) to hit **75.96% Acc / 3.10x Speedup**.                                          | - Table 1 "Ours" Row`<br>`- Best Model Checkpoint                               |
| **28-30** | **Lightweight Models**    | - Run MobileNetV2 & EfficientNet-B0 experiments (Ours vs Best Baseline).`<br>`- Verify latency targets on mobile-like settings (if simulated).                                             | - Table 2 & 3 Data`<br>`- Lightweight Model Checkpoints                         |
| **31-33** | **Optimization**          | - Integrate TensorRT (FP16).`<br>`- Profile and optimize `Block-ELL` kernels.`<br>`- Measure Latency/Throughput/Energy on primary GPU.                                                 | - Table 13 & 14 Data`<br>`- Optimized Kernels                                   |
| **34-35** | **VP2: Metric Audit**     | -**Verification Point 2:** Compare all collected results against `DELIVERABLES.md` targets.`<br>`- **CRITICAL:** If targets missed, trigger emergency tuning (weekend work). | -**VP2 Report:** Pass/Fail on Targets.`<br>`- Updated `DELIVERABLES.md` |

### Phase 3: Analysis, Cross-Platform & Final Polish (Days 36-44)

**Goal:** Deep analysis, ablation studies, and submission packaging.

| Days            | Task Category            | Specific Actions                                                                                                                                    | Deliverables                                              |
| :-------------- | :----------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- |
| **36-38** | **Ablation**       | - Run Module Ablation (w/o Hessian, w/o RL, etc.).`<br>`- Run Sensitivity Analysis (Table 8).`<br>`- Run Stability Analysis (Table 11).         | - Tables 7-11`<br>`- Figs 2, 3, 4                       |
| **39-40** | **Cross-Platform** | - Deploy models on RTX-5090, H100, A100, V100.`<br>`- Measure real-world latency and speedup.                                                     | - Table 4 Data`<br>`- Cross-platform logs               |
| **41-42** | **Visualization**  | - Generate Feature Map Reconstruction Error (Fig 1).`<br>`- Plot all charts (Matplotlib/Seaborn).`<br>`- Finalize all tables in LaTeX/Markdown. | -`docs/figures/*.png<br>`- Final Tables                 |
| **43**    | **Documentation**  | - Complete `README.md`, `API.md`, `EXPERIMENTS.md`.`<br>`- Clean code (remove comments, format).                                            | - Completed `docs/` folder`<br>`- Cleaned Source Code |
| **44**    | **Submission**     | - Build Docker image.`<br>`- Package Anonymous Code.`<br>`- Compile Supplementary Material PDF.                                                 | -**Final Submission Package**                       |

---

## Resource Requirements

- **Compute:** Minimum 4x NVIDIA A100 (80GB)  nodes for parallel baseline training.
- **Storage:** 2TB+ SSD for ImageNet dataset, checkpoints, and logs.
- **Software:** PyTorch 2.x, CUDA 12.1+, TensorRT 8.6+.

## Risk Management

- **Risk:** Baseline reproduction fails to match reported paper results.
  - *Mitigation:* Use official repos where possible; allow 1-2% margin; document discrepancies.
- **Risk:** H-OBS/R-SPAR fails to meet 3.10x speedup target.
  - *Mitigation:* Aggressively prune later layers; optimize `Block-ELL` kernel earlier (Day 20).
- **Risk:** Training instability (Loss spikes).
  - *Mitigation:* Enable "Stability Score" monitoring early; adjust LR warmup.

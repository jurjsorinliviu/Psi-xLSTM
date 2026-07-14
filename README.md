# Physics Structure-Informed Neural Networks for Constraint-Preserving TinyML and Sustainable Edge Deployment

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI)

> **Author**: Sorin Liviu Jurj  
> **Status**: Submitted

<img width="1854" height="3250" alt="figure1_methodology_pipeline" src="https://github.com/user-attachments/assets/3264c9eb-7806-4ed2-8df4-07ed24e85445" />


## 🧪 Deployment and Robustness Experiment Suite (`revision/`)

Beyond the seven-problem intermittent-training study, the repository contains a second
experiment suite under [`revision/`](revision/): strong compression baselines (`exp1`),
ε sensitivity (`exp2`), stochastic-interruption and lossy-checkpoint study (`exp3`),
clustering with centroid retraining (`exp4`, `distill_cluster.py`), **measured
Cortex-M4/M7 deployment** (`exp5_mcu/`: C exporter, firmware, linker scripts, QEMU
harness), scaling frontier (`exp6`), simulator-size sweep (`exp7`), robust estimator
(`exp8`), solar validation (`exp9`), and **four additional physics benchmarks** taking
the suite from 7 to 11 with a permutation test of the predictor question (`exp10`,
`pdes_extra.py`).

Key facts these experiments establish:

- **Memory must count the relation matrix.** A count of cluster centroids is not a
  measurement of memory: the relation matrix **R** costs 1 index byte per parameter,
  bounding the weight-memory compression of a clustered fp32 network near **4×**.
  The 13-cluster figure is a count of *distinct weight values*.
- **Clustering needs its centroids retrained.** Cluster-and-stop degrades the model at
  every ε (memristor at ε=0.1: test MSE 2.6e-2 vs 1.3e-6 unclustered); re-optimizing
  the K centroids with **R** frozen recovers, and can exceed, the unclustered accuracy.
- **Budget sensitivity spans 0.97×–23.7×.** On a scale-invariant paired log-ratio over
  the 11-problem suite, a halved training budget multiplies the solution error by
  between 0.97× and 23.7×, and no descriptor of the equation predicts the cost.
- **The structured Verilog-A export carries no simulator-side penalty** at any circuit
  size tested (1–256 devices); single-device wall-clock margins decay with circuit size
  and vanish by ~64 devices.

**Headline finding:** the three *elliptic* problems, identical under every descriptor
(elliptic, no time dependence, linear, 2nd order), span **1.33× (Laplace), 3.82×
(Helmholtz), 23.70× (Poisson)** in budget sensitivity. Poisson, the smoothest problem with
no dynamics at all, is the **most** budget-sensitive of all eleven. Budget sensitivity
cannot be read off the equation; it must be measured.

The harness that produces the paper's main tables (`control_arm.py`, `full_sweep.py`,
`experiments/reproducibility.py`) is included, so the main tables are reproducible from
a clean clone.

## 📋 Overview

This repository contains the complete implementation and experimental validation for a framework that couples **physics structure-informed learning** with **renewable-energy constraints** to address three challenges in Edge AI deployment:

1. **Hardware over-specification** — reads quantitative deployment requirements (operations, memory, power) a priori from the structured Psi-NN architecture to inform platform selection before any hardware is committed (a specification methodology; for the small models demonstrated here the platform choice is driven by task size rather than by the structure).
2. **Carbon footprint** — right-sizing the platform (Jetson Orin Nano → Nordic nRF52840) yields ~45× lifecycle carbon reduction per device; for the demonstrated small models this is driven by task size rather than by the structure, and solar-constrained training contributes under 1% of it.
3. **Renewable-power feasibility** — characterizes what intermittent training actually does via an orthogonal five-cell decomposition (n=10 seeds, paired bootstrap, 95% CI) across **eleven** physics benchmarks, isolating regularization, budget, and schedule contributions. The interruption effect itself is *measured* (stochastic timing, work rollback, degraded checkpoints) rather than assumed to be zero (`revision/exp3`).

### Key Results

| Metric                   | Value                                                        |
| ------------------------ | ------------------------------------------------------------ |
| **Cost Savings**         | 98.0% ($249 Jetson Orin Nano → $5 Nordic nRF52840)           |
| **Carbon Reduction**     | ~45× per device (238 kg → ~5.35 kg CO₂ over 5-year lifecycle) |
| **Sustainability lever** | ~99.9% from hardware right-sizing; <1% (~0.32 kg) from solar-constrained training |
| **Accuracy**             | No penalty for Burgers PDE (test MSE improves under solar training, Table V) |
| **Budget sensitivity**   | A halved budget multiplies solution error by 0.97×–23.7× across **eleven** benchmarks (9 of 11 resolved). **No descriptor of the equation predicts it** (permutation test, all p > 0.18) |

## 🔁 Reproduce Everything (One Command)

```bash
python reproduce_paper.py
```

This single entry point regenerates **every table, figure, and headline number**
in the paper into one self-contained tree:

```
paper_artifacts/
├── tables/    Table I, III, IV, V, VI, the κ-sweep, the binding-compute example,
│              and the carbon breakdown  (each as .csv AND .tex)
├── figures/   methodology pipeline, five-cell decomposition, κ-sweep curve
└── data/      all_derived_numbers.json + MANIFEST.md (artifact → paper element)
```

- **Default (archived data, ~seconds, deterministic):** every statistic is
  *recomputed* from the blessed per-seed arrays in `results/` using the documented
  paired bootstrap (mean of per-seed ratios, 10,000 resamples, 95% CI, seed=42) —
  nothing is copied from the manuscript. The carbon headline (238 → 5.35 kg, ~45×)
  is computed from the Table I inputs and Eqs. 56–61, so it is auditable rather than
  asserted.
- **`--retrain`:** re-runs the canonical Burgers three-regime + κ-sweep end-to-end
  (CPU backend, to match the blessed runs) and rebuilds from the fresh JSONs. The
  full seven-problem control-arm sweep that produced the archived JSONs is heavy
  (~25–80 h CPU); the remaining problems are reproduced from their archived per-seed
  data (see `paper_artifacts/data/MANIFEST.md`).

```bash
python reproduce_paper.py --retrain        # regenerate Burgers from scratch
python reproduce_paper.py --outdir my_dir  # choose output directory
```

> The per-problem scripts under `experiments/` remain available for inspection, but
> `reproduce_paper.py` is the recommended, clean-clone-safe route to the paper's
> artifacts.

## 🎯 Key Contributions

### 1. Hardware Requirement Extraction from the Structured Architecture

Reads quantitative, a priori hardware requirements from the structured Psi-NN — the memory footprint from its parameter-tying, symmetry, and sparsity, and throughput and power from its architecture shape:

- **TOPS Requirements** — minimum computational throughput
- **Memory Footprint** — RAM/ROM requirements
- **Power Budget** — average and peak consumption
- **Platform Recommendation** — TinyML, Mid-Range, or High-Performance tiers

> _Methodology note: the extraction is general, but for the small models demonstrated here the dense network already fits the target, so the platform choice is task-size-driven; the structure-specific discriminating value (clustered memory) is prospective, pending the distillation-based clustering of the prior Psi-NN method (not used here)._

### 2. Orthogonal Five-Cell Decomposition of Intermittent Training

A paired decomposition that isolates the contributions of regularization (D→C), budget (C→B), and the interruption schedule (B→E) to training outcomes under renewable intermittency. Under the deterministic 50% duty cycle and lossless Adam-state checkpointing assumed here, the active regime E reduces *by structural identity* to continuous training at the halved budget B — so the B→E contrast is zero by construction rather than an empirical finding about schedules in general. The two load-bearing empirical contrasts are therefore **regularization (D→C)** and **budget (C→B)**, both measured per problem under D-normalization (additive closure verified to residual ≤ 4×10⁻¹⁴). Schedule effects beyond this structural identity (stochastic interruption timing, lossy checkpoints) are measured separately in `revision/exp3`.

### 3. Budget Sensitivity as the Principal Deployment Risk

The dominant factor in training outcome under solar constraints is **budget sensitivity** — the C→B contrast at matched regularization. Measured with a scale-invariant paired log-ratio, a halved budget multiplies the solution error by between 0.97× (Burgers) and 23.7× (Poisson) across the eleven-problem suite (9 of 11 resolved). **No descriptor of the equation predicts this cost** (permutation test over PDE class, time dependence, nonlinearity, and derivative order: all p > 0.18). Practitioners must therefore measure C→B per problem before committing to renewable-powered training.

### 4. Adaptive Regularization (κ-Mechanism)

A power-responsive regularization amplification scheme: when the active period begins after an interruption, regularization weight is multiplied by `(1 + κ·exp(-t/τ))`. The κ-sweep on Burgers PDE (κ ∈ {0, 0.5, 1.0, 1.5, 2.0}, n=10 seeds) shows a **weak-monotone improvement curve** with point estimates from **−8.5% [−10.3, −6.6] at κ=0 to −11.2% [−13.5, −9.0] at κ=2**, each individually resolved at 95% CI. The endpoint span (−2.7% [−3.3, −2.2]) is also resolved and cross-validates at 0.00 pp against an independent passive-to-active comparison.

### 5. Solar Model Validation (Markov vs. Real PVGIS Data)

The Markov solar model has been validated against location-calibrated PVGIS data for Chemnitz, Germany (50.8°N, n=3 seeds, training loss as model-fidelity metric):

| Solar Panel Area (m²) | Real Duty Cycle | Markov Duty Cycle | Real Degradation | Markov Degradation | Δ (Degradation) |
| --------------------- | --------------- | ----------------- | ---------------- | ------------------ | --------------- |
| 2 (undersized)        | 0.3%            | 12.9%             | +2035%           | +109%              | Diverges        |
| 10                    | 21.7%           | 36.3%             | +89%             | +60%               | 29 pp           |
| 15 (target)           | 27.4%           | 39.5%             | +68%             | +56%               | **11.3 pp**     |

**Key Finding**: the Markov model reproduces training-degradation impact to within **11.3 pp** when panels are sized to approximately 50% duty cycle. Agreement is on degradation impact, not on duty-cycle fidelity — the model overestimates the realized duty cycle (Markov: 39.5% vs. real: 27.4%). This is a model-fidelity comparison (training loss) rather than a performance claim; n=3 seeds, indicative not definitive.

To run the validation:

```bash
# Default (2 m² panel — undersized, will fail)
python experiments/pvgis_solar_validation.py --epochs 3000 --seeds 3

# Properly sized for Northern Europe (15 m² panel)
python experiments/pvgis_solar_validation.py --epochs 3000 --seeds 3 \
  --panel-area 15.0 --peak-power 1500.0 \
  --output results/pvgis_validation_15m2
```

### 6. GPU Power Validation

The manuscript uses 250 W as a conservative estimate for RTX 4090 power consumption during training. Empirical measurement shows actual consumption is significantly lower:

| Configuration                                      | Mean Power | Max Power | Min Power | Manuscript Claim | Discrepancy   |
| -------------------------------------------------- | ---------- | --------- | --------- | ---------------- | ------------- |
| 4×50 neurons, 1000 collocation points, 6000 epochs | **57 W**   | 92 W      | 50 W      | 250 W            | **77% lower** |

**Implication**: PINN training is a lightweight GPU workload. The 250 W assumption is conservative, meaning solar feasibility is more readily achievable in practice than the presented analysis suggests.

```bash
python experiments/measure_gpu_power.py            # quick (500 epochs)
python experiments/measure_gpu_power.py --manuscript   # full (6000 epochs)
# Results saved to results/gpu_power_measurement/
```

### 7. Statistical Validation

- **60+ experiments** across 7 PDE problems (Burgers, Laplace, Heat, Wave, Advection, Allen-Cahn, Memristor)
- **10 independent random seeds** per configuration
- **Paired bootstrap** estimator (mean of per-seed ratios, 10,000 resamples, 95% CI)
- All contrasts D-normalized for Table VI (additive closure residual ≤ 4×10⁻¹⁴); C-normalized for Table IV (budget sensitivity)
- Canonical metric: **test MSE vs. analytical solution** (not training loss)

## 🏗️ Repository Structure

```
├── requirements.txt                # Python dependencies
├── reproduce_paper.py              # ⭐ One-command reproduction (see above)
├── sustainable_edge_ai.py          # Main implementation
├── generate_figure2_decomposition.py  # Figure 2 (five-cell decomposition)
├── generate_figure3_kappa_sweep.py # Figure 3 (Burgers κ-sweep curve)
│
├── experiments/                          # Individual problem experiments
│   ├── three_regime_burgers_experiment.py
│   ├── three_regime_laplace_experiment.py
│   ├── three_regime_heat_experiment.py
│   ├── three_regime_wave_experiment.py
│   ├── three_regime_advection_experiment.py
│   ├── three_regime_allen_cahn_experiment.py
│   ├── three_regime_memristor_experiment.py
│   ├── kappa_sweep_experiment.py         # Figure 3 (κ-sweep on Burgers)
│   ├── duty_cycle_sweep.py
│   ├── pvgis_solar_validation.py         # Markov model validation
│   ├── measure_gpu_power.py              # GPU power measurement
│   ├── realistic_solar_validation.py
│   ├── statistical_validation.py
│   ├── heat_wave_debug.py                # Hyperparameter debug utility
│   ├── export_results.py
│   └── methodology_pipeline.html         # Figure 1 source (HTML, rendered to PNG)
│
├── PSI-HDL-implementation/         # Base Ψ-HDL framework
│   ├── Code/
│   │   ├── structure_extractor.py  # Hierarchical clustering
│   │   ├── verilog_generator.py    # HDL code generation
│   │   └── vteam_baseline.py       # Memristor baseline
│   └── Psi-NN-main/                # Original Ψ-NN framework
│       ├── Module/
│       │   ├── PsiNN_burgers.py
│       │   ├── PsiNN_laplace.py
│       │   ├── PsiNN_poisson.py
│       │   └── Training.py
│       └── Config/                 # Experiment configurations
│
└── results/                              # Experimental outputs (CPU backend, blessed)
    ├── consolidated_sweep/                # 10-seed Pass/Cont/Active test-MSE runs (Table V)
    ├── control_arm/                       # D, C, B, A cells for Table VI decomposition
    ├── burgers_kappa_sweep/               # κ ∈ {0, 0.5, 1.0, 1.5, 2.0} for Figure 3
    ├── pvgis_validation/                  # Markov vs. PVGIS validation (2 m² panel)
    ├── pvgis_validation_50pct_duty/       # Markov vs. PVGIS (15 m² panel, target duty cycle)
    ├── pvgis_validation_10m2_panel/       # Markov vs. PVGIS (10 m² panel)
    ├── gpu_power_measurement/             # Measured RTX 4090 power during PINN training
    ├── architecture_sensitivity/          # Architecture-width sensitivity (Burgers deep/wide)
    ├── long_term_convergence/             # 10k-epoch convergence runs (Burgers, Laplace)
    └── statistical_validation/            # Per-PDE statistical validation outputs

paper_artifacts/                           # Generated by reproduce_paper.py (git-ignorable)
├── tables/                                # Table I/III/IV/V/VI, κ-sweep, carbon (.csv + .tex)
├── figures/                               # Pipeline, decomposition, κ-sweep
└── data/                                  # all_derived_numbers.json + MANIFEST.md
```

## 🚀 Quick Start

### Option 1: GitHub Codespaces (Recommended — Zero Setup)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI)

What's included:

- Python 3.11 with all dependencies pre-installed
- Jupyter Notebook support
- VS Code extensions for Python development
- Ready-to-run experiments

```bash
# Verify setup
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Run your first experiment (single PDE, three regimes)
python experiments/three_regime_burgers_experiment.py
```

### Option 2: Local Installation

```bash
# Python 3.8 or higher
python --version

# CUDA-capable GPU optional (CPU backend is the canonical one — see Reproducibility)
nvidia-smi
```

```bash
git clone https://github.com/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI.git
cd Psi-NNs-for-Sustainable-Edge-AI
pip install -r requirements.txt
pip install -r PSI-HDL-implementation/requirements.txt
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Running Experiments

> **To reproduce the paper's tables and figures, use [`reproduce_paper.py`](#-reproduce-everything-one-command).**
> The per-problem scripts below are provided for inspection and for training models
> from scratch. They assume the author's working-tree layout; if an import path fails
> on a fresh clone, prefer `reproduce_paper.py --retrain`, which is clean-clone-safe.

#### 1. Individual Problem Classes

```bash
# Three-regime comparison (Burgers PDE) — primary example
python experiments/three_regime_burgers_experiment.py

# Same protocol for the other six problems:
python experiments/three_regime_laplace_experiment.py
python experiments/three_regime_heat_experiment.py
python experiments/three_regime_wave_experiment.py
python experiments/three_regime_advection_experiment.py
python experiments/three_regime_allen_cahn_experiment.py
python experiments/three_regime_memristor_experiment.py

# Statistical validation with 10 seeds
python experiments/statistical_validation.py

# κ-sweep analysis (κ = 0.0 to 2.0)
python experiments/kappa_sweep_experiment.py

# Realistic weather-dependent solar patterns (PVGIS-calibrated)
python experiments/pvgis_solar_validation.py --panel-area 15.0 --peak-power 1500.0
```

#### 2. Generate Paper Figures

```bash
# Figure 2: Five-cell orthogonal decomposition framework
python generate_figure2_decomposition.py

# Figure 3: Burgers PDE κ-sweep improvement curve
python generate_figure3_kappa_sweep.py
```

Figure 1 (methodology pipeline) is built from the HTML source at
`experiments/methodology_pipeline.html` and rendered to PNG via a headless
browser (e.g. Edge `--headless --screenshot`).

## 📊 Core Modules

### 1. Solar-Constrained Training

```python
from sustainable_edge_ai import SolarConstrainedTrainer

trainer = SolarConstrainedTrainer(model, config={
    'duty_cycle': 0.5,            # 50% solar availability
    'active_period': 10,          # 10 steps on
    'idle_period': 10,            # 10 steps off
    'checkpoint_frequency': 100,  # Save every 100 steps
    'kappa': 2.0,                 # κ-mechanism amplification (0.0 = passive)
})

for epoch in range(num_epochs):
    loss = trainer.train_step(loss_fn=compute_loss, optimizer=optimizer)
    if trainer.should_checkpoint():
        trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
```

### 2. Hardware Specification Extraction

```python
from sustainable_edge_ai import HardwareSpecificationExtractor
from structure_extractor import StructureExtractor

struct_extractor = StructureExtractor(model, model_type="PsiNN_burgers")
hw_extractor = HardwareSpecificationExtractor(model, struct_extractor)

specs = {
    'operations': hw_extractor.compute_operations(),
    'tops':       hw_extractor.compute_tops_requirement(target_fps=30.0),
    'memory_kb':  hw_extractor.compute_memory_requirements() / 1024,
    'power_mw':   hw_extractor.estimate_power_consumption() * 1000,
}
print(f"TOPS Required: {specs['tops']:.6f}")
print(f"Memory:        {specs['memory_kb']:.2f} KB")
print(f"Power:         {specs['power_mw']:.2f} mW")
```

### 3. Platform Recommendation

```python
from sustainable_edge_ai import EdgeAIPlatformRecommender

recommender = EdgeAIPlatformRecommender()
platforms = recommender.recommend_platform(
    requirements=specs,
    constraints={'max_cost_usd': 100, 'max_power_mw': 10000},
)
for i, platform in enumerate(platforms[:3], 1):
    print(f"{i}. {platform['name']}: "
          f"${platform['cost']:.2f}, "
          f"{platform['utilization']*100:.1f}% utilization, "
          f"Fit: {platform['fit_category']}")
```

### 4. Carbon Footprint Analysis

```python
from sustainable_edge_ai import CarbonFootprintAnalyzer

analyzer = CarbonFootprintAnalyzer()
carbon = analyzer.compute_lifecycle_carbon(
    platform=platforms[0],
    deployment_years=5.0,
    training_regime='solar',  # vs 'grid'
    duty_cycle=0.5,
)
print(f"Training Carbon:   {carbon['training_kg_co2']:.3f} kg CO₂")
print(f"Deployment Carbon: {carbon['deployment_kg_co2']:.1f} kg CO₂")
print(f"Total Lifecycle:   {carbon['total_kg_co2']:.1f} kg CO₂")
```

## 🔬 Experimental Results

### Table V — Passive vs. Continuous (Own-Denominator, test MSE)

Test MSE change at the 50% duty cycle (κ=0, passive) relative to continuous
training at full budget. **Paired bootstrap, n=10 seeds, 95% CI, 10,000 resamples.**

| Problem     | PDE Type                     | Pass-Cont test MSE      | Status     |
| ----------- | ---------------------------- | ----------------------- | ---------- |
| **Burgers** | Parabolic (nonlinear)        | **−8.5% [−10.3, −6.6]** | improves   |
| Laplace     | Elliptic (steady-state)      | +7.5% [−7, +20]         | unresolved |
| Allen-Cahn  | Nonlinear reaction-diffusion | +121.2% [+72, +169]     | resolved   |
| Heat        | Parabolic                    | +155.9% [+98, +215]     | resolved   |
| Wave        | Hyperbolic (2nd-order)       | +691.3% [+193, +1462]   | wide CI    |
| Memristor   | ODE (device physics)         | +768.8% [+419, +1149]   | resolved   |
| Advection   | Hyperbolic (1st-order)       | +861.5% [+293, +1615]   | resolved   |

### Table IV — Budget Sensitivity C→B (C-normalized)

The cost of halving the training budget at matched regularization. **5 of 7 resolved at 95% CI.**

| Problem    | C→B Point Estimate | 95% CI        | Status     |
| ---------- | ------------------ | ------------- | ---------- |
| Burgers    | **−3.2%**          | [−4, −2]      | resolved   |
| Laplace    | +41.5%             | [+12, +79]    | resolved   |
| Allen-Cahn | +97.6%             | [+67, +129]   | resolved   |
| Heat       | +176.8%            | [+85, +305]   | resolved   |
| Memristor  | +306.4%            | [−69, +714]   | unresolved |
| Advection  | **+620.0%**        | [+239, +1118] | resolved   |
| Wave       | +3474%             | [+30, +9677]  | wide CI    |

**Key Finding**: on the scale-invariant paired log-ratio over the full eleven-problem
suite, a halved budget multiplies solution error by **0.97×–23.7×** (the table above
reports the C-normalized percentage contrasts for the original seven problems).
**No descriptor of the equation predicts this cost** (permutation test: all p > 0.18) —
parabolic Heat (+176.8%) is exceeded only by transport-dominated Advection (+620.0%)
among the seven, while on the extended suite the three elliptic problems alone span
1.33×–23.7×. The Memristor's sign reversal between C-normalization (+306.4%) and
D-normalization (−767%) indicates the estimator is at its lower n-bound for that
problem. C→B must be measured per problem.

### Figure 3 — Burgers PDE κ-Sweep (Weak-Monotone Improvement)

The κ-mechanism produces a weak-monotone improvement curve relative to the
continuous baseline. **All five points individually resolved at 95% CI.**

| κ Value | test MSE Change vs. Continuous | 95% CI        |
| ------- | ------------------------------ | ------------- |
| 0.0     | −8.5%                          | [−10.3, −6.6] |
| 0.5     | −9.1%                          | [−10.9, −7.3] |
| 1.0     | −9.7%                          | [−11.9, −7.5] |
| 1.5     | −10.3%                         | [−12.3, −8.4] |
| 2.0     | **−11.2%**                     | [−13.5, −9.0] |

Endpoint span (κ=0 to κ=2): **−2.7% [−3.3, −2.2]** — also resolved.
Cross-validation against independent passive-to-active comparison: agrees at 0.00 pp.

### Platform Recommendation Example (Burgers PDE)

| Platform            | TOPS  | Cost  | Power  | Utilization | Fit                           | Score  |
| ------------------- | ----- | ----- | ------ | ----------- | ----------------------------- | ------ |
| STM32H7             | 0.082 | $8    | 400 mW | 0.027%      | Over-specified                | 214    |
| **Nordic nRF52840** | 0.026 | $5.00 | 15 mW  | 0.085%      | **Over-specified (selected)** | 252    |
| TI AM62A            | 2.0   | $35   | 2 W    | <0.001%     | Over-specified                | 17,500 |

### Carbon Footprint Comparison (5-year lifecycle, per device)

| Scenario                    | Training | Deployment | Total        | Reduction     |
| --------------------------- | -------- | ---------- | ------------ | ------------- |
| **Solar + Nordic nRF52840** | 0.036 kg | 0.31 kg    | **~5.35 kg** | **~45× less** |
| Grid + Jetson Orin          | 0.356 kg | 238 kg     | **238 kg**   | baseline      |

**Per-device saving: 233 kg CO₂. ~99.9% of this is from the hardware change
(Jetson → nRF52840); solar-constrained training contributes <1% (~0.32 kg).**

At scale (10,000 devices): ~2,330 metric tons CO₂ saved.
At scale (1M devices): ~233,000 metric tons CO₂ saved.

## 📈 Reproducing Results

> [`reproduce_paper.py`](#-reproduce-everything-one-command) implements exactly the
> protocol documented in this section and regenerates Tables III–V, the κ-sweep, and
> the carbon breakdown from the archived per-seed data in seconds. The estimator,
> normalization choices, and runtimes below describe that same protocol; the runtime
> table applies to **training from scratch** (`--retrain` / per-problem scripts),
> not to the default archived-data rebuild.

### Statistical Validation Protocol

All performance contrasts use the same paired bootstrap estimator:

```python
import numpy as np

def paired_bootstrap_ci(num_arr, denom_arr, n_boot=10000, seed=42):
    """Mean of per-seed ratios, 95% percentile CI (10k resamples)."""
    num = np.array(num_arr); den = np.array(denom_arr)
    ratios = (num - den) / den
    point_est = ratios.mean()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ratios), size=(n_boot, len(ratios)))
    boot = ratios[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point_est * 100, lo * 100, hi * 100  # return as percentages
```

### Normalization choices (per table)

- **Table V**: own-denominator (each contrast normalized to its own reference)
- **Table VI**: D-normalized (additive closure D→C + C→B + B→E = D→E, residual ≤ 4×10⁻¹⁴)
- **Table IV**: C-normalized (budget sensitivity relative to full-budget at high reg)
- **κ-sweep (Figure 3)**: D-normalized (improvement relative to continuous baseline)

### Expected Runtime

| Experiment               | Seeds            | Epochs | Wall Clock (CPU) |
| ------------------------ | ---------------- | ------ | ---------------- |
| Single problem           | 1                | 3000   | ~20 minutes      |
| Statistical validation   | 10               | 3000   | ~3.5 hours       |
| Full 7-problem sweep     | 10 × 7           | 3000   | ~25 hours        |
| Decomposition (Table VI) | 10 × 4 cells × 7 | 3000   | ~80 hours        |
| κ-sweep (Figure 3)       | 10 × 5           | 3000   | ~14 hours        |

**Note**: Solar-constrained training extends wall-clock time by ~2× due to the
50% duty cycle. All blessed results in the paper are CPU-backend runs to ensure
bit-exact reproducibility across machines.

## 🎓 Citation

```bibtex
@article{jurj2026right_sizing,
  author  = {Sorin Liviu Jurj},
  title   = {Physics Structure-Informed Neural Networks for Constraint-Preserving TinyML and Sustainable Edge Deployment},
  journal = {Under Review},
  year    = {2026},
  url     = {https://github.com/jurjsorinliviu/Psi-NNs-for-Sustainable-Edge-AI}
}
```

## 📚 Related Publications

1. **Ψ-HDL Framework**: [PSI-HDL GitHub](https://github.com/jurjsorinliviu/PSI-HDL)
2. **Original Ψ-NN**: [Psi-NN GitHub](https://github.com/ZitiLiu/Psi-NN)
3. **Ψ-xLSTM**: [Psi-xLSTM GitHub](https://github.com/jurjsorinliviu/Psi-xLSTM) ([IEEE Access, 2026](https://doi.org/10.1109/ACCESS.2026.3678809))

## 🔍 Key Findings Summary

### ✅ What's established

1. **Hardware right-sizing is the dominant sustainability lever** — of the 233 kg
   per-device carbon saving, ~99.9% comes from the Jetson → Nordic nRF52840 platform
   change (task-size-driven for the demonstrated models); <1% comes from solar-constrained
   training.

2. **For Burgers PDE, solar-constrained training has no accuracy penalty** —
   test MSE improves under the κ-mechanism relative to the continuous baseline
   (κ=0 to κ=2 span: −8.5% to −11.2%, all individually resolved at 95% CI).

3. **Under the deterministic 50% duty cycle with lossless Adam-state
   checkpointing, B→E = 0 by structural identity** — the interruption schedule
   reduces by construction to continuous training at the halved budget, not as
   an empirical finding about schedules in general. The load-bearing empirical
   contrasts are regularization (D→C) and budget (C→B). `revision/exp3` measures
   the schedule effects beyond this identity: a stochastic outage pattern changes
   the outcome by exactly zero under lossless checkpointing, while discarding the
   optimizer state on resume costs up to +1,984% (Advection).

### ⚠️ What's a deployment risk

4. **Budget sensitivity (C→B) spans 0.97×–23.7× in solution error** across the
   eleven-problem suite (9 of 11 resolved on the scale-invariant paired
   log-ratio). **No descriptor of the equation predicts this cost** (permutation
   test over PDE class, time dependence, nonlinearity, and derivative order:
   all p > 0.18); the three elliptic problems alone span 1.33× (Laplace) to
   23.7× (Poisson). **Practitioners must measure C→B per problem before
   committing to renewable-powered training.**

5. **Wave and Memristor** are the two unresolved cases even under the robust
   estimator: Wave's C→B point estimate is very high (+3474%) but its CI is so
   wide that the contrast is effectively uninterpretable; Memristor's CI crosses
   zero. Both are excluded from the resolved-9-of-11 framing for cause.

### 🔬 Methodology robustness

6. The orthogonal decomposition's additive closure (D→C + C→B + B→E = D→E)
   holds to numerical residual (≤ 4×10⁻¹⁴) on all problems where it is measured.
   The κ-sweep cross-validates the κ=2 endpoint at 0.00 pp against an independent
   passive-vs-active comparison.

## 🛠️ Hardware Platforms Database

| Tier          | Platform         | TOPS  | Memory  | Power  | Cost  | Technology       |
| ------------- | ---------------- | ----- | ------- | ------ | ----- | ---------------- |
| **TinyML**    | Nordic nRF52840  | 0.026 | 256 KB  | 15 mW  | $5.00 | Cortex-M4        |
| **TinyML**    | STM32H7          | 0.082 | 1024 KB | 400 mW | $8    | Cortex-M7        |
| **Mid-Range** | TI AM62A         | 2.0   | 2 MB    | 2 W    | $35   | Cortex-A53+CNN   |
| **Mid-Range** | TI TDA4VM        | 8.0   | 8 MB    | 4.5 W  | $80   | Cortex-A72+DSP   |
| **High-Perf** | Hailo-8          | 26.0  | 4 MB    | 5 W    | $150  | Neural Processor |
| **High-Perf** | Jetson Orin Nano | 40.0  | 8 MB    | 10 W   | $249  | Ampere GPU       |

## 🌍 Environmental Impact

### Single Device (5-year lifecycle)

- **Traditional**: Grid training + Jetson Orin Nano = **238 kg CO₂**
- **Our framework**: Solar training + Nordic nRF52840 = **~5.35 kg CO₂**
- **Reduction**: ~233 kg CO₂ per device (≈ 97.8% less, or ~45× less)
- **Attribution**: ~99.9% from hardware change; <1% from solar training itself

### At Scale

| Deployment       | Traditional  | Our Framework | Reduction    | Equivalent                      |
| ---------------- | ------------ | ------------- | ------------ | ------------------------------- |
| 1,000 devices    | 238 tons     | 5.35 tons     | 233 tons     | ~51 cars removed for 1 year     |
| 10,000 devices   | 2,380 tons   | 53.5 tons     | 2,330 tons   | ~507 cars / ~2,774 acres forest |
| 1M devices (IoT) | 238,000 tons | 5,350 tons    | 233,000 tons | ~50,700 cars / ~277,000 acres   |

*Equivalencies based on EPA averages (4.6 t CO₂/car/year; 0.84 t CO₂/acre/year forest sequestration).*

## 🤝 Contributing

Areas of interest:

- [ ] Additional PDE families (biharmonic, higher-order dispersive, systems of PDEs) to extend the budget-sensitivity map beyond the eleven benchmarks
- [ ] Extended platform database (Qualcomm, Google Coral, Intel Movidius, RISC-V, neuromorphic)
- [ ] Multi-physics coupled problems (thermoelasticity, MHD) to test whether physical coupling amplifies budget sensitivity
- [ ] Real hardware deployment validation with field solar measurements
- [ ] Wind + solar hybrid renewable strategies
- [ ] Reducing budget sensitivity for transport-dominated problems (Advection)

## 📞 Contact

**Sorin Liviu Jurj**  
Email: jurjsorinliviu@yahoo.de  
GitHub: [@jurjsorinliviu](https://github.com/jurjsorinliviu)

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Ψ-HDL Framework: [Psi-HDL GitHub](https://github.com/jurjsorinliviu/Psi-HDL)
- Original Ψ-NN: [Psi-NN GitHub](https://github.com/ZitiLiu/Psi-NN)
- Ψ-xLSTM: [Psi-xLSTM GitHub](https://github.com/jurjsorinliviu/Psi-xLSTM)

---

**Last Updated**: July 2026  
**Paper Status**: Submitted  
**Code Version**: v1.0

# Ψ-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the implementation of the research paper **"Ψ-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling"**.
<img width="2816" height="1536" alt="PSI-xLSTM Methodology Pipeline" src="https://github.com/user-attachments/assets/c8e427a4-4963-49ef-9675-92e016f635d2" />

## Key Features

- Recurrent Relation-Aware Distillation (RRAD) for compressing xLSTM-based teachers into efficient recurrent surrogates
- Time-constant discovery from recurrent dynamics for interpretable temporal structure
- Low-rank matrix-memory compression for reducing recurrent complexity
- Automated behavioral Verilog-A synthesis for SPICE-compatible deployment
- End-to-end SPICE verification and reviewer experiment automation

## Results Summary

| Model                | Parameters | MSE (x1e-8)   | Latency (us)   | Speedup | Compression |
| -------------------- | ---------- | ------------- | -------------- | ------- | ----------- |
| Baseline PINN        | 8,577      | 7.90 +- 3.06  | 0.191 +- 0.007 | 1.0x    | -           |
| xLSTM Teacher        | 46,409     | 7.44 +- 1.83  | 0.107 +- 0.019 | 0.6x    | 0%          |
| Psi-xLSTM Clustering | 16,961     | 7.55 +- 3.26  | 0.014 +- 0.000 | 7.6x    | 63.5%       |
| Psi-xLSTM Low-Rank   | 7,377      | 16.62 +- 0.00 | 0.026 +- 0.001 | 4.1x    | 84.1%       |

Main manuscript results are stored in `chapter4_results_improved/`. Consolidated reviewer experiment outputs are stored in `reviewer experiments/`.

## Final Revision Experiments

The final acceptance-stage revision analyses are reproduced with:

```bash
python run_final_revision_experiments.py --run-all --include-supporting-validation --use-ngspice-active-data --output-dir "final revision experiments"
```

Key outputs are written to `final revision experiments/`:

- `supporting_validation/`: regenerated reviewer-validation outputs used by the final revision
- `cross_device_analysis/`: cross-device regime metrics and regime-map figure
- `ood_sensitivity/`: impulse-like disturbance sensitivity tables and figure
- `final_revision_summary.json`: compact summary of the final revision artifacts

## Quick Start

### Installation

```bash
git clone https://github.com/jurjsorinliviu/Psi-xLSTM.git
cd PSI-xLSTM

conda create -n psi_xlstm python=3.8
conda activate psi_xlstm

pip install -r requirements.txt

# Optional: spreadsheet support for public experimental datasets
pip install openpyxl
```

### Main Manuscript Experiments

```bash
python run_chapter4_experiments_improved.py
```

Outputs are written to `chapter4_results_improved/`.

### Verilog-A Generation

```bash
python -m psi_xlstm.hdl_generation.xlstm_verilog_gen \
    --model chapter4_results_improved/seeds/seed_42/psi_xlstm_clustering_final.pth \
    --output ./chapter4_results_improved/hdl/
```

### SPICE Verification

```bash
python compare_ngspice_pytorch.py
```

Outputs are written to `chapter4_results_improved/spice_verification/`.

### Reviewer Experiment Suite

Run the full reviewer experiment package into a single consolidated folder:

```bash
python run_reviewer_experiments.py --run-all --use-default-public-suite --experimental-suite-auto-download --use-ngspice-active-data --output-dir "reviewer experiments"
```

Quick smoke run:

```bash
python run_reviewer_experiments.py --run-all --use-default-public-suite --experimental-suite-auto-download --quick --epochs 1 --output-dir "reviewer experiments"
```

Generate a template for a custom three-dataset suite:

```bash
python run_reviewer_experiments.py --write-public-suite-template reviewer_public_suite.json
python run_reviewer_experiments.py --experimental-suite-json reviewer_public_suite.json
```

The `reviewer experiments/` folder contains:

- `reviewer_experiment_summary.json`: top-level summary of all reviewer experiments
- `exp_sigmoid_vs_exponential/`: gate ablation tables and FFT figures
- `exp_spectral_baseline_comparison/`: PINN, Fourier, SIREN, and xLSTM comparison tables
- `exp_multidevice_validation/`: memristor, MOSFET, and BJT tables
- `exp_fft_offset_diagnostics/`: FFT offset figure and JSON summary
- `exp_dc_iv_matching/`: DC I-V figures and RMSE table
- `exp_active_small_large_signal/`: small-signal sweep and large-signal THD outputs
- `exp_experimental_suite/`: public experimental datasets, converted CSV files, and aggregate tables

## Repository Structure

```text
psi_xlstm/                         Main Python package
chapter4_results_improved/         Main manuscript experiment outputs
reviewer experiments/              Consolidated reviewer experiment outputs
final revision experiments/        Final acceptance-revision experiment outputs
run_chapter4_experiments_improved.py
run_reviewer_experiments.py
run_final_revision_experiments.py
compare_ngspice_pytorch.py
requirements.txt
README.md
```

## Methodology Overview

### 1. xLSTM-PINN Teacher

High-capacity teacher with exponential gating and matrix memory:

```python
from psi_xlstm.models import xLSTMTeacher

teacher = xLSTMTeacher(
    input_dim=2,
    hidden_size=64,
    num_layers=2,
    output_dim=1,
    use_mlstm=True,
    num_heads=4,
)
```

### 2. RRAD Distillation

Temporal knowledge transfer between recurrent models:

```python
from psi_xlstm.training import RecurrentRelationAwareDistillation

rrad = RecurrentRelationAwareDistillation(
    teacher=teacher,
    student=student,
    alpha=0.5,
    beta=0.5,
    gamma=0.1,
)
```

### 3. Structure Discovery

Time-constant clustering extracts interpretable temporal modes. Low-rank compression reduces recurrent matrix-memory cost while preserving useful dynamical structure.

### 4. Behavioral Verilog-A Synthesis

The trained recurrent surrogate is exported into SPICE-compatible behavioral HDL for downstream circuit simulation.

## Reproducing Results

### Complete Reproduction

```bash
python run_chapter4_experiments_improved.py
python compare_ngspice_pytorch.py
python run_reviewer_experiments.py --run-all --use-default-public-suite --experimental-suite-auto-download --use-ngspice-active-data --output-dir "reviewer experiments"
```

### Quick Validation

Pretrained manuscript models are available under `chapter4_results_improved/seeds/seed_42/`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{jurj2026psixlstm,
  title={Psi-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling},
  author={Jurj, Sorin Liviu},
  journal={},
  year={2026},
  note={Manuscript and code repository}
}
```

## Related Work
- Ψ-NN: https://www.nature.com/articles/s41467-025-64624-3
- Ψ-HDL: https://ieeexplore.ieee.org/document/11373324
- xLSTM: https://arxiv.org/abs/2405.04517
- xLSTM-PINN: https://arxiv.org/abs/2511.12512

## License

This project is licensed under the MIT License.

## Data Availability

Source code, trained models, experiment scripts, generated Verilog-A artifacts, and reviewer experiment outputs are included in this repository. Public experimental datasets used by `run_reviewer_experiments.py` are downloaded automatically from their original sources during execution unless already present locally.

## External Datasets

Validation in `run_reviewer_experiments.py` uses public third-party datasets from Zenodo and KIT RADAR under CC BY 4.0 licenses. Full attribution, DOI information, and source links are listed in [DATASETS.md](DATASETS.md).

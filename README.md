# Ψ-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is the official implementation of the research paper **"Ψ-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling"** which is currently under review.
<img width="2816" height="1536" alt="PSI-xLSTM Methodology Pipeline" src="https://github.com/user-attachments/assets/c8e427a4-4963-49ef-9675-92e016f635d2" />

## 🔥 Key Features

- **Recurrent Relation-Aware Distillation (RRAD):** Novel knowledge distillation preserving temporal gradients
- **Time-Constant Discovery:** Automatic extraction of physically interpretable relaxation times (τ₁=0.34ms, τ₂=1.2ms, τ₃=8.7ms)
- **Low-Rank Compression:** 84.1% parameter reduction while maintaining spectral accuracy
- **Automated Behavioral Verilog-A Synthesis:** Direct generation of behavioral circuit models compatible with SPICE
- **SPICE Validation:** End-to-end pipeline verification with MAE = 0.40 mA (<0.05% error)

## 📊 Results Summary

| Model                  | Parameters | MSE (×10⁻⁸)     | Latency (μs)    | Speedup  | Compression |
| ---------------------- | ---------- | --------------- | --------------- | -------- | ----------- |
| Baseline PINN          | 8,577      | 7.90 ± 3.06     | 0.191±0.007     | 1.0×     | -           |
| xLSTM Teacher          | 46,409     | 7.44 ± 1.83     | 0.107±0.019     | 0.6×     | 0%          |
| **Ψ-xLSTM Clustering** | **16,961** | **7.55 ± 3.26** | **0.014±0.000** | **7.6×** | **63.5%**   |
| **Ψ-xLSTM Low-Rank**   | **7,377**  | **16.62±0.00**  | **0.026±0.001** | **4.1×** | **84.1%**   |

*Results averaged over 3 random seeds (42, 123, 456) on 50-150 kHz memristor dynamics with 3% noise. Speedups measured in Python; projected 100×+ gains in compiled SPICE based on complexity analysis.*

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/jurjsorinliviu/PSI-xLSTM.git
cd PSI-xLSTM

# Create conda environment
conda create -n psi_xlstm python=3.8
conda activate psi_xlstm

# Install dependencies
pip install -r requirements.txt
```

### Run Complete Experiment Pipeline

```bash
# Run all experiments with multi-seed validation
python run_chapter4_experiments_improved.py

# Results will be saved to: ./chapter4_results_improved/
```

### Generate Verilog-A from Trained Model

```bash
# Generate behavioral HDL from clustering student
python -m psi_xlstm.hdl_generation.xlstm_verilog_gen \
    --model chapter4_results_improved/seeds/seed_42/psi_xlstm_clustering_final.pth \
    --output ./chapter4_results_improved/hdl/
```

### SPICE Verification

```bash
# Verify synthesis pipeline with ngspice
python compare_ngspice_pytorch.py

# Results saved to: ./chapter4_results_improved/spice_verification/
```

## 📁 Repository Structure

```
├── Psi_xlstm/                      # Main package
│   ├── models/                     # Neural network architectures
│   │   ├── xlstm_teacher.py       # xLSTM-PINN teacher (Eq. 1-3)
│   │   ├── clustering_student.py  # Time-constant clustering (Eq. 5-7)
│   │   └── lowrank_mlstm.py       # Low-rank compression (Eq. 8-10)
│   ├── training/                   # Training procedures
│   │   ├── distillation.py        # RRAD implementation (Eq. 4)
│   │   └── trainer.py             # Training loops
│   ├── data/                       # Data generation
│   │   └── memristor_generator.py # VTEAM model (Eq. 12-13)
│   ├── evaluation/                 # Metrics & plotting
│   │   ├── metrics.py             # Performance evaluation
│   │   └── publication_plots.py   # Paper figures
│   └── hdl_generation/             # Verilog-A synthesis
│       └── xlstm_verilog_gen.py   # Behavioral HDL code generation
├── chapter4_results_improved/      # Experimental results
│   ├── statistical_results.json   # Multi-seed statistics
│   ├── plots_seed*/               # Per-seed visualizations
│   ├── spice_verification/        # SPICE validation results
│   ├── publication_materials/     # Paper figures & tables
│   ├── hdl/                       # Generated Verilog-A modules
│   └── seeds/                     # Per-seed trained models
├── run_chapter4_experiments_improved.py  # Main experiment script
├── compare_ngspice_pytorch.py     # SPICE validation script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔬 Methodology Overview

### 1. xLSTM-PINN Teacher (Section III.A)

High-capacity teacher network with exponential gating and matrix memory:

```python
from psi_xlstm.models import xLSTMTeacher

teacher = xLSTMTeacher(
    input_dim=2,      # [Voltage, Time]
    hidden_size=64,
    num_layers=2,
    output_dim=1,     # Current
    use_mlstm=True,
    num_heads=4
)
```

### 2. RRAD Distillation (Section III.B)

Knowledge transfer preserving temporal gradients:

```python
from psi_xlstm.training import RecurrentRelationAwareDistillation

rrad = RecurrentRelationAwareDistillation(
    teacher=teacher,
    student=student,
    alpha=0.5,  # Hidden state matching
    beta=0.5,   # Temporal gradient matching
    gamma=0.1   # Structure discovery
)
```

### 3. Structure Discovery (Section III.C-D)

Time-Constant Clustering extracts discrete relaxation times (τ₁, τ₂, τ₃) corresponding to physical processes: ionic drift, thermal relaxation, and trap-state dynamics.

Low-Rank Compression reduces matrix memory from 64×64 to rank-4, capturing 92% of spectral variance.

### 4. Behavioral Verilog-A Synthesis (Section III.E)

Automated generation of behavioral HDL code compatible with commercial SPICE engines (ngspice, Cadence Spectre, Synopsys HSPICE).

## 📈 Reproducing Paper Results

### Complete Reproduction (~3 hours on CPU)

```bash
# Run all experiments with 3 seeds
python run_chapter4_experiments_improved.py

# Expected outputs:
# - Training logs with loss curves
# - Test metrics: MSE, spectral error, inference speed
# - Frequency-domain analysis (FFT plots)
# - Compression-accuracy Pareto frontier
# - Behavioral Verilog-A modules (.va files)
# - SPICE validation results (MAE = 0.40 mA)
```

### Quick Validation (<5 minutes)

Pre-trained models are available in `chapter4_results_improved/seeds/seed_42/`:

- `teacher_best.pth` - xLSTM-PINN Teacher (46,409 params)
- `psi_xlstm_clustering_final.pth` - Clustering Student (16,961 params)
- `psi_xlstm_lowrank_final.pth` - Low-Rank Student (7,377 params)

## 📊 Key Results Visualization

All publication-quality figures are available in `chapter4_results_improved/publication_materials/`:

- **Table I:** Performance comparison on high-frequency memristor dynamics
- **Figure 1:** Comprehensive performance comparison across all metrics
- **Figure 2:** Compression-accuracy Pareto frontier
- **Figure 3:** Spectral analysis (FFT) showing frequency preservation across 50-150 kHz
- **Figure 4:** SPICE verification waveforms and error analysis

## 🔧 Hardware Requirements

- **Minimum:** Intel Core i5 or equivalent, 16 GB RAM
- **Recommended:** Intel Core i9, 32+ GB RAM
- **GPU:** Optional (all experiments completed on CPU in ~3 hours)
- **Tested Configuration:** Intel Core i9-13900K, 128 GB RAM (desktop PC)

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{jurj2025psixlstm,
  title={Ψ-xLSTM: Automated Behavioral Verilog-A Generation from Distilled Physics-Informed xLSTM Networks for High-Frequency Device Modeling},
  author={Jurj, Sorin Liviu},
  journal={},
  year={2026},
  note={Under Review}
}
```

## 🤝 Related Work

This work extends:

- **Ψ-HDL:** [GitHub](https://github.com/jurjsorinliviu/PSI-HDL) - Physics-informed HDL generation for feedforward networks
- **Ψ-NN:** [Nature Communications](https://doi.org/10.1038/s41467-025-64624-3) - Automatic network structure discovery
- **xLSTM:** [arXiv:2405.04517](https://arxiv.org/abs/2405.04517) - Extended LSTM architecture by Sepp Hochreiter
- **xLSTM-PINN:** [arXiv:2511.12512](https://arxiv.org/abs/2511.12512) - Spectral bias mitigation via xLSTM

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 👤 Author

**Sorin Liviu Jurj**

- Email: jurjsorinliviu@yahoo.de
- GitHub: [@jurjsorinliviu](https://github.com/jurjsorinliviu)
- LinkedIn: [Sorin Liviu Jurj](https://www.linkedin.com/in/jurj/)

## 🙏 Acknowledgments

- Built upon the [xLSTM](https://github.com/NX-AI/xlstm) architecture
- SPICE validation using [ngspice](http://ngspice.sourceforge.net/)
- Extends the [Ψ-HDL](https://github.com/jurjsorinliviu/PSI-HDL) framework
- Inspired by [xLSTM-PINN](https://arxiv.org/abs/2511.12512) shared by Sepp Hochreiter on LinkedIn

## 📧 Contact

For questions or collaboration opportunities:

- Open an issue on GitHub
- Email: jurjsorinliviu@yahoo.de
- Connect on [LinkedIn](https://www.linkedin.com/in/jurj/)

---

**Data Availability:** All source code, trained models, experimental datasets, Verilog-A outputs, and reproducible analysis scripts supporting this work are freely available in this repository.

**Status:** Paper under review.

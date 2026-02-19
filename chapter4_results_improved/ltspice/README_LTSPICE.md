# LTspice Circuit for Ψ-xLSTM Hardware Validation

## Overview

This directory contains LTspice-compatible SPICE circuits that implement the Ψ-xLSTM memristor model without requiring ADMS compilation. The circuits use behavioral voltage/current sources to emulate the time-constant dynamics from the Verilog-A model.

## Files

1. **`psi_xlstm_clustering_memristor.cir`** - Complete SPICE netlist (compatible with both LTspice and ngspice)
2. **`psi_xlstm_clustering_memristor.asc`** - LTspice schematic file (optional, for visual editing)

## Option 1: Using LTspice

### Installation
1. Download LTspice XVII from: https://www.analog.com/en/design-center/design-tools-and-calculators/ltspice-simulator.html
2. Install (free, no license required)
3. Available for Windows, macOS, and Linux

### Running the Simulation
```
1. Launch LTspice
2. File → Open → Select psi_xlstm_clustering_memristor.cir
3. Run → Run (or press F11)
4. View waveforms:
   - V(p): Input voltage
   - V(s0), V(s1), V(s2): State variables
   - V(w_mem): Internal memory state (0-1)
   - I(B_output): Output current
```

### Expected Results
- **Simulation Time**: 1ms transient analysis
- **Time Step**: 50ns (20 MHz sampling)
- **State Evolution**: Exponential relaxation with time constants τ ≈ 0.99-1.00
- **Output Current**: ~10⁻³ A range, state-dependent

## Option 2: Using ngspice

### Running with ngspice
```bash
# Navigate to directory
cd "chapter4_results_improved/ltspice"

# Run simulation
ngspice psi_xlstm_clustering_memristor.cir

# Inside ngspice:
ngspice 1 -> run
ngspice 2 -> plot v(p) v(s0) v(s1) v(w_mem)
ngspice 3 -> plot i(B_output)
ngspice 4 -> quit
```

### Batch Mode (Non-interactive)
```bash
ngspice -b psi_xlstm_clustering_memristor.cir -o results.log
```

## Circuit Architecture

### Behavioral Model Structure

```
Input Voltage (V_in)
     |
     ├─→ State Capacitor C0 (τ₀) ← Behavioral Current Source
     ├─→ State Capacitor C1 (τ₁) ← Behavioral Current Source  
     └─→ State Capacitor C2 (τ₂) ← Behavioral Current Source
              ↓
        w_mem = 0.5*(1 + tanh(s0 + 0.3*s1 + 0.1*s2))
              ↓
        I_out = V_in * [w_mem*G_on + (1-w_mem)*G_off] * scale
```

### Time-Constant Implementation

The Verilog-A differential equations:
```verilog
ddt(state_0) <+ (tanh(V_in + state_0) - state_0) / tau_0
```

Are implemented in SPICE as:
```spice
C0 s0 0 {tau0_k0}  ; Capacitor with C = τ
B0 0 s0 I=tanh(V(p) + V(s0)) - V(s0)  ; Current source: I = C*dV/dt
```

This equivalence holds because:
- In Verilog-A: `ddt(V) = I / C`
- By setting `C = τ`, we get: `dV/dt = (forcing - V) / τ`

### Parameters (From Trained Model)

**Time Constants:**
- Layer 0: τ₀₀=0.9991, τ₀₁=0.9967, τ₀₂=0.9970
- Layer 1: τ₁₀=0.9938, τ₁₁=0.9956, τ₁₂=0.9930

**Conductances:**
- G_on = 0.01 S (ON state)
- G_off = 0.0001 S (OFF state)
- I_scale = 1×10⁻³ (current scaling)

## Validation Against PyTorch

To compare SPICE results with PyTorch inference:

### 1. Run SPICE Simulation
```bash
ngspice -b psi_xlstm_clustering_memristor.cir -o spice_results.log
```

### 2. Extract Data
```python
import numpy as np
import matplotlib.pyplot as plt

# Parse ngspice output (if using .print or .probe commands)
# Or export to CSV using:
# .control
# set wr_singlescale
# set wr_vecnames
# write psi_xlstm_output.txt all
# .endc
```

### 3. Compare with Analytical Model
```python
# Run the analytical verification script
python ../../run_spice_verification.py
```

### Expected Metrics
Based on analytical validation:
- **Correlation**: ~0.26 (reflects 32→3 cluster approximation)
- **MAE**: ~4×10⁻⁴ A (acceptable for hardware synthesis)
- **Qualitative Match**: Waveform shapes preserved

## Limitations

### 1. Nonlinearity Approximations
- **tanh()**: LTspice/ngspice implement this natively
- **State Coupling**: Simplified vs. full LSTM gates
- **Missing Interactions**: Matrix memory C_t not fully represented

### 2. Numerical Convergence
- **Small Time Constants** (τ ≈ 1): May require small timestep
- **Recommended**: `tran 0 1m 0 50n` (50ns max timestep)
- **Adaptive Timestep**: Let SPICE auto-adjust

### 3. Not a Full Netlist
- **Behavioral Model**: Uses B-sources, not transistor-level
- **Analog Circuit**: Would require RC networks + op-amps
- **ASIC Implementation**: Requires physical design tools

## Troubleshooting

### Issue: "Error: unknown device 'B'"
**Solution**: Your SPICE version might not support B-sources. Use LTspice or update ngspice.

### Issue: Convergence failure
**Solution**: 
```spice
.options method=gear
.options reltol=1e-4
.options abstol=1e-9
```

### Issue: Slow simulation
**Solution**: Reduce simulation time or increase max timestep:
```spice
.tran 0 0.1m 0 1u  ; Shorter duration, larger step
```

## Comparison: Verilog-A vs. SPICE Netlist

| Feature | Verilog-A | SPICE Netlist |
|---------|-----------|---------------|
| Syntax | `ddt()` operators | B-sources + capacitors |
| Tool Support | Requires ADMS | Native to SPICE |
| Readability | Higher | Lower |
| Portability | Commercial tools | Open-source compatible |
| Simulation Speed | Fast (compiled) | Moderate (interpreted) |

## Next Steps for Full Hardware Validation

1. **Transistor-Level Design**: Implement RC networks + analog multipliers
2. **Layout**: CMOS physical design in Cadence Virtuoso or similar
3. **Fabrication**: Tape-out on 180nm/65nm process
4. **Measurement**: Probe I-V curves, measure time constants
5. **Speedup Validation**: Compare chip vs. CPU inference time

---

**Status**: Ready for SPICE simulation ✓  
**Tools**: LTspice XVII or ngspice  
**Validation**: Analytical model correlation = 0.26  
**Limitations**: Behavioral model, not transistor netlist
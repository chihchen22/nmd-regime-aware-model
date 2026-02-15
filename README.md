# Beyond Static Assumptions: A Comprehensive Analysis of Nonmaturity Deposit Behavioral Modeling Using Market-Implied Volatility and Path-Dependent Simulation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the Python implementation and research outputs for the paper:

**"Beyond Static Assumptions: A Comprehensive Analysis of Nonmaturity Deposit Behavioral Modeling Using Market-Implied Volatility and Path-Dependent Simulation"**

Author: Chih L. Chen, BTRM, CFA, FRM  
Date: February 2026

📄 **[Read the Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5997414)**

---

## Overview

This framework provides a comprehensive comparison of three distinct approaches to nonmaturity deposit (NMD) behavioral modeling, calibrated to current market conditions using the complete **SOFR cap/floor volatility surface** (October 2025). The analysis implements a **Hull-White stochastic interest rate model** for Monte Carlo simulation and quantifies embedded customer optionality that traditional static approaches systematically miss.

### Three Modeling Approaches

1. **Traditional Bifurcation Model** — Static allocation into stable/nonstable categories with sensitivity analysis across 0%–40% nonstable allocations
2. **Analytical Closed-Form Approximation** — Lognormal deposit evolution using market-implied volatility and integrated variance calculations
3. **Monte Carlo Simulation** — 5,000-path simulation with path-dependent customer behavioral dynamics and balance-sensitive rate sensitivity feedback loops

### Key Research Contributions

1. **Analytical Model Systematic Bias**: The analytical approximation overestimates weighted average life (WAL) by **14.8%** relative to Monte Carlo, despite incorporating market-implied volatility — revealing fundamental limitations of closed-form approaches in capturing behavioral feedback loops
2. **Embedded Optionality Quantification**: Monte Carlo captures **7× to 174×** more embedded customer optionality than analytical approximations, with the differential increasing dramatically at longer horizons
3. **Bifurcation Calibration Insight**: Monte Carlo WAL corresponds to approximately **3.2% nonstable allocation**, substantially lower than the 20% commonly assumed in practice
4. **Path-Dependent Behavioral Feedback**: Customers experiencing early adverse rate scenarios exhibit higher subsequent outflow propensities — a positive feedback loop that static models cannot represent

### Core Model Equations

**Hull-White Stochastic Rate Model:**
```
dr_t = κ(θ_t - r_t)dt + σ(t)dW_t
```

**Component Decay Model:**
```
B(t+1) = B(t) × (1 - h) × exp[g - β(B) × (r_t + credit_spread)]

where:
  β(B) = β_base × (1 + γ × ln(B / B₀))
```

Balance-sensitive β reflects empirical observations that larger depositors exhibit higher rate awareness and lower switching costs, creating nonlinear relationships between account characteristics and behavioral responses.

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/chihchen22/nmd-regime-aware-model.git
cd nmd-regime-aware-model

# Install dependencies
pip install numpy pandas scipy matplotlib
```

### Running the Model

```bash
# Run the comprehensive framework (reproduces all paper results)
python nmd_comprehensive_framework.py
```

This generates all CSV outputs used in the paper, including volatility surface data, simulated rate paths, balance evolution profiles, bifurcation sensitivity results, and analytical approximation comparisons.

---

## Repository Structure

```
nmd-regime-aware-model/
│
├── README.md                              # This file
├── LICENSE                                # MIT License
│
├── nmd_comprehensive_framework.py         # Complete framework (START HERE)
│                                          #   - Volatility surface construction
│                                          #   - Hull-White calibration
│                                          #   - Monte Carlo simulation
│                                          #   - Component decay model
│                                          #   - Bifurcation sensitivity analysis
│                                          #   - Analytical approximation
│                                          #   - CSV export
│
├── chart_script.py                        # Visualization and figure generation
│
├── SOFR_Market_Data_20250930.xlsx         # Input: October 2025 market data
├── SOFR_Market_Data_20220331.xlsx         # Input: March 2022 market data
├── SOFR_History.xlsx                      # Input: Historical SOFR rates
│
└── Output CSVs (generated):
    ├── volatility_surface_complete.csv    # Full extended volatility surface
    ├── sofr_paths_simulation.csv          # Simulated SOFR rate paths
    ├── balance_paths_simulation.csv       # Monte Carlo balance evolution
    ├── bifurcation_profiles_detailed.csv  # Bifurcation allocation profiles
    ├── bifurcation_sensitivity_analysis.csv  # WAL by nonstable %
    └── analytical_approximation_results.csv  # Analytical vs MC comparison
```

---

## Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| h | 0.01 | Monthly account closure rate (~12% annual) |
| g | 0.0017 | Monthly organic growth (~2% annual) |
| β_base | 0.20 | Base rate sensitivity |
| γ | 0.50 | Balance scaling factor for rate sensitivity |
| κ | 0.025 | Hull-White mean reversion speed |
| credit_spread | varies | FHLB-SOFR credit spread |
| n_paths | 5,000 | Monte Carlo simulation paths |
| n_months | 360 | Projection horizon (30 years) |

### Volatility Surface Calibration

| Maturity | ATM Volatility (bp) | Description |
|----------|---------------------|-------------|
| 1 month | 117 | Extrapolated short-term policy uncertainty |
| 1 year | 87 | Market-observed (Tullett Prebon) |
| 5 years | 52 | Market-observed |
| 10 years | 42 | Market-observed |
| 20 years | 33 | Market-observed |

Market data: 11 strike levels (1.00%–7.00%) × 12 maturities (1Y–20Y), extended to 1M–360M through econometric extrapolation.

---

## Key Results

### Weighted Average Life Comparison

| Methodology | WAL (years) | vs Monte Carlo |
|-------------|-------------|----------------|
| Deterministic (Market-Implied Forward) | 6.09 | +3.2% |
| **Monte Carlo Simulation** | **5.90** | **Benchmark** |
| Analytical Approximation | 6.77 | +14.8% |
| Calibrated Bifurcation (3.2% nonstable) | 5.90 | Match |

### Embedded Optionality (Expected − Stable Balance)

| Horizon | Analytical | Monte Carlo | MC / Analytical Ratio |
|---------|-----------|-------------|----------------------|
| 1 Year | 0.0024 | 0.0169 | **7.0×** |
| 2 Years | 0.0024 | 0.0340 | **14.3×** |
| 5 Years | 0.0014 | 0.0521 | **36.5×** |
| 10 Years | 0.0005 | 0.0379 | **79.8×** |
| 20 Years | 0.0001 | 0.0180 | **173.9×** |

Monte Carlo optionality peaks around 5-year horizons before declining — reflecting the balance between accumulated behavioral uncertainty and natural deposit decay. The analytical approach shows monotonic decline, missing the complex dynamics that create optimal exercise opportunities at intermediate horizons.

### Bifurcation Sensitivity Analysis

| Nonstable Allocation | WAL (years) |
|----------------------|-------------|
| 0% | 6.09 |
| 5% | 5.79 |
| 10% | 5.48 |
| 20% | 4.88 |
| 30% | 4.27 |
| 40% | 3.66 |

**Key Finding**: The Monte Carlo WAL of 5.90 years maps to **~3.2% nonstable allocation** — far below the 20% commonly used in practice, suggesting traditional approaches may significantly underestimate deposit stability under current market conditions.

### Early-Period Retention

| Period | Deterministic | Analytical | Monte Carlo |
|--------|---------------|-----------|-------------|
| 6-Month | 90.7% | 91.0% | 90.5% |
| 12-Month | 82.8% | 83.4% | 82.1% |

---

## Methodology Highlights

### Volatility Surface Construction
- Complete SOFR cap/floor surface from Tullett Prebon (October 2025)
- Short-term extrapolation using exponential decay: σ(T) = σ₁ᵧ + (σ₁ₘ − σ₁ᵧ) × e^(−λT)
- Fixes "flat early decay" artifact present when volatility begins at 1-year maturity
- Forward curve extends from current SOFR through market-implied forwards to 4.5% long-term equilibrium

### Path-Dependent Behavioral Dynamics
- **Behavioral feedback loops**: Declining balances reduce relationship stickiness, accelerating further decay
- **American-style option analogy**: Customers continuously evaluate withdrawal options based on path-specific information
- **Balance-sensitive β**: Rate sensitivity ranges from 5% floor (depleted accounts) to ~36% (full-balance accounts)
- **Composition effects**: As rate-sensitive customers exit, remaining portfolio becomes progressively less sensitive

### Model Selection Guidance

| Application | Recommended Approach |
|-------------|---------------------|
| Daily risk reporting | Analytical (with MC calibration adjustment) |
| Standard ALM analysis | Analytical or Bifurcation |
| Stress testing (CCAR/DFAST) | Monte Carlo |
| Deposit franchise valuation | Monte Carlo |
| Strategic decision-making | Monte Carlo |
| Regulatory capital assessment | Monte Carlo |

---

## Requirements

```
Python 3.8+
numpy
pandas
scipy
matplotlib
```

Install all dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

---

## Citation

If you use this code or methodology in academic work, please cite:

```bibtex
@article{chen2026beyondstatic,
  title={Beyond Static Assumptions: A Comprehensive Analysis of Nonmaturity Deposit Behavioral Modeling Using Market-Implied Volatility and Path-Dependent Simulation},
  author={Chen, Chih L.},
  year={2026},
  note={Working Paper, Available at SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5997414}
}
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Contact

Chih L. Chen, BTRM, CFA, FRM  
📧 chihchen22@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/chih-chen-60a185/)  
🐙 [GitHub](https://github.com/chihchen22)

For questions or collaboration inquiries, please open an issue on this repository.

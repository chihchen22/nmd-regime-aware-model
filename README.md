# Beyond Static Bifurcation: A Regime-Aware Approach to Nonmaturity Deposit Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the Python implementation and research outputs for the paper:

**"Beyond Static Bifurcation: A Regime-Aware Approach to Nonmaturity Deposit Modeling"**

Author: Chih L. Chen, BTRM, CFA, FRM  
Date: January 2026

📄 **[Read the Paper on SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5997414)**

---

## Overview

This framework provides a comprehensive analysis of nonmaturity deposit (NMD) behavioral dynamics using a **regime-aware component decay model**. The key innovation is the **regime amplification factor (α)** that captures how deposit sensitivity increases when current rates exceed their trailing moving average, without requiring subjective stable/non-stable bifurcation.

### Key Contributions

1. **Convergence Zone Validation**: A testable criterion for α calibration based on P05 WAL convergence across rate regimes
2. **Dual-Sensitivity Framework**: Integrates both rate sensitivity and credit sensitivity (FHLB-SOFR spread)
3. **Parameter Dominance Finding**: Behavioral parameter uncertainty (±2 years WAL) dwarfs methodological choice (0.13 years)
4. **Surge Deposit Dynamics**: Captures transitory balances accumulated during low-rate periods without explicit cohort tracking

### Core Model Equation

```
B(t+1) = B(t) × (1-h) × exp[g - β_rate × r(t) - β_credit × s(t)]

where:
  β_rate = β_rate_0 × (B/B_0)^γ_rate × [1 + α × max(0, r - r_MA)]
```

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/chihchen22/nmd-regime-aware-model.git
cd nmd-regime-aware-model

# Install dependencies
pip install numpy pandas scipy matplotlib openpyxl
```

### Running the Model

```bash
# Run all analyses with default parameters (α=2.0, MA=24 months)
python nmd_master.py

# Run specific module
python nmd_master.py --module sensitivity_2d

# Custom parameters
python nmd_master.py --alpha 1.5 --ma-window 12

# List available modules
python nmd_master.py --list
```

---

## Repository Structure

```
nmd-regime-aware-model/
│
├── README.md                         # This file
├── LICENSE                           # MIT License
│
├── nmd_master.py                     # Master controller script (START HERE)
├── config.py                         # Central parameter configuration
│
├── comprehensive_nmd_model.py        # Core NMD model (Hull-White + Component Decay)
├── sensitivity_matrix_2d.py          # 2D sensitivity analysis (α × MA window)
├── dual_period_alpha_sensitivity.py  # March 2022 vs September 2025 comparison
├── regime_amplification_analysis.py  # Regime effects analysis
├── parameter_sensitivity.py          # Behavioral parameter sensitivity
├── validate_model.py                 # Model validation utilities
│
├── market_data_loader.py             # Market data extraction (SOFR curves)
├── sabr_volatility.py                # SABR volatility calibration
│
├── SOFR_Market_Data_20250930.xlsx    # Input: September 2025 market data
├── SOFR_Market_Data_20220331.xlsx    # Input: March 2022 market data
├── SOFR_History.xlsx                 # Input: Historical SOFR rates
│
└── model_outputs/                    # Generated outputs (CSV + PNG)
    ├── 01_bootstrapped_curve.csv
    ├── 02_forward_rates.csv
    ├── ...
    ├── fig1_forward_curve.png
    ├── fig2_total_balance_comparison.png
    └── ...
```

---

## Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **α (alpha)** | 2.0 | Regime amplification factor |
| **MA Window** | 24 months | Moving average lookback for regime detection |
| β_rate_0 | 0.30 | Base rate sensitivity |
| β_credit_0 | 0.15 | Base credit sensitivity |
| γ_rate | 0.30 | Rate sensitivity elasticity (balance scaling) |
| γ_credit | 0.40 | Credit sensitivity elasticity |
| h | 0.01 | Monthly closure rate (~12% annual) |
| g | 0.0017 | Monthly organic growth (~2% annual) |
| n_paths | 5,000 | Monte Carlo simulation paths |
| n_months | 360 | Projection horizon (30 years) |

---

## Key Results

### Dual-Period Validation (Table 2 in Paper)

| Metric | March 2022 (Transition) | September 2025 (Stable) |
|--------|-------------------------|-------------------------|
| Mean WAL | 4.61 years | 4.85 years |
| P05 Stable WAL | 3.09 years | 4.02 years |
| Mean - P05 Gap | 1.52 years | 0.85 years |
| **Tail Risk Compression** | — | **44%** |

### Parameter Sensitivity Ranking

| Rank | Parameter | WAL Impact |
|------|-----------|------------|
| 1 | Closure Rate (h) | ±1.25 years |
| 2 | Rate Sensitivity (β_rate) | ±0.89 years |
| 3 | Credit Sensitivity (β_credit) | ±0.49 years |
| 4 | Methodological Choice | 0.13 years |

**Key Finding**: Behavioral parameter uncertainty exceeds methodological choice by 10-15×.

---

## Analysis Modules

| Module | Script | Description |
|--------|--------|-------------|
| Base Model | `comprehensive_nmd_model.py` | Hull-White calibration, Monte Carlo simulation, WAL calculation |
| 2D Sensitivity | `sensitivity_matrix_2d.py` | α × MA window grid analysis with heatmaps |
| Dual Period | `dual_period_alpha_sensitivity.py` | March 2022 vs September 2025 comparison |
| Regime Analysis | `regime_amplification_analysis.py` | Regime excursion effects |
| Parameter Sensitivity | `parameter_sensitivity.py` | h, β_rate, β_credit tornado analysis |

---

## Output Files

### Data (CSV)
| File | Description |
|------|-------------|
| `06_wal_comparison.csv` | WAL by methodology (MC, Analytical, Bifurcation) |
| `dual_period_regime_comparison.csv` | March 2022 vs September 2025 metrics |
| `sensitivity_matrix_2d.csv` | Full α × MA sensitivity grid |
| `parameter_sensitivity_results.csv` | Behavioral parameter impacts |
| `balance_size_sensitivity.csv` | WAL by account size tier |
| `spread_stress_results.csv` | Credit stress scenario analysis |

### Figures (PNG)
| File | Description |
|------|-------------|
| `fig1_forward_curve.png` | SOFR forward curve (September 2025) |
| `fig1_march2022_forward_curve.png` | SOFR forward curve (March 2022) |
| `fig2_total_balance_comparison.png` | Balance evolution by methodology |
| `fig3_stable_balance_comparison.png` | P05 stable balance comparison |
| `fig5_wal_decomposition.png` | WAL waterfall decomposition |
| `sensitivity_heatmaps_by_period.png` | 2D sensitivity heatmaps |
| `parameter_sensitivity_tornado.png` | Parameter impact ranking |

---

## Requirements

```
Python 3.8+
numpy
pandas  
scipy
matplotlib
openpyxl
```

Install all dependencies:
```bash
pip install numpy pandas scipy matplotlib openpyxl
```

---

## Citation

If you use this code or methodology in academic work, please cite:

```bibtex
@article{chen2026bifurcation,
  title={Beyond Static Bifurcation: A Regime-Aware Approach to Nonmaturity Deposit Modeling},
  author={Chen, Chih L.},
  year={2026},
  note={Working Paper}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

Chih L. Chen, BTRM, CFA, FRM  
📧 chihchen22@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/chih-chen-60a185/)  
🐙 [GitHub](https://github.com/chihchen22)

For questions or collaboration inquiries, please open an issue on this repository.

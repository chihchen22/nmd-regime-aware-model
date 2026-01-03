"""
================================================================================
NMD MODEL - SHARED CONFIGURATION
================================================================================
Central configuration module for all NMD analysis scripts.

Import this module to use consistent parameters across all analyses:

    from config import PARAMS, get_params

The PARAMS dictionary contains all default parameter values.
Use get_params() to get a fresh copy that can be modified.

================================================================================
"""

import os
import numpy as np
from datetime import datetime

# ============================================================================
# CORE PARAMETERS - MODIFY THESE TO CHANGE DEFAULTS
# ============================================================================

# Regime Amplification (KEY PARAMETERS)
ALPHA = 2.0                      # Regime amplification factor
MA_WINDOW = 24                   # Moving average window (months)

# Base Sensitivities
BETA_RATE_0 = 0.30               # Base rate sensitivity
BETA_CREDIT_0 = 0.15             # Base credit sensitivity
GAMMA_RATE = 0.30                # Rate sensitivity elasticity
GAMMA_CREDIT = 0.40              # Credit sensitivity elasticity

# Natural Decay
H = 0.01                         # Monthly attrition rate (~12% annual)
G = 0.0017                       # Monthly organic growth (~2% annual)

# Simulation Settings
N_PATHS = 5000                   # Monte Carlo paths
N_MONTHS = 360                   # Projection horizon (30 years)
INITIAL_BALANCE = 1_000_000.0    # Starting balance ($1M)

# Hull-White Calibration
KAPPA = 0.03                     # Mean reversion speed

# Credit Spread
CREDIT_SPREAD_BASE = 0.005       # 50 bps base spread
CREDIT_SPREAD_SHOCK = 0.015      # 150 bps stress scenario

# Output Directory
OUTPUT_DIR = 'model_outputs'

# Random Seed
RANDOM_SEED = 42


# ============================================================================
# PARAMETER DICTIONARY - USED BY ALL MODULES
# ============================================================================

PARAMS = {
    # Regime amplification
    'alpha': ALPHA,
    'ma_window': MA_WINDOW,
    
    # Base sensitivities
    'beta_rate_0': BETA_RATE_0,
    'beta_credit_0': BETA_CREDIT_0,
    'gamma_rate': GAMMA_RATE,
    'gamma_credit': GAMMA_CREDIT,
    
    # Natural decay
    'h': H,
    'g': G,
    
    # Simulation
    'n_paths': N_PATHS,
    'n_months': N_MONTHS,
    'initial_balance': INITIAL_BALANCE,
    'B_0': INITIAL_BALANCE,  # Alias for backward compatibility
    
    # Hull-White
    'kappa': KAPPA,
    
    # Credit spread
    'credit_spread_base': CREDIT_SPREAD_BASE,
    'credit_spread_shock': CREDIT_SPREAD_SHOCK,
    
    # Output
    'output_dir': OUTPUT_DIR,
    
    # Random seed
    'random_seed': RANDOM_SEED,
}


# ============================================================================
# MARKET DATA FILE PATHS
# ============================================================================

DATA_FILES = {
    'sofr_sep25': 'SOFR_Market_Data_20250930.xlsx',
    'sofr_mar22': 'SOFR_Market_Data_20220331.xlsx',
    'sofr_history': 'SOFR_History.xlsx',
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_params(**overrides):
    """
    Get a copy of default parameters with optional overrides.
    
    Parameters:
        **overrides: Key-value pairs to override defaults
    
    Returns:
        Dictionary of parameters
    
    Example:
        # Get defaults
        params = get_params()
        
        # Override alpha
        params = get_params(alpha=1.5)
        
        # Override multiple
        params = get_params(alpha=2.5, ma_window=12, n_paths=10000)
    """
    params = PARAMS.copy()
    params.update(overrides)
    
    # Keep B_0 in sync with initial_balance
    if 'initial_balance' in overrides:
        params['B_0'] = overrides['initial_balance']
    
    return params


def ensure_output_dir(output_dir=None):
    """Ensure output directory exists."""
    dir_path = output_dir or OUTPUT_DIR
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def set_random_seed(seed=None):
    """Set numpy random seed for reproducibility."""
    np.random.seed(seed or RANDOM_SEED)


def print_params(params=None):
    """Print formatted parameter summary."""
    p = params or PARAMS
    
    print("\n" + "=" * 70)
    print("MODEL PARAMETERS")
    print("=" * 70)
    print(f"\nRegime Amplification:")
    print(f"  α (alpha):           {p['alpha']}")
    print(f"  MA Window:           {p['ma_window']} months")
    
    print(f"\nBase Sensitivities:")
    print(f"  β_rate_0:            {p['beta_rate_0']}")
    print(f"  β_credit_0:          {p['beta_credit_0']}")
    print(f"  γ_rate:              {p['gamma_rate']}")
    print(f"  γ_credit:            {p['gamma_credit']}")
    
    print(f"\nNatural Decay:")
    print(f"  h (attrition):       {p['h']} ({p['h']*12*100:.1f}% annual)")
    print(f"  g (growth):          {p['g']} ({p['g']*12*100:.1f}% annual)")
    
    print(f"\nSimulation:")
    print(f"  Paths:               {p['n_paths']:,}")
    print(f"  Horizon:             {p['n_months']} months ({p['n_months']//12} years)")
    print(f"  Initial Balance:     ${p['initial_balance']:,.0f}")
    
    print(f"\nHull-White:")
    print(f"  κ (mean reversion):  {p['kappa']}")
    
    print(f"\nOutput Directory:      {p['output_dir']}/")
    print("=" * 70)


# ============================================================================
# SENSITIVITY ANALYSIS GRIDS
# ============================================================================

# Default alpha values for sensitivity analysis
ALPHA_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Default MA window values for sensitivity analysis
MA_WINDOW_GRID = [6, 12, 18, 24, 36, 48, 60]

# Balance size sensitivity factors
BALANCE_SIZE_FACTORS = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00]

# Credit spread stress scenarios (bps)
CREDIT_SPREAD_SCENARIOS = {
    'base': 50,
    'mild_stress': 100,
    'moderate_stress': 150,
    'severe_stress': 250,
    'crisis': 400,
}

# Parameter sensitivity grids
H_SENSITIVITY = [0.005, 0.01, 0.015]              # Closure rate: 0.5%, 1%, 1.5%
BETA_RATE_SENSITIVITY = [0.20, 0.30, 0.40, 0.50]  # Rate sensitivity: 20%, 30%, 40%, 50%
BETA_CREDIT_SENSITIVITY = [0.00, 0.15, 0.30, 0.45]  # Credit sensitivity: 0%, 15%, 30%, 45%


# ============================================================================
# INITIALIZATION
# ============================================================================

# Ensure output directory exists on import
ensure_output_dir()

# Set random seed on import for reproducibility
set_random_seed()


if __name__ == '__main__':
    # Print parameters if run directly
    print("\nNMD Model Configuration")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_params()
    
    print("\n\nSensitivity Analysis Grids:")
    print(f"  Alpha grid:      {ALPHA_GRID}")
    print(f"  MA window grid:  {MA_WINDOW_GRID}")
    print(f"  Balance factors: {BALANCE_SIZE_FACTORS}")

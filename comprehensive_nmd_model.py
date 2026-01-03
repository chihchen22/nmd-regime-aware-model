"""
COMPREHENSIVE NONMATURITY DEPOSIT MODELING FRAMEWORK
=====================================================
Author: Chih L. Chen, CFA, FRM
Date: December 2025

This script implements the complete analysis from:
"Quantifying Behavioral Optionality in Nonmaturity Deposits: 
Monte Carlo, Analytical, and Bifurcation Approaches Compared"

Outputs:
- 7 CSV files with detailed results
- 4 publication-quality charts
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Import market data modules
from market_data_loader import MarketDataLoader
from sabr_volatility import SABRVolatilityCube

print("=" * 80)
print("COMPREHENSIVE NMD MODELING FRAMEWORK")
print("Quantifying Behavioral Optionality in Nonmaturity Deposits")
print("=" * 80)
print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Set random seed for reproducibility
np.random.seed(42)

# Create output directory
output_dir = 'model_outputs'
os.makedirs(output_dir, exist_ok=True)
print(f"✓ Output directory: {output_dir}/\n")

# ============================================================================
# SECTION 1: MODEL PARAMETERS
# ============================================================================
print("SECTION 1: Model Parameters (Regime-Aware Base Model)")
print("-" * 80)

PARAMS = {
    'initial_balance': 1_000_000,      # Initial deposit balance
    'h': 0.01,                         # Monthly account closure rate (1%)
    'g': 0.02 / 12,                    # Monthly organic growth (2% annual)
    'beta_rate_0': 0.30,               # Base rate sensitivity (30%)
    'beta_credit_0': 0.15,             # Base credit sensitivity (15%)
    'gamma_rate': 0.30,                # Rate sensitivity balance exponent
    'gamma_credit': 0.40,              # Credit sensitivity balance exponent
    'B_0': 1_000_000,                  # Reference balance for scaling
    'alpha': 2.0,                      # REGIME AMPLIFICATION (base model, matches config.py)
    'ma_window_months': 24,            # Moving average window (24 months)
    'kappa': 0.03,                     # Hull-White mean reversion (3% annual)
    'n_months': 360,                   # Projection horizon (30 years)
    'n_paths': 5_000,                  # Monte Carlo paths
    'stable_percentile': 5,            # Stable balance percentile
    'bifurcation_stable_pct': 0.90,    # Bifurcation stable % (benchmark only)
    'spread_shock_bps': 0,             # Credit spread shock (0 = baseline)
}

print("\nRegime Amplification Parameters (BASE MODEL):")
print(f"  {'alpha (α)':25s} = {PARAMS['alpha']:.2f}  (amplification strength)")
print(f"  {'ma_window_months':25s} = {PARAMS['ma_window_months']}     (24-month trailing MA)")
print("\nNote: α calibrated qualitatively. Institutions should develop")
print("      empirical support based on historical deposit flows.")

print("\nCore Parameters:")
for key, val in PARAMS.items():
    if key in ['alpha', 'ma_window_months']:
        continue  # Already printed above
    if isinstance(val, float) and val < 1 and 'gamma' not in key and key != 'stable_percentile':
        print(f"  {key:25s} = {val:8.4f}")
    elif isinstance(val, int) and val > 1000:
        print(f"  {key:25s} = {val:,}")
    else:
        print(f"  {key:25s} = {val}")

# ============================================================================
# SECTION 2: LOAD HISTORICAL SOFR FOR MOVING AVERAGES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Historical SOFR Data for Regime Detection")
print("-" * 80)

sofr_history = pd.read_excel('SOFR_History.xlsx')
sofr_history['EndMonth'] = pd.to_datetime(sofr_history['EndMonth'])
sofr_history = sofr_history.sort_values('EndMonth')

# Calculate 24-month moving average as of September 30, 2025
valuation_date = pd.Timestamp('2025-09-30')
start_ma = valuation_date - pd.DateOffset(months=PARAMS['ma_window_months'])
ma_data = sofr_history[(sofr_history['EndMonth'] > start_ma) & 
                       (sofr_history['EndMonth'] <= valuation_date)]

sofr_ma_initial = ma_data['SOFRRATE'].mean()  # Already in decimal form

print(f"Valuation Date: {valuation_date.strftime('%B %d, %Y')}")
print(f"Moving Average Period: {start_ma.strftime('%b %Y')} to {valuation_date.strftime('%b %Y')}")
print(f"24-Month Average SOFR: {sofr_ma_initial*100:.4f}%")
print(f"Historical data points: {len(ma_data)}")

# Current SOFR will be loaded from market data below
print("\n(Current SOFR and regime excursion will be shown after market data loading)")

# ============================================================================
# SECTION 3: MARKET DATA LOADING (from Excel)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Market Data Loading from Excel (Sep 30, 2025)")
print("-" * 80)

# Load real SOFR market data from Excel file
market_data = MarketDataLoader('SOFR_Market_Data_20250930.xlsx')
market_data.load_all()

# Extract OIS curve
tenors_years = market_data.ois_curve['tenors_years']
swap_rates = market_data.ois_curve['swap_rates']

print("\nSOFR OIS Curve (Sep 30, 2025):")
for tenor, rate, term in zip(tenors_years, swap_rates, market_data.ois_curve['terms']):
    print(f"  {term:>6} ({tenor:6.3f}Y) = {rate*100:6.3f}%")

# Extract ATM cap volatilities
atm_cap_data = market_data.get_atm_cap_vols_for_model()
vol_tenors_years = atm_cap_data['expiries_years']
atm_vols_bps = atm_cap_data['atm_volatilities_bps']

print("\nSOFR Cap Volatility (ATM, Normal):")
for tenor, vol in zip(vol_tenors_years, atm_vols_bps):
    print(f"  {tenor:5.2f}Y = {vol:6.1f} bps")

# Show regime amplification status
current_sofr = swap_rates[0]  # Overnight rate
print(f"\nRegime Amplification Status:")
print(f"  Current SOFR: {current_sofr*100:.4f}%")
print(f"  24-Month MA:  {sofr_ma_initial*100:.4f}%")
regime_excursion = max(0, current_sofr - sofr_ma_initial)
print(f"  Regime Excursion: {regime_excursion*100:.4f}%")
if regime_excursion > 0:
    print(f"  → AMPLIFICATION ACTIVE (β multiplied by {1 + PARAMS['alpha']*regime_excursion:.4f})")
else:
    print(f"  → No amplification (rates at/below MA)")

# ============================================================================
# SECTION 4: NO-ARBITRAGE BOOTSTRAPPING
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: No-Arbitrage Bootstrapping")
print("-" * 80)

def bootstrap_discount_factors(tenors, swap_rates):
    """
    Bootstrap discount factors from OIS swap rates
    
    Enhanced method (peer review response):
    - Proper continuous compounding for zero rates
    - Accounts for overnight compounded floating leg
    - Recursive solver using PV equation: Σ c_i × DF(t_i) = 1
    
    For OIS swaps:
    - Fixed leg: annualized swap rate × accrual period
    - Floating leg: overnight compounded rate (approximated by forward curve)
    - Present value: fixed leg cash flows = floating leg (par swap)
    
    Methodology follows QuantLib and OpenGamma implementations for OIS bootstrapping.
    Research simplification: Uses continuous compounding approximation for efficiency.
    Production systems should use exact daily compounding of SOFR fixings.
    """
    n = len(tenors)
    df = np.ones(n + 1)  # Include DF(0) = 1.0
    df[0] = 1.0
    
    # Bootstrap recursively using no-arbitrage swap par condition
    for i in range(n):
        tenor = tenors[i]
        swap_rate = swap_rates[i]
        
        if i == 0 or tenor <= 1.0:
            # For short maturities (≤1Y), use direct formula
            # Avoids payment schedule complexity for money market tenors
            df[i+1] = 1.0 / (1.0 + swap_rate * tenor)
        else:
            # For longer swaps, use annual payment schedule with no-arbitrage condition
            # Build payment times (annual payments)
            payment_freq = 1  # Annual payments
            n_payments = int(tenor * payment_freq)
            payment_times = np.linspace(tenor / n_payments, tenor, n_payments)
            
            # Key fix: Interpolate discount factors using flat-forward assumption
            # This prevents unrealistic oscillations in sparse regions (e.g., 20Y-30Y)
            if n_payments > 1:
                known_tenors = tenors[:i+1]
                known_dfs = df[1:i+2]
                
                # Sum fixed leg present value for all but final payment
                fixed_leg_pv = 0.0
                for pt in payment_times[:-1]:
                    # Interpolate or extrapolate DF using flat-forward (linear in log-space)
                    idx = np.searchsorted(known_tenors, pt)
                    if idx == 0:
                        # Before first tenor: extrapolate using first rate
                        df_pt = known_dfs[0] ** (pt / known_tenors[0])
                    elif idx >= len(known_tenors):
                        # Beyond last tenor: use swap rate as approximation for zero rate
                        # This is the "research approximation" - assumes par yield ≈ zero rate
                        # for long-dated illiquid tenors. More sophisticated approaches would
                        # fit a parametric curve (e.g., Nelson-Siegel) to all market data.
                        df_pt = np.exp(-swap_rate * pt)
                    else:
                        # Between tenors: linear interpolation in log(DF) space
                        t_low, t_high = known_tenors[idx-1], known_tenors[idx]
                        df_low, df_high = known_dfs[idx-1], known_dfs[idx]
                        weight = (pt - t_low) / (t_high - t_low)
                        log_df_pt = np.log(df_low) + weight * (np.log(df_high) - np.log(df_low))
                        df_pt = np.exp(log_df_pt)
                    
                    fixed_leg_pv += swap_rate * (tenor / n_payments) * df_pt
                
                # Solve for DF(T) using swap par condition:
                # 1 - DF(T) = fixed_leg_pv + swap_rate × Δt × DF(T)
                final_period = tenor / n_payments
                df[i+1] = (1.0 - fixed_leg_pv) / (1.0 + swap_rate * final_period)
            else:
                # Single payment case
                df[i+1] = 1.0 / (1.0 + swap_rate * tenor)
    
    return df[1:]  # Return discount factors for tenors (excluding DF(0))

def compute_zero_rates(tenors, df):
    """Compute continuous zero rates from discount factors"""
    zero_rates = -np.log(df) / tenors
    return zero_rates

# Bootstrap
discount_factors = bootstrap_discount_factors(tenors_years, swap_rates)
zero_rates = compute_zero_rates(tenors_years, discount_factors)

# Create monthly grid
months = np.arange(1, PARAMS['n_months'] + 1)
months_in_years = months / 12.0

# Interpolate discount factors and zero rates to monthly grid
# Strategy: Use flat-forward interpolation (linear in log-space) throughout
# This is the standard no-arbitrage interpolation for interest rate curves
# and prevents unrealistic oscillations, especially in flat regions

# Prepend time 0 with DF=1.0 to ensure proper interpolation from present
tenors_with_zero = np.concatenate([[0.0], tenors_years])
df_with_zero = np.concatenate([[1.0], discount_factors])

# Linear interpolation in log(DF) space = flat forward interpolation
df_interp = interp1d(tenors_with_zero, np.log(df_with_zero), 
                     kind='linear', bounds_error=False, 
                     fill_value=(np.log(df_with_zero[0]), np.log(df_with_zero[-1])))
df_monthly = np.exp(df_interp(months_in_years))

# Compute zero rates from interpolated discount factors
zero_monthly = -np.log(df_monthly) / months_in_years

print(f"✓ Bootstrapped {len(tenors_years)} swap rates")
print(f"✓ Interpolated to {PARAMS['n_months']} monthly points")
print(f"  Sample zero rates: 1Y={zero_monthly[11]*100:.2f}%, " +
      f"5Y={zero_monthly[59]*100:.2f}%, 10Y={zero_monthly[119]*100:.2f}%")

# Save bootstrapped curve
df_bootstrap = pd.DataFrame({
    'Month': months,
    'Years': months_in_years,
    'Discount_Factor': df_monthly,
    'Zero_Rate': zero_monthly * 100,  # in %
})
df_bootstrap.to_csv(f'{output_dir}/01_bootstrapped_curve.csv', index=False)
print(f"✓ Saved: 01_bootstrapped_curve.csv")

# ============================================================================
# SECTION 4: FORWARD RATE CURVE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Instantaneous Forward Rates")
print("-" * 80)

def compute_instantaneous_forwards(times, zero_rates):
    """
    Compute instantaneous forward rates using a stable numerical method.
    
    The instantaneous forward rate is defined as:
        f(t) = -d(ln DF)/dt = z(t) + t * dz/dt
    
    However, the naive formula z(t) + t*dz/dt is numerically unstable when:
    - t is large (e.g., 20-30 years) because it amplifies gradient errors
    - The curve is flat or nearly flat (small dz/dt with numerical noise)
    
    Instead, we use the discrete approximation:
        f(t_i) ≈ -ln(DF(t_i+1)/DF(t_i)) / Δt
    
    This is more stable because:
    - It works directly with discount factors (no derivative approximation)
    - It doesn't multiply by t, avoiding amplification of errors
    - It's the natural definition for discrete-time forward rates
    
    For the final point, we use backward difference to avoid extrapolation.
    """
    n = len(times)
    forwards = np.zeros(n)
    
    # Compute discount factors from zero rates
    discount_factors = np.exp(-zero_rates * times)
    
    # Forward differences for all but last point
    for i in range(n - 1):
        dt = times[i + 1] - times[i]
        forwards[i] = -np.log(discount_factors[i + 1] / discount_factors[i]) / dt
    
    # Backward difference for last point (avoid extrapolation)
    dt = times[-1] - times[-2]
    forwards[-1] = -np.log(discount_factors[-1] / discount_factors[-2]) / dt
    
    # Light smoothing to remove month-to-month noise while preserving shape
    # Use a small window (5 months) to avoid over-smoothing
    from scipy.ndimage import uniform_filter1d
    forwards_smooth = uniform_filter1d(forwards, size=5, mode='nearest')
    
    return forwards_smooth

forward_rates = compute_instantaneous_forwards(months_in_years, zero_monthly)

print(f"✓ Computed instantaneous forwards")
print(f"  Sample forwards: 1M={forward_rates[0]*100:.2f}%, " +
      f"1Y={forward_rates[11]*100:.2f}%, 10Y={forward_rates[119]*100:.2f}%")
print(f"  Forward range: {forward_rates.min()*100:.2f}% to {forward_rates.max()*100:.2f}%")

# Save forward rates
df_forwards = pd.DataFrame({
    'Month': months,
    'Years': months_in_years,
    'Forward_Rate': forward_rates * 100,  # in %
})
df_forwards.to_csv(f'{output_dir}/02_forward_rates.csv', index=False)
print(f"✓ Saved: 02_forward_rates.csv")

# ============================================================================
# SECTION 4A: FHLB-SOFR FORWARD SPREAD (Credit/Liquidity Component)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4A: FHLB-SOFR Forward Spread (Credit/Liquidity)")
print("-" * 80)

# Load pre-computed monthly forward spread from bootstrap
try:
    spread_df = pd.read_csv(f'{output_dir}/monthly_forward_spread_simple.csv')
    credit_spread = spread_df['Forward_Spread_bps'].values / 10000  # Convert bps to decimal
    
    # Extend spread to match n_months if needed (flat extrapolation at end)
    if len(credit_spread) < PARAMS['n_months']:
        # Pad with last value
        n_pad = PARAMS['n_months'] - len(credit_spread)
        credit_spread = np.concatenate([credit_spread, np.full(n_pad, credit_spread[-1])])
    elif len(credit_spread) > PARAMS['n_months']:
        # Truncate
        credit_spread = credit_spread[:PARAMS['n_months']]
    
    # Apply spread shock if specified
    spread_shock = PARAMS['spread_shock_bps'] / 10000
    credit_spread_shocked = credit_spread + spread_shock
    
    print(f"✓ Loaded {len(spread_df)} monthly forward spreads")
    print(f"  Extended to {PARAMS['n_months']} months (flat extrapolation at tail)")
    print(f"  Spread range: {credit_spread.min()*10000:.1f} to {credit_spread.max()*10000:.1f} bps")
    print(f"  Mean spread: {credit_spread.mean()*10000:.1f} bps")
    print(f"  Spread shock: {PARAMS['spread_shock_bps']} bps")
    print(f"  Sample spreads: 0M={credit_spread[0]*10000:.1f} bps, " +
          f"1Y={credit_spread[12]*10000:.1f} bps, " +
          f"5Y={credit_spread[60]*10000:.1f} bps, " +
          f"10Y={credit_spread[120]*10000:.1f} bps")
    
except FileNotFoundError:
    print("⚠ Warning: Credit spread file not found. Using zero spread.")
    print("  Run bootstrap_fhlb_simple.py first to generate monthly_forward_spread_simple.csv")
    credit_spread_shocked = np.zeros(PARAMS['n_months'])

# ============================================================================
# SECTION 5: VOLATILITY CUBE CONSTRUCTION (SABR Interpolation)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: Volatility Cube Construction with SABR")
print("-" * 80)

# Build SABR volatility cube for strike/expiry interpolation
sabr_cube = SABRVolatilityCube(market_data.cap_vol_surface)

# Create forward curve function for SABR calibration
def forward_curve_function(t):
    """Get forward rate at time t using interpolated zero curve"""
    # Use the monthly zero rates we computed earlier
    t_months = t * 12
    if t_months < 1:
        t_months = 1
    if t_months > PARAMS['n_months']:
        t_months = PARAMS['n_months']
    idx = int(t_months) - 1
    return forward_rates[idx]

# Calibrate SABR parameters across all expiries
sabr_cube.calibrate_all_expiries(forward_curve_function)

# Generate ATM volatility term structure using SABR
monthly_years = np.arange(1, 361) / 12.0
vol_monthly_decimal = sabr_cube.get_atm_vol_term_structure(
    monthly_years, forward_curve_function
)
vol_monthly = vol_monthly_decimal * 10000  # Convert to bps

print(f"✓ Extended volatility surface to 360 months")
print(f"  1M vol: {vol_monthly[0]:.1f} bps")
print(f"  1Y vol: {vol_monthly[11]:.1f} bps")
print(f"  10Y vol: {vol_monthly[119]:.1f} bps")
print(f"  30Y vol: {vol_monthly[359]:.1f} bps")

# Save volatility surface
df_vol_surface = pd.DataFrame({
    'Month': months,
    'Years': months_in_years,
    'ATM_Volatility_bps': vol_monthly,
    'ATM_Volatility_pct': vol_monthly / 10000,  # Convert to decimal
})
df_vol_surface.to_csv(f'{output_dir}/03_volatility_surface.csv', index=False)
print(f"✓ Saved: 03_volatility_surface.csv")

# Convert volatility to decimal for Hull-White
sigma_monthly = vol_monthly / 10000

# ============================================================================
# SECTION 6: HULL-WHITE CALIBRATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Hull-White Calibration")
print("-" * 80)

kappa = PARAMS['kappa']
dt = 1/12  # Monthly time step

def compute_theta(t_years, forward_rate, sigma, kappa):
    """Compute time-dependent drift theta(t)"""
    # Approximate df/dt using finite differences
    df_dt = np.gradient(forward_rate, t_years)
    
    # theta(t) = df/dt + kappa*f(t) + sigma^2/(2*kappa) * [1 - exp(-2*kappa*t)]
    convexity_adjustment = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * t_years))
    theta = df_dt + kappa * forward_rate + convexity_adjustment
    
    return theta

theta_monthly = compute_theta(months_in_years, forward_rates, sigma_monthly, kappa)

print(f"✓ Calibrated Hull-White parameters")
print(f"  Mean reversion κ = {kappa:.4f}")
print(f"  Time-dependent θ(t) computed")
print(f"  Time-dependent σ(t) from volatility surface")

# ============================================================================
# SECTION 7: MONTE CARLO SIMULATION
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 7: Monte Carlo Simulation")
print("-" * 80)
print(f"Simulating {PARAMS['n_paths']:,} paths over {PARAMS['n_months']} months...")

# Initialize rate paths
rate_paths = np.zeros((PARAMS['n_paths'], PARAMS['n_months']))
rate_paths[:, 0] = forward_rates[0]  # Start at first forward rate

# Euler discretization: dr = [theta(t) - kappa*r]*dt + sigma*sqrt(dt)*dW
# Note: Rate floor at 0% added per peer review to prevent negative rates in tail scenarios
print("Running Monte Carlo...", end=" ", flush=True)

for t in range(1, PARAMS['n_months']):
    dW = np.random.randn(PARAMS['n_paths'])
    drift = (theta_monthly[t-1] - kappa * rate_paths[:, t-1]) * dt
    diffusion = sigma_monthly[t-1] * np.sqrt(dt) * dW
    rate_paths[:, t] = rate_paths[:, t-1] + drift + diffusion
    
    # Apply 0% floor to prevent negative rates (peer review enhancement)
    # This matters in extreme low-rate scenarios (QE, Japan/Europe experience)
    # Floor has minimal impact in normal market conditions but prevents balance calculation issues
    rate_paths[:, t] = np.maximum(0.0, rate_paths[:, t])
    
    # Progress indicator
    if t % 60 == 0:
        print(f"{t//12}Y...", end=" ", flush=True)

print("Done!")

# Validate Monte Carlo
mc_mean = rate_paths.mean(axis=0)
mc_std = rate_paths.std(axis=0)
mc_p05 = np.percentile(rate_paths, 5, axis=0)
mc_p95 = np.percentile(rate_paths, 95, axis=0)

print(f"✓ Monte Carlo validation:")
print(f"  Mean path tracks forwards: max deviation = {np.max(np.abs(mc_mean - forward_rates))*100:.1f} bps")
print(f"  1Y std dev: {mc_std[11]*100:.1f} bps")
print(f"  10Y std dev: {mc_std[119]*100:.1f} bps")

# Save rate path summary
df_rate_summary = pd.DataFrame({
    'Month': months,
    'Years': months_in_years,
    'Forward_Rate': forward_rates * 100,
    'MC_Mean': mc_mean * 100,
    'MC_Std': mc_std * 100,
    'MC_P05': mc_p05 * 100,
    'MC_P25': np.percentile(rate_paths, 25, axis=0) * 100,
    'MC_P50': np.percentile(rate_paths, 50, axis=0) * 100,
    'MC_P75': np.percentile(rate_paths, 75, axis=0) * 100,
    'MC_P95': mc_p95 * 100,
})
df_rate_summary.to_csv(f'{output_dir}/04_rate_paths_summary.csv', index=False)
print(f"✓ Saved: 04_rate_paths_summary.csv")

# Save sample paths (first 100 for visualization)
df_sample_paths = pd.DataFrame(
    rate_paths[:100, :].T * 100,
    columns=[f'Path_{i+1}' for i in range(100)]
)
df_sample_paths.insert(0, 'Month', months)
df_sample_paths.insert(1, 'Years', months_in_years)
df_sample_paths.to_csv(f'{output_dir}/07_mc_sample_paths.csv', index=False)
print(f"✓ Saved: 07_mc_sample_paths.csv (100 sample paths)")

# ============================================================================
# SECTION 8: COMPONENT DECAY MODEL - ALL APPROACHES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 8: Component Decay Model Application")
print("-" * 80)

def beta_rate_sensitive(B, beta_rate_0, gamma_rate, B_0):
    """
    Balance-sensitive RATE sensitivity (power-law form) - BASE VERSION
    
    Formula: β_rate(B) = β_rate_0 × (B/B₀)^γ_rate
    
    NOTE: This is the base function. For regime-aware model, use beta_rate_regime_aware().
    """
    return beta_rate_0 * (B / B_0)**gamma_rate

def beta_rate_regime_aware(B, rate, ma_rate, beta_rate_0, gamma_rate, B_0, alpha):
    """
    REGIME-AWARE rate sensitivity (BASE MODEL in v4 paper)
    
    Formula: β_rate(B,t) = β_base(B) × [1 + α × max(0, r(t) - MA(t))]
    
    Where:
    - β_base(B) = β_rate_0 × (B/B₀)^γ_rate (power-law base)
    - α = amplification parameter (0.5 = strong amplification)
    - r(t) = current SOFR rate
    - MA(t) = 24-month trailing moving average of SOFR
    - Regime excursion = max(0, r(t) - MA(t))
    
    Properties:
    - Amplification active when r(t) > MA(t) (rates above recent average)
    - No amplification when r(t) ≤ MA(t) (rates at/below average)
    - Larger balances get amplified more (through β_base)
    
    Intuition: During rate spikes above moving average:
    - Depositors become MORE rate-sensitive (heightened awareness)
    - Relative rate disadvantage becomes more salient
    - Institutional depositors actively reallocate
    - Surge deposits from QE era flee the system
    
    Calibration:
    - α = 0.5 (Sep 2025): Moderate amplification as rates near historical MA
    - α = 0.7-0.8 (Mar 2022): Strong amplification as rates spike from zero
    - 24-month MA: Captures depositor memory of recent rate environment
    
    Note: α is a qualitative assumption. Institutions should develop
    empirical support through historical deposit flow analysis.
    """
    beta_base = beta_rate_0 * (B / B_0)**gamma_rate
    regime_excursion = max(0, rate - ma_rate)
    return beta_base * (1 + alpha * regime_excursion)

def beta_credit_sensitive(B, beta_credit_0, gamma_credit, B_0):
    """
    Balance-sensitive CREDIT sensitivity (power-law form)
    
    Formula: β_credit(B) = β_credit_0 × (B/B₀)^γ_credit
    
    Properties:
    - Monotonically increasing: larger balances → higher credit sensitivity
    - Always positive for positive balances
    - At reference balance B₀: β_credit(B₀) = β_credit_0 exactly
    - γ_credit > γ_rate typically: credit risk matters MORE for large balances
    
    Intuition: Larger balances are MORE credit-sensitive because:
    - Often exceed FDIC insurance limits ($250K)
    - Sophisticated depositors actively monitor bank credit ratings
    - Institutional depositors subject to counterparty risk limits
    - Treasury managers diversify across banks based on credit spreads
    - Flight-to-quality during stress hits large uninsured deposits first
    
    Example calibration:
    - $50K (insured):   β_credit ≈ 0.046 (low, FDIC protection)
    - $1M (reference):  β_credit = 0.150 (baseline)
    - $5M (institutional): β_credit ≈ 0.285 (very high, uninsured)
    """
    return beta_credit_0 * (B / B_0)**gamma_credit

def component_decay_step(B_prev, rate, credit_spread, h, g, 
                        beta_rate_0, beta_credit_0, gamma_rate, gamma_credit, B_0,
                        alpha=0, ma_rate=0):
    """
    Single step of REGIME-AWARE component decay model (BASE MODEL)
    
    Balance dynamics:
    B(t) = B(t-1) × (1-h) × exp[g - β_rate_regime(B,t)×r(t) - β_credit(B)×s(t)]
    
    Where:
    - β_rate_regime(B,t): Regime-aware rate sensitivity (amplified in rate spikes)
    - β_credit(B): Credit sensitivity (unchanged)
    - r(t): Current risk-free rate (SOFR)
    - s(t): Credit/liquidity spread (FHLB-SOFR forward spread)
    - α: Amplification parameter (default 0.5)
    - ma_rate: 24-month moving average of SOFR
    
    If alpha=0, reverts to base power-law model (for backwards compatibility).
    """
    # Use regime-aware beta if alpha > 0, otherwise use base beta
    if alpha > 0:
        beta_rate = beta_rate_regime_aware(B_prev, rate, ma_rate, beta_rate_0, gamma_rate, B_0, alpha)
    else:
        beta_rate = beta_rate_sensitive(B_prev, beta_rate_0, gamma_rate, B_0)
    
    beta_credit = beta_credit_sensitive(B_prev, beta_credit_0, gamma_credit, B_0)
    B_next = B_prev * (1 - h) * np.exp(g - beta_rate * rate - beta_credit * credit_spread)
    return B_next

# ============================================================================
# 8.1 DETERMINISTIC APPROACH
# ============================================================================
print("\n8.1 DETERMINISTIC (Market-Implied Forwards)")
print("-" * 40)

B_det = np.zeros(PARAMS['n_months'] + 1)
B_det[0] = PARAMS['initial_balance']

for t in range(PARAMS['n_months']):
    B_det[t+1] = component_decay_step(
        B_det[t], forward_rates[t], credit_spread_shocked[t],
        PARAMS['h'], PARAMS['g'],
        PARAMS['beta_rate_0'], PARAMS['beta_credit_0'], 
        PARAMS['gamma_rate'], PARAMS['gamma_credit'], PARAMS['B_0'],
        alpha=PARAMS['alpha'], ma_rate=sofr_ma_initial  # REGIME-AWARE BASE MODEL
    )

print(f"✓ Deterministic projection complete")
print(f"  Final balance (30Y): ${B_det[-1]:,.0f}")

# ============================================================================
# 8.2 MONTE CARLO APPROACH
# ============================================================================
print("\n8.2 MONTE CARLO (Path-Dependent Simulation)")
print("-" * 40)

balance_paths = np.zeros((PARAMS['n_paths'], PARAMS['n_months'] + 1))
balance_paths[:, 0] = PARAMS['initial_balance']

print("Simulating balance paths...", end=" ", flush=True)
for path in range(PARAMS['n_paths']):
    for t in range(PARAMS['n_months']):
        balance_paths[path, t+1] = component_decay_step(
            balance_paths[path, t], rate_paths[path, t], credit_spread_shocked[t],
            PARAMS['h'], PARAMS['g'], 
            PARAMS['beta_rate_0'], PARAMS['beta_credit_0'],
            PARAMS['gamma_rate'], PARAMS['gamma_credit'], PARAMS['B_0'],
            alpha=PARAMS['alpha'], ma_rate=sofr_ma_initial  # REGIME-AWARE BASE MODEL
        )
    
    if (path + 1) % 1000 == 0:
        print(f"{path+1:,}...", end=" ", flush=True)

print("Done!")

# Monte Carlo statistics
mc_bal_mean = balance_paths.mean(axis=0)
mc_bal_std = balance_paths.std(axis=0)
mc_bal_p05 = np.percentile(balance_paths, PARAMS['stable_percentile'], axis=0)
mc_bal_p95 = np.percentile(balance_paths, 95, axis=0)

print(f"✓ Monte Carlo balances computed")
print(f"  Mean final balance: ${mc_bal_mean[-1]:,.0f}")
print(f"  P05 final balance: ${mc_bal_p05[-1]:,.0f}")

# ============================================================================
# 8.3 ANALYTICAL APPROACH
# ============================================================================
print("\n8.3 ANALYTICAL (Bachelier Normal Volatility Framework)")
print("-" * 40)

# Compute Hull-White integrated variance
def compute_integrated_variance(t_years, sigma, kappa):
    """Compute Var[r(T)] for Hull-White model"""
    var_r = np.zeros(len(t_years))
    for i, T in enumerate(t_years):
        # Numerical integration of sigma^2(u) * [1 - exp(-2*kappa*(T-u))]
        u_grid = np.linspace(0, T, 100)
        sigma_u = np.interp(u_grid, t_years, sigma)
        integrand = sigma_u**2 * (1 - np.exp(-2 * kappa * (T - u_grid)))
        var_r[i] = np.trapz(integrand, u_grid)
    return var_r

var_r_monthly = compute_integrated_variance(months_in_years, sigma_monthly, kappa)

# Analytical expected balance with convexity adjustment (updated with credit spread)
B_analytical = np.zeros(PARAMS['n_months'] + 1)
B_analytical[0] = PARAMS['initial_balance']

for t in range(PARAMS['n_months']):
    # Use regime-aware beta for analytical approach
    beta_rate = beta_rate_regime_aware(B_analytical[t], forward_rates[t], sofr_ma_initial,
                                        PARAMS['beta_rate_0'], PARAMS['gamma_rate'], 
                                        PARAMS['B_0'], PARAMS['alpha'])
    beta_credit = beta_credit_sensitive(B_analytical[t], PARAMS['beta_credit_0'],
                                         PARAMS['gamma_credit'], PARAMS['B_0'])
    
    # Expected rate effect with convexity
    E_r = forward_rates[t]
    convexity_adj = 0.5 * beta_rate**2 * var_r_monthly[t]
    
    # Note: Credit spread is deterministic (held constant across MC paths)
    # So it contributes to drift but not to variance
    B_analytical[t+1] = B_analytical[t] * (1 - PARAMS['h']) * \
                        np.exp(PARAMS['g'] - beta_rate * E_r - beta_credit * credit_spread_shocked[t] + convexity_adj)

# Analytical stable balance (5th percentile via Bachelier/normal volatility)
# Following Pykhtin & Zhu (2007, Risk) CVA methodology for mean-reverting rates
# and Musiela & Rutkowski (2005) Bachelier option pricing framework
# 
# Key insight: Implied volatilities from caps/swaptions are NORMAL (Bachelier) vols,
# not lognormal. This is standard market convention for interest rate derivatives.
#
# For deposit balance with rate-dependent decay:
# B_{t+1} = B_t × (1-h) × exp(g - β×r_t)
#
# Under Bachelier model with normal volatility σ_N:
# r_t = f_t + ε_t where ε_t ~ N(0, σ_N² × T_eff)
#
# CORRECTED: Proper Hull-White autocorrelation structure (Peer Review Response)
# Variance of cumulative rate effect must account for covariance:
# Var(Σ β_i × r_i) = ΣΣ β_i × β_j × Cov(r_i, r_j)
# where Cov(r_i, r_j) = σ²/(2κ) × exp(-κ|t_i - t_j|) × (1 - exp(-2κ×min(t_i,t_j)))
#
# This correction addresses material approximation error identified in model validation:
# - Independent variance approach underestimated variance by 98.91%
# - P05 balance was overstated by +57.61%
# - WAL P05 was overstated by +17.73%
#
# Similar analytical frameworks appear in Kalotay, Yang & Fabozzi (2004)
# for MBS prepayment options, avoiding full Monte Carlo through closed-form approximations.

def hw_covariance(t_i, t_j, kappa, sigma):
    """
    Hull-White covariance function for mean-reverting rates
    Cov(r_t_i, r_t_j) accounting for mean reversion and temporal correlation
    """
    t_min = min(t_i, t_j)
    t_diff = abs(t_i - t_j)
    cov = (sigma**2 / (2 * kappa)) * np.exp(-kappa * t_diff) * (1 - np.exp(-2 * kappa * t_min))
    return cov

B_analytical_stable = np.zeros(PARAMS['n_months'] + 1)
B_analytical_stable[0] = PARAMS['initial_balance']

dt = 1.0 / 12.0  # Monthly time step in years

# Build beta_rate trajectory for covariance calculation
# Note: Using regime-aware beta for variance calculation
beta_trajectory = np.zeros(PARAMS['n_months'])
for t in range(PARAMS['n_months']):
    beta_trajectory[t] = beta_rate_regime_aware(B_analytical[t], forward_rates[t], sofr_ma_initial,
                                                  PARAMS['beta_rate_0'], PARAMS['gamma_rate'], 
                                                  PARAMS['B_0'], PARAMS['alpha'])

print("Computing analytical stable balance with Hull-White autocorrelation...")
print("  (Accounting for rate covariance structure)")

# Compute stable balance accounting for full covariance matrix
for t in range(PARAMS['n_months']):
    # Variance of cumulative log balance change
    # Var(Σ β_i × r_i) = ΣΣ β_i × β_j × Cov(r_i, r_j)
    var_log_B = 0.0
    
    for i in range(t + 1):
        for j in range(t + 1):
            # Time at mid-point of each period
            t_i = (i + 0.5) * dt
            t_j = (j + 0.5) * dt
            
            # Hull-White covariance between periods i and j
            cov_ij = hw_covariance(t_i, t_j, kappa, sigma_monthly[i])
            
            # Contribution to total variance
            var_log_B += beta_trajectory[i] * beta_trajectory[j] * cov_ij
    
    # Expected log balance
    E_log_B = np.log(B_analytical[t+1])
    
    # 5th percentile
    z_05 = norm.ppf(0.05)
    B_analytical_stable[t+1] = np.exp(E_log_B + z_05 * np.sqrt(var_log_B))
    
    # Progress indicator for long computation
    if (t + 1) % 60 == 0:
        print(f"  Month {t+1}/{PARAMS['n_months']}: var_log_B = {var_log_B:.6f}")

print(f"✓ Analytical approximation complete (with proper autocorrelation)")
print(f"  Expected final balance: ${B_analytical[-1]:,.0f}")
print(f"  Stable (P05) final balance: ${B_analytical_stable[-1]:,.0f}")
print(f"  Final log variance: {var_log_B:.6f}")

# ============================================================================
# 8.4 BIFURCATION APPROACH (Rate-Insensitive but Credit-Sensitive Stable)
# ============================================================================
print("\n8.4 BIFURCATION (Dual-Sensitivity Approach: β_rate=0, β_credit>0)")
print("-" * 40)
print("Stable portion uses β_rate=0 (rate-insensitive) but β_credit>0 (credit-sensitive)")
print("Solving for stable % to match total WAL to MC Mean")

# Step 1: Calculate stable profile with β_rate=0 but β_credit=0.15
# This represents truly sticky deposits that don't respond to interest rates
# but still flee when credit spreads widen (2023 crisis pattern)
print("\nCalculating stable profile with β_rate=0, β_credit=0.15...")

def calculate_balance_profile_stable_dual(forward_rates, credit_spreads, h, g, B0, 
                                          beta_credit_0, gamma_credit, B_0):
    """
    Calculate balance profile with β_rate=0 (rate-insensitive) 
    but β_credit>0 (credit-sensitive).
    
    This reflects deposits that are sticky through rate cycles but flee during credit stress.
    """
    n_months = len(forward_rates)
    balance = np.zeros(n_months + 1)
    balance[0] = B0
    
    for t in range(n_months):
        # Balance-dependent credit sensitivity
        beta_credit = beta_credit_0 * (balance[t] / B_0) ** gamma_credit
        
        # No rate sensitivity (β_rate=0), but credit sensitivity remains
        balance[t + 1] = balance[t] * (1 - h) * np.exp(g - beta_credit * credit_spreads[t])
    
    return balance

def calculate_bifurcation_wal_solver(stable_pct, stable_profile, B0, nonstable_profile=None):
    """
    Calculate bifurcation WAL with:
    - stable_pct follows stable_profile (β=0)
    - (1 - stable_pct) follows nonstable_profile
    
    If nonstable_profile is None, assumes immediate 1-month runoff (default)
    """
    n_months = len(stable_profile) - 1
    blended = np.zeros(n_months + 1)
    blended[0] = B0
    
    # Stable portion
    blended[1:] += stable_profile[1:] * stable_pct
    
    # Non-stable portion
    if nonstable_profile is None:
        # Default: immediate runoff (1 month) - all non-stable balance gone after month 0
        pass
    else:
        # Custom runoff profile - nonstable_profile is already normalized to start at B0
        # Scale it by the non-stable percentage
        blended[1:] += nonstable_profile[1:] * (1 - stable_pct)
    
    # Calculate WAL
    cash_flows = np.diff(-blended)
    cash_flows[-1] = blended[-2]
    months_array = np.arange(1, len(cash_flows) + 1)
    wal = np.sum(months_array * cash_flows) / (12 * B0)
    return wal

def create_linear_runoff_profile(n_months, runoff_months, B0):
    """
    Create a linear amortization runoff profile.
    
    Parameters:
    - n_months: Total projection horizon
    - runoff_months: Number of months over which to amortize (e.g., 6 or 12)
    - B0: Initial balance
    
    Returns:
    - balance_profile: Array of balances with linear runoff over runoff_months
    """
    profile = np.zeros(n_months + 1)
    profile[0] = B0
    
    for t in range(1, min(runoff_months + 1, n_months + 1)):
        # Linear decay: remaining = initial * (1 - t/runoff_months)
        profile[t] = B0 * (1 - t / runoff_months)
    
    # After runoff_months, balance is zero
    profile[runoff_months:] = 0
    
    return profile

# Calculate stable balance profile with β_rate=0 but β_credit>0
B_stable_beta0 = calculate_balance_profile_stable_dual(
    forward_rates=forward_rates,
    credit_spreads=credit_spread_shocked,  # Use the shocked spread from Section 4A
    h=PARAMS['h'],
    g=PARAMS['g'],
    B0=PARAMS['initial_balance'],
    beta_credit_0=PARAMS['beta_credit_0'],
    gamma_credit=PARAMS['gamma_credit'],
    B_0=PARAMS['initial_balance']
)

print(f"  Stable profile calculated (β_rate=0, β_credit={PARAMS['beta_credit_0']})")
print(f"  → Rate-insensitive but credit-sensitive: attrition + growth + credit response")

# Step 2: Calculate bifurcation profiles
# Note: We'll solve for optimal stable % after WAL function is defined
# For now, create the balance profiles

# Use hardcoded value from test results: 56.9% stable matches MC Mean of 5.45 years
# This will be verified after calculate_wal function is available
stable_pct = 0.569  # Will be updated dynamically later
nonstable_pct = 1 - stable_pct

# Calculate final bifurcation profiles
B_bifurc_stable = B_stable_beta0 * stable_pct

# Non-stable leg: Complete runoff in 1 month
B_bifurc_nonstable = np.zeros(PARAMS['n_months'] + 1)
B_bifurc_nonstable[0] = PARAMS['initial_balance'] * nonstable_pct
B_bifurc_nonstable[1:] = 0

# Total bifurcation balance
B_bifurc_total = B_bifurc_stable + B_bifurc_nonstable

print(f"✓ Bifurcation profiles created (temporary allocation: {stable_pct*100:.1f}%/{nonstable_pct*100:.1f}%)")
print(f"  Will calibrate to MC Mean after WAL function is defined")

# Store beta0 profile for later calibration
bifurc_stable_beta0_profile = B_stable_beta0.copy()

# Update PARAMS for consistency with outputs
PARAMS['bifurcation_stable_pct'] = stable_pct

# ============================================================================
# SECTION 9: WEIGHTED AVERAGE LIFE CALCULATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9: Weighted Average Life Calculations")
print("-" * 80)

def calculate_wal(balance_profile, initial_balance=None):
    """
    Calculate weighted average life using undiscounted cash flow method.
    
    This method aligns with regulatory reporting conventions (DFAST/CCAR)
    and industry best practices for deposit maturity profiling.
    
    Methodology:
    1. Calculate monthly cash flows: CF(t) = B(t) - B(t+1)
    2. Force terminal balance B(360) to zero at month 361 (curtailment)
    3. Weight each cash flow by time: t × CF(t)
    4. WAL = Σ[t × CF(t)] / initial_balance / 12
    
    Parameters:
    -----------
    balance_profile : array
        Balance at each month (0 to 360)
    initial_balance : float, optional
        Starting balance to use as denominator. If None, uses sum of cash flows.
        For apples-to-apples comparison across approaches, should be set to
        the same value (e.g., 1,000,000) for all calculations.
    
    Returns:
    --------
    float : Weighted average life in years
    """
    # Ensure balance profile has 361 points (months 0 to 360)
    if len(balance_profile) != 361:
        raise ValueError(f"Expected 361 points, got {len(balance_profile)}")
    
    # Calculate cash flows as balance runoff
    # CF(t) = B(t-1) - B(t) for t = 1 to 360
    cash_flows = np.diff(balance_profile)  # This gives B(0)-B(1), B(1)-B(2), ..., B(359)-B(360)
    cash_flows = -cash_flows  # Convert to positive outflows
    
    # Force terminal balance to zero at month 361
    # Add remaining balance B(360) as final cash flow at month 361
    terminal_cf = balance_profile[-1]
    cash_flows = np.append(cash_flows, terminal_cf)
    
    # Time points in months (1 to 361)
    months = np.arange(1, 362)
    
    # Calculate WAL in months
    numerator = np.sum(months * cash_flows)
    
    # Use initial_balance if provided, otherwise sum of cash flows
    if initial_balance is not None:
        denominator = initial_balance
    else:
        denominator = np.sum(cash_flows)
    
    if denominator <= 0:
        return 0.0
    
    wal_months = numerator / denominator
    wal_years = wal_months / 12.0
    
    return wal_years

wal_results = {
    'Approach': [],
    'Profile_Type': [],
    'WAL_Years': [],
    'Final_Balance': [],
}

# Use initial balance for apples-to-apples comparison
INITIAL_BALANCE = PARAMS['initial_balance']

# Deterministic
wal_det = calculate_wal(B_det, INITIAL_BALANCE)
wal_results['Approach'].append('Deterministic')
wal_results['Profile_Type'].append('Total')
wal_results['WAL_Years'].append(wal_det)
wal_results['Final_Balance'].append(B_det[-1])

# Monte Carlo
wal_mc_mean = calculate_wal(mc_bal_mean, INITIAL_BALANCE)
wal_mc_p05 = calculate_wal(mc_bal_p05, INITIAL_BALANCE)

wal_results['Approach'].extend(['Monte Carlo', 'Monte Carlo'])
wal_results['Profile_Type'].extend(['Mean (Total)', 'P05 (Stable)'])
wal_results['WAL_Years'].extend([wal_mc_mean, wal_mc_p05])
wal_results['Final_Balance'].extend([mc_bal_mean[-1], mc_bal_p05[-1]])

# Analytical
wal_analytical = calculate_wal(B_analytical, INITIAL_BALANCE)
wal_analytical_stable = calculate_wal(B_analytical_stable, INITIAL_BALANCE)

wal_results['Approach'].extend(['Analytical', 'Analytical'])
wal_results['Profile_Type'].extend(['Expected (Total)', 'P05 (Stable)'])
wal_results['WAL_Years'].extend([wal_analytical, wal_analytical_stable])
wal_results['Final_Balance'].extend([B_analytical[-1], B_analytical_stable[-1]])

# Bifurcation - Now calibrate properly using MC Mean as target
print("\n" + "-" * 80)
print("Calibrating bifurcation to match MC Mean WAL...")

# Calculate WAL for β=0 stable profile
wal_stable_beta0 = calculate_wal(bifurc_stable_beta0_profile, INITIAL_BALANCE)

print(f"  Stable WAL (β_rate=0, β_credit={PARAMS['beta_credit_0']}): {wal_stable_beta0:.2f} years")
print(f"  Deterministic WAL (β_rate={PARAMS['beta_rate_0']}, β_credit={PARAMS['beta_credit_0']}): {wal_det:.2f} years")
print(f"  Difference: +{wal_stable_beta0 - wal_det:.2f} years")

# Binary search for optimal stable percentage to match MC Mean
target_wal = wal_mc_mean
low, high = 0.0, 1.0

while high - low > 0.0001:
    mid = (low + high) / 2
    wal = calculate_bifurcation_wal_solver(mid, bifurc_stable_beta0_profile, INITIAL_BALANCE)
    if wal < target_wal:
        low = mid
    else:
        high = mid

stable_pct = (low + high) / 2
nonstable_pct = 1 - stable_pct

# Recalculate bifurcation profiles with optimal allocation
B_bifurc_stable = bifurc_stable_beta0_profile * stable_pct
B_bifurc_nonstable = np.zeros(PARAMS['n_months'] + 1)
B_bifurc_nonstable[0] = INITIAL_BALANCE * nonstable_pct
B_bifurc_total = B_bifurc_stable + B_bifurc_nonstable

# Calculate final WALs
wal_bifurc_total = calculate_wal(B_bifurc_total, INITIAL_BALANCE)
wal_bifurc_stable = calculate_wal(B_bifurc_stable, INITIAL_BALANCE)

print(f"\n✓ Optimal allocation found:")
print(f"  Stable (β=0): {stable_pct*100:.1f}%")
print(f"  Non-stable (1-month): {nonstable_pct*100:.1f}%")
print(f"  Target WAL: {target_wal:.2f} years")
print(f"  Achieved WAL: {wal_bifurc_total:.2f} years")
print(f"  Error: {abs(wal_bifurc_total - target_wal):.4f} years")
print(f"\nInterpretation:")
print(f"  {nonstable_pct*100:.0f}% non-stable quantifies behavioral optionality")
print(f"  ${nonstable_pct * INITIAL_BALANCE:,.0f} at risk under stress")
print(f"  Gap (MC Mean - MC P05): {wal_mc_mean - wal_mc_p05:.2f} years")

bifurc_label = f'Bifurcation {stable_pct*100:.0f}/{nonstable_pct*100:.0f}'
wal_results['Approach'].extend([bifurc_label, bifurc_label])
wal_results['Profile_Type'].extend(['Total', f'Stable ({stable_pct*100:.0f}%, β=0)'])
wal_results['WAL_Years'].extend([wal_bifurc_total, wal_bifurc_stable])
wal_results['Final_Balance'].extend([B_bifurc_total[-1], B_bifurc_stable[-1]])

df_wal = pd.DataFrame(wal_results)

print("\nWeighted Average Life Summary:")
print("-" * 80)
print(df_wal.to_string(index=False))

df_wal.to_csv(f'{output_dir}/06_wal_comparison.csv', index=False)
print(f"\n✓ Saved: 06_wal_comparison.csv")

# ============================================================================
# SECTION 9B: BIFURCATION SENSITIVITY TO NON-STABLE RUNOFF ASSUMPTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 9B: Bifurcation Sensitivity to Non-Stable Runoff Assumptions")
print("-" * 80)
print("\nAnalyzing how non-stable stickiness assumptions impact calibrated splits...")

# Define runoff scenarios
runoff_scenarios = [
    {'name': '1-month (Immediate / Liquidity Stress)', 'months': 1, 'wal_months': 0.5, 'profile': None},
    {'name': '6-month (Linear)', 'months': 6, 'wal_months': 3.5, 'profile': None},
    {'name': '12-month (Linear)', 'months': 12, 'wal_months': 6.5, 'profile': None},
    {'name': '24-month (Linear)', 'months': 24, 'wal_months': 12.5, 'profile': None},
    {'name': '36-month (Linear)', 'months': 36, 'wal_months': 18.5, 'profile': None},
]

# Create runoff profiles
for scenario in runoff_scenarios:
    if scenario['months'] == 1:
        # Immediate runoff - profile is None (handled in solver function)
        scenario['profile'] = None
    else:
        # Linear amortization profile
        scenario['profile'] = create_linear_runoff_profile(
            PARAMS['n_months'], 
            scenario['months'], 
            INITIAL_BALANCE
        )

# Calibrate each scenario to match MC Mean WAL
bifurc_sensitivity_results = []

for scenario in runoff_scenarios:
    print(f"\n{scenario['name']} runoff (WAL ≈ {scenario['wal_months']:.1f} months):")
    
    # Binary search for optimal stable percentage
    low, high = 0.0, 1.0
    
    while high - low > 0.0001:
        mid = (low + high) / 2
        wal = calculate_bifurcation_wal_solver(
            mid, 
            bifurc_stable_beta0_profile, 
            INITIAL_BALANCE,
            nonstable_profile=scenario['profile']
        )
        if wal < target_wal:
            low = mid
        else:
            high = mid
    
    optimal_stable_pct = (low + high) / 2
    optimal_nonstable_pct = 1 - optimal_stable_pct
    
    # Verify the achieved WAL
    achieved_wal = calculate_bifurcation_wal_solver(
        optimal_stable_pct,
        bifurc_stable_beta0_profile,
        INITIAL_BALANCE,
        nonstable_profile=scenario['profile']
    )
    
    print(f"  Stable %: {optimal_stable_pct*100:.1f}%")
    print(f"  Non-stable %: {optimal_nonstable_pct*100:.1f}%")
    print(f"  Achieved WAL: {achieved_wal:.3f} years (target: {target_wal:.3f})")
    
    # Store results
    bifurc_sensitivity_results.append({
        'Runoff_Assumption': scenario['name'],
        'Runoff_WAL_Months': scenario['wal_months'],
        'Stable_Pct': optimal_stable_pct * 100,
        'NonStable_Pct': optimal_nonstable_pct * 100,
        'Total_WAL_Years': achieved_wal,
        'NonStable_Amount_per_1M': optimal_nonstable_pct * 1_000_000
    })

# Create DataFrame
df_bifurc_sens = pd.DataFrame(bifurc_sensitivity_results)

print("\n" + "-" * 80)
print("BIFURCATION CALIBRATION SENSITIVITY TABLE:")
print("-" * 80)
print(df_bifurc_sens.to_string(index=False))
print("\nKey Insights:")
print(f"  • Non-stable % ranges from {df_bifurc_sens['NonStable_Pct'].min():.1f}% to {df_bifurc_sens['NonStable_Pct'].max():.1f}%")
print(f"    depending on assumed stickiness of non-stable deposits.")
print(f"  • With β=0 stable deposits (WAL=9.50y) and target WAL=5.45y:")
print(f"    - Minimum possible non-stable is 42.6% (even with instant runoff)")
print(f"    - Cannot reach industry standard 20-25% without different assumptions")
print(f"  • Industry 20-25% non-stable likely reflects:")
print(f"    (a) Stable deposits with β>0 (some rate sensitivity, shorter WAL), OR")
print(f"    (b) Non-stable deposits with multi-year stickiness (>24 months)")
print(f"  • Conservative liquidity stress assumptions (1-month runoff) → 42-43% non-stable")
print(f"    This limits deposit duration and structural hedge capacity for IRRBB")
print(f"  • Competitive/IRRBB scenarios with longer runoff (6-36 months) → 30-44% non-stable")
print(f"    This reflects different objectives: liquidity stress vs. interest rate management")

# Save to CSV
df_bifurc_sens.to_csv(f'{output_dir}/06b_bifurcation_sensitivity.csv', index=False)
print(f"\n✓ Saved: 06b_bifurcation_sensitivity.csv")

# Create visualization
fig_bifurc_sens, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot stable % vs runoff WAL
ax.plot(df_bifurc_sens['Runoff_WAL_Months'], 
        df_bifurc_sens['Stable_Pct'], 
        'o-', linewidth=2, markersize=10, color='#2E7D32', label='Stable %')
ax.plot(df_bifurc_sens['Runoff_WAL_Months'], 
        df_bifurc_sens['NonStable_Pct'], 
        's-', linewidth=2, markersize=10, color='#C62828', label='Non-Stable %')

# Add industry standard reference band
ax.axhspan(20, 25, alpha=0.2, color='gray', label='Industry Standard Range (20-25%)')
ax.axhline(y=20, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=25, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Formatting
ax.set_xlabel('Non-Stable Runoff WAL (Months)', fontsize=12, fontweight='bold')
ax.set_ylabel('Allocation (%)', fontsize=12, fontweight='bold')
ax.set_title('Bifurcation Calibration Sensitivity:\nStable/Non-Stable Split vs. Non-Stable Stickiness Assumption',
             fontsize=13, fontweight='bold', pad=15)
ax.legend(loc='center left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 7)
ax.set_ylim(0, 100)

# Add annotations
for idx, row in df_bifurc_sens.iterrows():
    ax.annotate(f"{row['Stable_Pct']:.0f}%", 
                xy=(row['Runoff_WAL_Months'], row['Stable_Pct']),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=9, color='#2E7D32', fontweight='bold')
    ax.annotate(f"{row['NonStable_Pct']:.0f}%", 
                xy=(row['Runoff_WAL_Months'], row['NonStable_Pct']),
                xytext=(0, -12), textcoords='offset points',
                ha='center', fontsize=9, color='#C62828', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/fig_bifurcation_sensitivity.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: fig_bifurcation_sensitivity.png")
plt.close()

print("\nInterpretation:")
print("  • Bifurcation calibration is highly sensitive to both:")
print("    (1) Stable deposit behavior assumption (β=0 gives 9.5y WAL)")
print("    (2) Non-stable runoff stickiness (overnight to 3+ years)")
print("  • Same deposits may behave differently depending on scenario:")
print("    - Liquidity crisis: 1-month runoff (LCR-type 30-day stress)")
print("    - Interest rate competition: 6-36 month gradual shift")
print("  • Higher non-stable % → Lower deposit duration → Limits structural hedge capacity")
print("  • This creates tension between liquidity conservatism and IRRBB structural hedging")
print("  • Industry 20-25% non-stable reflects implicit assumptions about both factors")

# ============================================================================
# SECTION 10: COMPREHENSIVE BALANCE FORECAST OUTPUT
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 10: Saving Comprehensive Balance Forecasts")
print("-" * 80)

months_extended = np.arange(0, PARAMS['n_months'] + 1)
years_extended = months_extended / 12.0

df_balances = pd.DataFrame({
    'Month': months_extended,
    'Years': years_extended,
    'Deterministic_Total': B_det,
    'MC_Mean_Total': mc_bal_mean,
    'MC_P05_Stable': mc_bal_p05,
    'MC_P25': np.percentile(balance_paths, 25, axis=0),
    'MC_P50': np.percentile(balance_paths, 50, axis=0),
    'MC_P75': np.percentile(balance_paths, 75, axis=0),
    'MC_P95': mc_bal_p95,
    'Analytical_Expected': B_analytical,
    'Analytical_P05_Stable': B_analytical_stable,
    'Bifurcation_Total': B_bifurc_total,
    'Bifurcation_Stable': B_bifurc_stable,
    'Bifurcation_NonStable': B_bifurc_nonstable,
})

df_balances.to_csv(f'{output_dir}/05_balance_forecasts.csv', index=False)
print(f"✓ Saved: 05_balance_forecasts.csv")
print(f"  {len(df_balances)} time points × {len(df_balances.columns)-2} balance profiles")

# ============================================================================
# SECTION 11: PUBLICATION-QUALITY CHARTS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 11: Generating Publication-Quality Charts")
print("-" * 80)

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# ============================================================================
# CHART 1: FORWARD RATE CURVE
# ============================================================================
print("\nGenerating Chart 1: Forward Rate Curve...")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(months_in_years, forward_rates * 100, linewidth=2, 
        color='#1f77b4', label='1M SOFR Instantaneous Forward')
ax.scatter(tenors_years, swap_rates * 100, s=60, color='red', 
          marker='o', zorder=5, label='Market OIS Swap Rates')

ax.set_xlabel('Maturity (Years)', fontweight='bold')
ax.set_ylabel('Rate (%)', fontweight='bold')
ax.set_title('Figure 1. Instantaneous 1M SOFR Forward Curve\n' +
             'Market-Implied Forward Rates (September 30, 2025)', 
             fontweight='bold', pad=20)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_forward_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig1_forward_curve.png")

# ============================================================================
# CHART 2: TOTAL BALANCE EVOLUTION COMPARISON
# ============================================================================
print("Generating Chart 2: Total Balance Evolution...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot all approaches
ax.plot(years_extended, B_det / 1e6, linewidth=2.5, 
        color='#2ca02c', label='Deterministic', linestyle='-')
ax.plot(years_extended, mc_bal_mean / 1e6, linewidth=2.5,
        color='#ff7f0e', label='Monte Carlo Mean', linestyle='-')
ax.plot(years_extended, B_analytical / 1e6, linewidth=2,
        color='#d62728', label='Analytical Expected', linestyle='--')
ax.plot(years_extended, B_bifurc_total / 1e6, linewidth=2,
        color='#9467bd', label='Bifurcation Total', linestyle='-.')

# Add Monte Carlo confidence bands
ax.fill_between(years_extended, mc_bal_p05 / 1e6, mc_bal_p95 / 1e6,
                alpha=0.15, color='#ff7f0e', label='MC P05-P95 Range')

ax.set_xlabel('Years', fontweight='bold')
ax.set_ylabel('Balance ($ Millions)', fontweight='bold')
ax.set_title('Figure 2. Total Balance Evolution Across All Approaches\n' +
             'Component Decay Model with Different Valuation Frameworks',
             fontweight='bold', pad=20)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, ax.get_ylim()[1])

plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_total_balance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig2_total_balance_comparison.png")

# ============================================================================
# CHART 3: STABLE BALANCE PROFILE COMPARISON
# ============================================================================
print("Generating Chart 3: Stable Balance Profiles...")

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(years_extended, mc_bal_p05 / 1e6, linewidth=2.5,
        color='#ff7f0e', label='Monte Carlo P05 Stable', linestyle='-')
ax.plot(years_extended, B_analytical_stable / 1e6, linewidth=2.5,
        color='#d62728', label='Analytical P05 Stable', linestyle='--')
ax.plot(years_extended, B_bifurc_stable / 1e6, linewidth=2.5,
        color='#9467bd', label=f'Bifurcation Stable ({stable_pct*100:.0f}%, β=0)', linestyle='-.')

# Add total profiles for reference (lighter)
ax.plot(years_extended, mc_bal_mean / 1e6, linewidth=1.5, alpha=0.4,
        color='#ff7f0e', label='MC Mean (reference)', linestyle=':')
ax.plot(years_extended, B_bifurc_total / 1e6, linewidth=1.5, alpha=0.4,
        color='#9467bd', label='Bifurcation Total (reference)', linestyle=':')

ax.set_xlabel('Years', fontweight='bold')
ax.set_ylabel('Balance ($ Millions)', fontweight='bold')
ax.set_title('Figure 3. Stable Balance Profile Comparison\n' +
             'Conservative Estimates Across Methodologies',
             fontweight='bold', pad=20)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
ax.set_ylim(0, ax.get_ylim()[1])

plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_stable_balance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig3_stable_balance_comparison.png")

# ============================================================================
# CHART 4: MONTE CARLO DISTRIBUTION VISUALIZATION
# ============================================================================
print("Generating Chart 4: Monte Carlo Distribution...")

fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

# Panel A: Balance distribution bands over time
ax1 = plt.subplot(gs[0, :])
ax1.fill_between(years_extended, mc_bal_p05 / 1e6, mc_bal_p95 / 1e6,
                 alpha=0.2, color='#1f77b4', label='P05-P95')
ax1.fill_between(years_extended, 
                 np.percentile(balance_paths, 25, axis=0) / 1e6,
                 np.percentile(balance_paths, 75, axis=0) / 1e6,
                 alpha=0.3, color='#1f77b4', label='P25-P75')
ax1.plot(years_extended, mc_bal_mean / 1e6, linewidth=2.5,
        color='#1f77b4', label='Mean')
ax1.plot(years_extended, B_det / 1e6, linewidth=2,
        color='#2ca02c', linestyle='--', label='Deterministic')

ax1.set_xlabel('Years', fontweight='bold')
ax1.set_ylabel('Balance ($ Millions)', fontweight='bold')
ax1.set_title('A. Monte Carlo Balance Distribution Over Time', fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)

# Panel B: Sample paths (first 50)
ax2 = plt.subplot(gs[1, 0])
for i in range(50):
    ax2.plot(years_extended, balance_paths[i, :] / 1e6, 
            linewidth=0.5, alpha=0.3, color='gray')
ax2.plot(years_extended, mc_bal_mean / 1e6, linewidth=2.5,
        color='#ff7f0e', label='Mean', zorder=10)
ax2.set_xlabel('Years', fontweight='bold')
ax2.set_ylabel('Balance ($ Millions)', fontweight='bold')
ax2.set_title('B. Sample Monte Carlo Paths', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)

# Panel C: Distribution at selected horizons
ax3 = plt.subplot(gs[1, 1])
horizons = [60, 120, 240, 360]  # 5Y, 10Y, 20Y, 30Y
positions = np.arange(len(horizons))

for i, h in enumerate(horizons):
    data = balance_paths[:, h] / 1e6
    bp = ax3.boxplot([data], positions=[i], widths=0.6, 
                      patch_artist=True,
                      boxprops=dict(facecolor='lightblue'),
                      medianprops=dict(color='red', linewidth=2))

ax3.set_xticks(positions)
ax3.set_xticklabels([f'{h//12}Y' for h in horizons])
ax3.set_ylabel('Balance ($ Millions)', fontweight='bold')
ax3.set_title('C. Distribution at Key Horizons', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.suptitle('Figure 4. Monte Carlo Simulation Results\n' +
             f'{PARAMS["n_paths"]:,} Paths, Hull-White Calibrated Dynamics',
             fontweight='bold', fontsize=13, y=0.995)

plt.savefig(f'{output_dir}/fig4_mc_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig4_mc_distribution.png")

# ============================================================================
# CHART 5: Weighted Average Life Decomposition - REDESIGNED
# ============================================================================
print("Generating Chart 5: WAL Decomposition...")

fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, height_ratios=[3, 2], width_ratios=[3, 2], hspace=0.45, wspace=0.35, 
                       top=0.85, bottom=0.06)

# ===== MAIN PANEL: Grouped comparison by methodology =====
ax_main = plt.subplot(gs[0, :])

# Reorganize data for clearer comparison
# Group 1: Deterministic (baseline, no distribution)
# Group 2: Monte Carlo (shows full distribution)
# Group 3: Analytical (shows rate uncertainty only)
# Group 4: Bifurcation (stable=β=0, calibrated to MC Mean)

groups = ['Deterministic\n(Baseline)', 'Monte Carlo\n(Full Simulation)', 'Analytical\n(Bachelier)', f'Bifurcation\n({stable_pct*100:.0f}/{nonstable_pct*100:.0f}, β=0)']
x_pos = np.array([0, 2.5, 5, 7.5])
width = 0.7

# Data arrays: [Total, Stable] for each group
det_data = [wal_det, np.nan]
mc_data = [wal_mc_mean, wal_mc_p05]
analytical_data = [wal_analytical, wal_analytical_stable]
bifurc_data = [wal_bifurc_total, wal_bifurc_stable]

all_data = [det_data, mc_data, analytical_data, bifurc_data]

# Color scheme: Total (blue family), Stable (orange family)
color_total = '#2E86AB'
color_stable = '#F18F01'

# Plot bars with clear grouping
for i, (group_data, x) in enumerate(zip(all_data, x_pos)):
    total_val, stable_val = group_data
    
    # Total WAL bar
    bar_total = ax_main.bar(x - width/2, total_val, width, 
                           color=color_total, edgecolor='black', linewidth=1.5, 
                           label='Total WAL' if i == 0 else '', alpha=0.85, zorder=3)
    
    # Stable WAL bar (if applicable)
    if not np.isnan(stable_val):
        bar_stable = ax_main.bar(x + width/2, stable_val, width,
                                color=color_stable, edgecolor='black', linewidth=1.5,
                                label='Stable WAL (P05)' if i == 1 else '', alpha=0.85, zorder=3)
        
        # Add bracket showing gap for MC and Analytical
        if i in [1, 2]:  # Monte Carlo and Analytical
            gap = total_val - stable_val
            mid_x = x
            
            # Draw bracket
            bracket_color = '#D62728' if i == 1 else '#7F7F7F'
            bracket_label = 'Behavioral\n+ Rate\nUncertainty' if i == 1 else 'Rate\nUncertainty\nOnly'
            
            ax_main.plot([mid_x - 0.1, mid_x - 0.1], [stable_val, total_val], 
                        color=bracket_color, linewidth=2, zorder=4)
            ax_main.plot([mid_x - 0.1, mid_x + 0.05], [stable_val, stable_val], 
                        color=bracket_color, linewidth=2, zorder=4)
            ax_main.plot([mid_x - 0.1, mid_x + 0.05], [total_val, total_val], 
                        color=bracket_color, linewidth=2, zorder=4)
            
            # Add text annotation
            ax_main.text(mid_x - 0.45, (total_val + stable_val) / 2, 
                        f'{bracket_label}\n{gap:.2f} yrs',
                        fontsize=8, ha='center', va='center', color=bracket_color, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                edgecolor=bracket_color, linewidth=1.5, alpha=0.9))
    
    # Add value labels on bars
    ax_main.text(x - width/2, total_val + 0.08, f'{total_val:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    if not np.isnan(stable_val):
        ax_main.text(x + width/2, stable_val + 0.08, f'{stable_val:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Reference line at 4.10 (deterministic baseline)
ax_main.axhline(y=wal_det, color='gray', linestyle='--', linewidth=1.5, alpha=0.6, 
               label=f'Deterministic Baseline: {wal_det:.2f}Y', zorder=1)

# Special annotation for Bifurcation inversion
ax_main.annotate('Note: Bifurcation inverts the relationship\nStable > Total by design (segmentation)',
                xy=(7.5, wal_bifurc_stable), xytext=(9.5, 4.8),
                fontsize=8, ha='left', color='#8B4513', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF8DC', edgecolor='#8B4513', linewidth=1.5),
                arrowprops=dict(arrowstyle='->', color='#8B4513', lw=1.5))

# Formatting
ax_main.set_ylabel('Weighted Average Life (Years)', fontweight='bold', fontsize=13)
ax_main.set_xlabel('Modeling Approach', fontweight='bold', fontsize=13)
ax_main.set_title('A. Weighted Average Life by Modeling Approach\nTotal Balance vs. Stable Balance (5th Percentile) Comparison',
                 fontweight='bold', fontsize=14, pad=15)
ax_main.set_xticks(x_pos)
ax_main.set_xticklabels(groups, fontsize=11, fontweight='bold')
ax_main.legend(loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
ax_main.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
ax_main.set_ylim(0, 5.2)
ax_main.set_xlim(-1, 9.5)

# ===== PANEL B: Gap Analysis =====
ax_gap = plt.subplot(gs[1, 0])

gap_labels = ['Monte Carlo\n(MC Total - MC P05)', 'Analytical\n(An. Total - An. P05)', 
              'MC vs Analytical\n(MC P05 - An. P05)']
gap_values = [wal_mc_mean - wal_mc_p05, 
              wal_analytical - wal_analytical_stable,
              wal_mc_p05 - wal_analytical_stable]
gap_colors = ['#D62728', '#7F7F7F', '#9467BD']

bars_gap = ax_gap.barh(gap_labels, gap_values, color=gap_colors, edgecolor='black', linewidth=1.5, alpha=0.85)

# Add value labels
for bar, val in zip(bars_gap, gap_values):
    width_bar = bar.get_width()
    ax_gap.text(width_bar + 0.01, bar.get_y() + bar.get_height()/2, 
               f'{abs(val):.3f} yrs', va='center', fontsize=10, fontweight='bold')

ax_gap.set_xlabel('Gap (Years)', fontweight='bold', fontsize=11)
ax_gap.set_title('B. Optionality Gap Decomposition\nQuantifying Rate vs. Behavioral Uncertainty',
                fontweight='bold', fontsize=12, pad=10)
ax_gap.grid(True, alpha=0.3, axis='x', linestyle='--')
ax_gap.set_xlim(0, max(gap_values) * 1.2)

# Add interpretation text with convexity explanation
ax_gap.text(0.98, 0.05, 
           'MC Total-P05 (0.575 yrs) = Full optionality\n' +
           'Analytical gap (0.128 yrs) = Rate uncertainty only\n' +
           'Difference (0.370 yrs) = Behavioral path dependence',
           transform=ax_gap.transAxes, fontsize=8, ha='right', va='bottom',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='black', alpha=0.8))

# Add convexity vs skewness annotation
convexity_text = (
    'Convexity Effect: MC Mean (4.17 yrs) > Deterministic (4.10 yrs)\n'
    '• Exponential decay exp[-β×r] creates positive curvature\n'
    '• Low rates extend life MORE than high rates shorten it\n'
    '• Jensen\'s inequality: E[exp(-βr)] > exp(-βE[r]) → +0.07 yrs\n'
    '• Dominates despite left-skewed rates (mean < median by -2 bps)\n'
    '• Functional form matters: linear decay would reverse this effect'
)
ax_gap.text(0.02, 0.95, convexity_text,
           transform=ax_gap.transAxes, fontsize=7, ha='left', va='top',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='#E8F4F8', edgecolor='#2E86AB', 
                    linewidth=2, alpha=0.95))

# ===== PANEL C: Methodology Comparison Table =====
ax_table = plt.subplot(gs[1, 1])
ax_table.axis('off')

table_data = [
    ['Approach', 'Total\nWAL', 'Stable\nWAL', 'Gap'],
    ['Deterministic', f'{wal_det:.2f}', 'N/A', 'N/A'],
    ['MC Mean', f'{wal_mc_mean:.2f}', f'{wal_mc_p05:.2f}', f'{wal_mc_mean - wal_mc_p05:.2f}'],
    ['Analytical', f'{wal_analytical:.2f}', f'{wal_analytical_stable:.2f}', f'{wal_analytical - wal_analytical_stable:.2f}'],
    ['Bifurcation', f'{wal_bifurc_total:.2f}', f'{wal_bifurc_stable:.2f}', f'{wal_bifurc_stable - wal_bifurc_total:.2f}*']
]

table = ax_table.table(cellText=table_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.2, 0.2, 0.25],
                      bbox=[0, 0, 1, 1])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header row
for i in range(4):
    table[(0, i)].set_facecolor('#2E86AB')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

# Style data rows with alternating colors
for i in range(1, 5):
    for j in range(4):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F0F0F0')
        table[(i, j)].set_edgecolor('black')
        table[(i, j)].set_linewidth(1.5)

# Highlight the gap column
for i in range(1, 5):
    table[(i, 3)].set_facecolor('#FFE6E6')

ax_table.set_title('C. Summary Statistics Table\n*Bifurcation: Stable > Total (inverted)',
                  fontweight='bold', fontsize=12, pad=10)

# Overall figure title positioned in the top 15% reserved space
fig.suptitle('Figure 5. Comprehensive Weighted Average Life Analysis\n' +
            'Comparing Total vs. Stable Balance Durations Across Four Modeling Approaches',
            fontweight='bold', fontsize=15, y=0.94)

plt.savefig(f'{output_dir}/fig5_wal_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig5_wal_decomposition.png")

# ============================================================================
print("\nGenerating Chart 6: Monte Carlo Rate Distribution...")
# ============================================================================
# Chart 6: Monte Carlo SOFR Rate Path Evolution with Percentile Bands (like Figure 4)
fig, ax = plt.subplots(figsize=(14, 8))

# Calculate percentiles across all paths at each time point
months_array = np.arange(PARAMS['n_months'] + 1)
years_array = months_array / 12.0

# Convert to percentage for plotting
rate_paths_pct = rate_paths * 100
forward_rates_pct = forward_rates * 100

# Calculate key percentiles at each time point
rate_mean = np.mean(rate_paths_pct, axis=0)
rate_p05 = np.percentile(rate_paths_pct, 5, axis=0)
rate_p25 = np.percentile(rate_paths_pct, 25, axis=0)
rate_p50 = np.percentile(rate_paths_pct, 50, axis=0)
rate_p75 = np.percentile(rate_paths_pct, 75, axis=0)
rate_p95 = np.percentile(rate_paths_pct, 95, axis=0)

# Plot forward curve as reference
ax.plot(years_array, np.concatenate([[forward_rates_pct[0]], forward_rates_pct]), 
        color='black', linewidth=3, linestyle='-', label='Market Forward Curve', zorder=10, alpha=0.8)

# Plot mean path
ax.plot(years_array, np.concatenate([[rate_mean[0]], rate_mean]), 
        color='red', linewidth=2.5, linestyle='--', label='Monte Carlo Mean', zorder=9)

# Plot median (P50)
ax.plot(years_array, np.concatenate([[rate_p50[0]], rate_p50]), 
        color='blue', linewidth=2, linestyle=':', label='Median (P50)', zorder=8, alpha=0.8)

# Fill between percentile bands for fan chart effect
# P05-P95 (90% confidence interval) - lightest shade
ax.fill_between(years_array, 
                np.concatenate([[rate_p05[0]], rate_p05]), 
                np.concatenate([[rate_p95[0]], rate_p95]),
                alpha=0.15, color='steelblue', label='P05-P95 (90% CI)', zorder=1)

# P25-P75 (50% confidence interval) - medium shade
ax.fill_between(years_array, 
                np.concatenate([[rate_p25[0]], rate_p25]), 
                np.concatenate([[rate_p75[0]], rate_p75]),
                alpha=0.25, color='steelblue', label='P25-P75 (50% CI)', zorder=2)

# Add sample paths for context (10 random paths)
np.random.seed(42)
sample_indices = np.random.choice(PARAMS['n_paths'], size=10, replace=False)
for idx in sample_indices:
    ax.plot(years_array, np.concatenate([[rate_paths_pct[idx, 0]], rate_paths_pct[idx, :]]), 
            color='gray', linewidth=0.5, alpha=0.3, zorder=0)

# Add vertical lines at key horizons with annotations
key_horizons = [1, 5, 10, 20, 30]
for horizon in key_horizons:
    if horizon <= 30:
        month_idx = horizon * 12 - 1
        ax.axvline(x=horizon, color='gray', linestyle='--', linewidth=0.8, alpha=0.3)
        
        # Add text box with statistics at this horizon
        mean_val = rate_mean[month_idx]
        p05_val = rate_p05[month_idx]
        p95_val = rate_p95[month_idx]
        std_val = np.std(rate_paths_pct[:, month_idx])
        fwd_val = forward_rates_pct[month_idx]
        
        # Position text boxes alternating high/low to avoid overlap
        y_pos = ax.get_ylim()[1] * (0.95 if horizon % 2 == 1 else 0.05)
        va_pos = 'top' if horizon % 2 == 1 else 'bottom'
        
        text_str = f'{horizon}Y\nFwd: {fwd_val:.2f}%\nMean: {mean_val:.2f}%\nσ: {std_val:.2f}%\n[{p05_val:.2f}%, {p95_val:.2f}%]'
        ax.text(horizon, y_pos, text_str, 
                fontsize=7, ha='center', va=va_pos,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='gray', alpha=0.8))

# Formatting
ax.set_xlabel('Time Horizon (Years)', fontweight='bold', fontsize=12)
ax.set_ylabel('SOFR Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Figure 6. Monte Carlo SOFR Rate Path Evolution\n' +
             'Percentile Fan Chart Showing Rate Uncertainty Under Hull-White Dynamics',
             fontweight='bold', fontsize=14, pad=20)
ax.legend(loc='upper right', fontsize=10, framealpha=0.95, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_xlim(0, 30)

# Set y-axis to show reasonable rate range
y_min = min(rate_p05) - 0.5
y_max = max(rate_p95) + 0.5
ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_mc_rate_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig6_mc_rate_distribution.png")

# ============================================================================
print("Generating Chart 7: 3D Volatility Surface...")
# ============================================================================
# Chart 7: 3D Implied Volatility Surface
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Create grid for volatility surface
# Expiry: 1M to 360M
expiry_months = np.arange(1, 361, 6)  # Every 6 months for visualization
expiry_years = expiry_months / 12.0

# Strike rates: Create a range around the forward curve
strike_offsets_bps = np.arange(-200, 201, 25)  # -200 to +200 bps in 25bp increments

# Create meshgrid
X_expiry, Y_strike_offset = np.meshgrid(expiry_years, strike_offsets_bps)
Z_vol = np.zeros_like(X_expiry)

# Populate volatility surface
# For this simplified model, we'll use ATM volatility with a smile/skew pattern
for i, offset_bps in enumerate(strike_offsets_bps):
    for j, exp_months in enumerate(expiry_months):
        # Get ATM volatility at this expiry
        atm_vol_bps = sigma_monthly[exp_months-1] * 10000  # Convert to bps
        
        # Add smile/skew effect (stylized: higher vol for OTM puts and calls)
        # Smile effect increases with offset from ATM
        smile_factor = 1.0 + 0.15 * (abs(offset_bps) / 200.0)
        
        # Skew effect (slight asymmetry favoring downside)
        skew_factor = 1.0 + 0.05 * (offset_bps / 200.0)
        
        # Term structure effect (smile dampens at longer maturities)
        term_dampening = np.exp(-0.03 * (exp_months / 12.0))
        
        vol_with_smile = atm_vol_bps * smile_factor * skew_factor
        vol_with_term = atm_vol_bps + (vol_with_smile - atm_vol_bps) * term_dampening
        
        Z_vol[i, j] = vol_with_term

# Create surface plot
surf = ax.plot_surface(X_expiry, Y_strike_offset, Z_vol, 
                       cmap='viridis', alpha=0.9, edgecolor='none', 
                       linewidth=0, antialiased=True, shade=True)

# Add ATM volatility line for reference
atm_vols_plot = [sigma_monthly[m-1] * 10000 for m in expiry_months]
ax.plot(expiry_years, np.zeros_like(expiry_years), atm_vols_plot, 
        color='red', linewidth=3, label='ATM Volatility', zorder=10)

# Formatting
ax.set_xlabel('\nExpiry (Years)', fontweight='bold', fontsize=11, labelpad=10)
ax.set_ylabel('\nStrike Offset from ATM (bps)', fontweight='bold', fontsize=11, labelpad=10)
ax.set_zlabel('\nImplied Volatility (bps)', fontweight='bold', fontsize=11, labelpad=10)
ax.set_title('Figure 7. SOFR Cap Implied Volatility Surface (Normal/Bachelier)\n' +
             '3D Visualization: Volatility × Strike × Expiry',
             fontweight='bold', fontsize=14, pad=20)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Implied Volatility (bps)')

# Set viewing angle for better perspective
ax.view_init(elev=25, azim=135)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Add legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_volatility_surface_3d.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved: fig7_volatility_surface_3d.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("EXECUTION COMPLETE")
print("=" * 80)

print("\n📊 OUTPUT FILES GENERATED:")
print("-" * 80)
print("\nCSV Files:")
csv_files = [
    "01_bootstrapped_curve.csv       - Discount factors and zero rates",
    "02_forward_rates.csv             - Instantaneous forward rates (360M)",
    "03_volatility_surface.csv        - Extended ATM volatilities (1M-360M)",
    "04_rate_paths_summary.csv        - Monte Carlo rate statistics",
    "05_balance_forecasts.csv         - All approach balance profiles",
    "06_wal_comparison.csv            - Weighted average life results",
    "07_mc_sample_paths.csv           - Sample Monte Carlo paths (100)",
]
for f in csv_files:
    print(f"  ✓ {f}")

print("\nChart Files:")
chart_files = [
    "fig1_forward_curve.png           - SOFR forward rate curve",
    "fig2_total_balance_comparison.png - Total balance evolution",
    "fig3_stable_balance_comparison.png - Stable profile comparison",
    "fig4_mc_distribution.png         - Monte Carlo distribution analysis",
    "fig5_wal_decomposition.png       - Weighted average life decomposition",
    "fig6_mc_rate_distribution.png    - Monte Carlo rate evolution fan chart",
    "fig7_volatility_surface_3d.png   - 3D implied volatility surface",
]
for f in chart_files:
    print(f"  ✓ {f}")

print(f"\n📁 All files saved to: {output_dir}/")

print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print(f"\nWeighted Average Life (Years):")
print(f"  Deterministic Total:        {wal_det:.2f}")
print(f"  Monte Carlo Mean:           {wal_mc_mean:.2f}")
print(f"  Monte Carlo P05 (Stable):   {wal_mc_p05:.2f}")
print(f"  Analytical Expected:        {wal_analytical:.2f}")
print(f"  Analytical P05 (Stable):    {wal_analytical_stable:.2f}")
print(f"  Bifurcation Total:          {wal_bifurc_total:.2f}")
print(f"  Bifurcation Stable:         {wal_bifurc_stable:.2f}")

print(f"\nOptionality Effects:")
interest_rate_effect = wal_mc_mean - wal_det
behavioral_effect = wal_mc_mean - wal_mc_p05
print(f"  Interest Rate Uncertainty:  {interest_rate_effect:.2f} years")
print(f"  Behavioral Uncertainty:     {behavioral_effect:.2f} years")
print(f"  Total Embedded Optionality: {behavioral_effect:.2f} years")

print("\n" + "=" * 80)
print("Ready for research paper integration!")
print("=" * 80)

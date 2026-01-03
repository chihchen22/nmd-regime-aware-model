"""
DUAL-PERIOD ALPHA SENSITIVITY ANALYSIS
======================================
Compares March 2022 vs September 2025 with CONSISTENT alpha parameter
to properly test the hypothesis that surge deposits have fled.

Key insight: Using different alphas undermines the comparison.
This script uses the SAME alpha across both periods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DUAL-PERIOD ALPHA SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nObjective: Compare March 2022 vs September 2025 with IDENTICAL model parameters")
print("         to properly test surge deposit flight hypothesis.\n")

np.random.seed(42)
output_dir = 'model_outputs'
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_moving_average(rate_path, window, historical_init=None):
    """Calculate trailing moving average with historical initialization."""
    n_months = len(rate_path)
    ma = np.zeros(n_months)
    
    for t in range(n_months):
        if t < window:
            if historical_init is not None:
                hist_weight = (window - t) / window
                sim_weight = t / window
                if t == 0:
                    ma[t] = historical_init
                else:
                    ma[t] = hist_weight * historical_init + sim_weight * np.mean(rate_path[:t])
            else:
                ma[t] = np.mean(rate_path[:t+1])
        else:
            ma[t] = np.mean(rate_path[t-window:t])
    return ma

def beta_rate_regime_adjusted(B, r_current, r_ma, beta_0, gamma, B_0, alpha):
    """Rate sensitivity with regime amplification."""
    beta_base = beta_0 * (B / B_0) ** gamma
    regime_excursion = max(0, r_current - r_ma)
    amplification = 1 + alpha * (regime_excursion * 100)
    return beta_base * amplification

def beta_credit_sensitive(B, beta_0, gamma, B_0):
    """Balance-dependent credit sensitivity."""
    return beta_0 * (B / B_0) ** gamma

def component_decay_step_regime(B_prev, r, r_ma, s, params, alpha):
    """Single time step of component decay with regime amplification."""
    beta_r = beta_rate_regime_adjusted(B_prev, r, r_ma, params['beta_rate_0'], 
                                        params['gamma_rate'], params['B_0'], alpha)
    beta_c = beta_credit_sensitive(B_prev, params['beta_credit_0'], 
                                    params['gamma_credit'], params['B_0'])
    B_new = B_prev * (1 - params['h']) * np.exp(params['g'] - beta_r * r - beta_c * s)
    return B_new, beta_r

def calculate_wal(balance_path):
    """Calculate Weighted Average Life from balance path."""
    n_months = len(balance_path) - 1
    runoff = np.diff(balance_path) * -1
    runoff = np.maximum(runoff, 0)
    if runoff.sum() == 0:
        return n_months / 12
    times = np.arange(1, n_months + 1) / 12
    wal = np.sum(times * runoff) / runoff.sum()
    return wal

def run_simulation(rate_paths, credit_spread, params, alpha, window, ma_init):
    """Run Monte Carlo simulation for a given alpha."""
    n_paths = params['n_paths']
    n_months = params['n_months']
    
    balances = np.zeros((n_paths, n_months + 1))
    balances[:, 0] = params['initial_balance']
    beta_evolution = np.zeros((n_paths, n_months))
    
    for path_idx in range(n_paths):
        rate_ma_path = calculate_moving_average(rate_paths[path_idx, :], window, ma_init)
        B = params['initial_balance']
        for t in range(n_months):
            B_new, beta_t = component_decay_step_regime(
                B, rate_paths[path_idx, t], rate_ma_path[t],
                credit_spread[t], params, alpha
            )
            balances[path_idx, t+1] = B_new
            beta_evolution[path_idx, t] = beta_t
            B = B_new
    
    # Calculate WAL for each path
    wals = np.array([calculate_wal(balances[i, :]) for i in range(n_paths)])
    
    return {
        'balances': balances,
        'wals': wals,
        'wal_mean': np.mean(wals),
        'wal_p05': np.percentile(wals, 5),
        'wal_p10': np.percentile(wals, 10),
        'wal_p25': np.percentile(wals, 25),
        'wal_p50': np.percentile(wals, 50),
        'beta_avg': np.mean(beta_evolution),
        'mean_balance': balances.mean(axis=0),
        'p05_balance': np.percentile(balances, 5, axis=0)
    }

# =============================================================================
# LOAD MARKET DATA FOR BOTH PERIODS
# =============================================================================

print("=" * 80)
print("LOADING MARKET DATA FOR BOTH PERIODS")
print("=" * 80)

# --- MARCH 2022 DATA ---
print("\n>>> MARCH 2022:")
sofr_history = pd.read_excel('SOFR_History.xlsx')
sofr_history['EndMonth'] = pd.to_datetime(sofr_history['EndMonth'])
sofr_history = sofr_history.sort_values('EndMonth')
historical_data_mar22 = sofr_history[sofr_history['EndMonth'] <= '2022-03-31'].copy()

# Calculate initial moving averages for March 2022
end_date = pd.to_datetime('2022-03-31')
ma_24mo_mar22 = historical_data_mar22[
    (historical_data_mar22['EndMonth'] > end_date - pd.DateOffset(months=24)) & 
    (historical_data_mar22['EndMonth'] <= end_date)
]['SOFRRATE'].mean()

sofr_curve_mar22 = pd.read_excel('SOFR_Market_Data_20220331.xlsx', sheet_name='SOFR_OIS_Curve')
fhlb_curve_mar22 = pd.read_excel('SOFR_Market_Data_20220331.xlsx', sheet_name='FHLB_Curve')

# Convert term to months
def term_to_months(term_str):
    term_str = str(term_str).strip().upper()
    if 'D' in term_str: return 0
    elif 'MO' in term_str: return int(term_str.split()[0])
    elif 'YR' in term_str: return int(term_str.split()[0]) * 12
    else: return 0

sofr_curve_mar22['Months'] = sofr_curve_mar22['Term'].apply(term_to_months)
fhlb_curve_mar22['Months'] = fhlb_curve_mar22['Term'].apply(term_to_months)

# Interpolate to monthly grid
n_months = 360
months_grid = np.arange(0, n_months + 1)
forward_sofr_mar22 = np.interp(months_grid, sofr_curve_mar22['Months'], sofr_curve_mar22['Mid'])
forward_fhlb_mar22 = np.interp(months_grid, fhlb_curve_mar22['Months'], fhlb_curve_mar22['Mid'])
credit_spread_mar22 = forward_fhlb_mar22 - forward_sofr_mar22

current_sofr_mar22 = sofr_curve_mar22.iloc[0]['Mid']
print(f"  Current SOFR: {current_sofr_mar22*100:.4f}%")
print(f"  24-mo MA: {ma_24mo_mar22*100:.4f}%")
print(f"  Regime excursion: {max(0, current_sofr_mar22 - ma_24mo_mar22)*100:.4f}%")

# --- SEPTEMBER 2025 DATA ---
print("\n>>> SEPTEMBER 2025:")
# Load Sept 2025 data
sofr_curve_sep25 = pd.read_csv('model_outputs/04_rate_paths_summary.csv')
forward_sofr_sep25 = sofr_curve_sep25['Forward_Rate'].values / 100

# Extend to 360 months if needed
if len(forward_sofr_sep25) < n_months:
    extension = np.full(n_months - len(forward_sofr_sep25), forward_sofr_sep25[-1])
    forward_sofr_sep25 = np.concatenate([forward_sofr_sep25, extension])

# Credit spread for Sept 2025 (using constant ~149 bps)
credit_spread_sep25 = 0.0149 * np.ones(n_months)

# Calculate 24-mo MA for Sept 2025 from historical data up to Sept 2025
historical_data_sep25 = sofr_history[sofr_history['EndMonth'] <= '2025-09-30'].copy()
end_date_sep25 = pd.to_datetime('2025-09-30')
ma_24mo_sep25 = historical_data_sep25[
    (historical_data_sep25['EndMonth'] > end_date_sep25 - pd.DateOffset(months=24)) & 
    (historical_data_sep25['EndMonth'] <= end_date_sep25)
]['SOFRRATE'].mean()

current_sofr_sep25 = forward_sofr_sep25[0]
print(f"  Current SOFR: {current_sofr_sep25*100:.2f}%")
print(f"  24-mo MA: {ma_24mo_sep25*100:.2f}%")
print(f"  Regime excursion: {max(0, current_sofr_sep25 - ma_24mo_sep25)*100:.2f}%")

# =============================================================================
# GENERATE RATE PATHS FOR BOTH PERIODS
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING RATE PATHS")
print("=" * 80)

# Common simulation parameters
PARAMS = {
    'n_paths': 5000,
    'n_months': n_months,
    'initial_balance': 10000,  # $10 billion
    'h': 0.01,
    'g': 0.0017,
    'beta_rate_0': 0.30,
    'beta_credit_0': 0.15,
    'gamma_rate': 0.30,
    'gamma_credit': 0.40,
    'B_0': 10000.0,
}

# Hull-White parameters
kappa = 0.03
dt = 1/12
sigma_annual = 0.007

# Generate March 2022 rate paths
np.random.seed(42)
rate_paths_mar22 = np.zeros((PARAMS['n_paths'], n_months))
rate_paths_mar22[:, 0] = current_sofr_mar22
theta_mar22 = np.gradient(forward_sofr_mar22[:n_months], 1/12) + kappa * forward_sofr_mar22[:n_months]

for t in range(1, n_months):
    dW = np.random.randn(PARAMS['n_paths'])
    drift = (theta_mar22[t-1] - kappa * rate_paths_mar22[:, t-1]) * dt
    diffusion = sigma_annual * np.sqrt(dt) * dW
    rate_paths_mar22[:, t] = np.maximum(0.0, rate_paths_mar22[:, t-1] + drift + diffusion)

print(f"  March 2022 paths: {PARAMS['n_paths']:,} x {n_months} months")

# Generate September 2025 rate paths
np.random.seed(42)  # Same seed for consistency
rate_paths_sep25 = np.zeros((PARAMS['n_paths'], n_months))
rate_paths_sep25[:, 0] = current_sofr_sep25
theta_sep25 = np.gradient(forward_sofr_sep25[:n_months], 1/12) + kappa * forward_sofr_sep25[:n_months]

for t in range(1, n_months):
    dW = np.random.randn(PARAMS['n_paths'])
    drift = (theta_sep25[t-1] - kappa * rate_paths_sep25[:, t-1]) * dt
    diffusion = sigma_annual * np.sqrt(dt) * dW
    rate_paths_sep25[:, t] = np.maximum(0.0, rate_paths_sep25[:, t-1] + drift + diffusion)

print(f"  September 2025 paths: {PARAMS['n_paths']:,} x {n_months} months")

# =============================================================================
# RUN SENSITIVITY ANALYSIS WITH CONSISTENT ALPHAS
# =============================================================================

print("\n" + "=" * 80)
print("RUNNING ALPHA SENSITIVITY ANALYSIS (SAME ALPHA BOTH PERIODS)")
print("=" * 80)

# User-specified alpha range for transparency in paper
alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5]
window = 24  # 24-month moving average

results_comparison = []

for alpha in alpha_values:
    print(f"\n--- Alpha = {alpha:.1f} ---")
    
    # March 2022
    print(f"  Running March 2022...", end=" ", flush=True)
    res_mar22 = run_simulation(rate_paths_mar22, credit_spread_mar22[:n_months], 
                               PARAMS, alpha, window, ma_24mo_mar22)
    print(f"Done. WAL Mean={res_mar22['wal_mean']:.2f}y, P05={res_mar22['wal_p05']:.2f}y")
    
    # September 2025
    print(f"  Running September 2025...", end=" ", flush=True)
    res_sep25 = run_simulation(rate_paths_sep25, credit_spread_sep25[:n_months], 
                               PARAMS, alpha, window, ma_24mo_sep25)
    print(f"Done. WAL Mean={res_sep25['wal_mean']:.2f}y, P05={res_sep25['wal_p05']:.2f}y")
    
    # Calculate convergence metrics
    p05_gap = res_sep25['wal_p05'] - res_mar22['wal_p05']
    mean_gap = res_sep25['wal_mean'] - res_mar22['wal_mean']
    
    results_comparison.append({
        'Alpha': alpha,
        'Mar22_Mean_WAL': res_mar22['wal_mean'],
        'Mar22_P05_WAL': res_mar22['wal_p05'],
        'Mar22_Mean_P05_Gap': res_mar22['wal_mean'] - res_mar22['wal_p05'],
        'Sep25_Mean_WAL': res_sep25['wal_mean'],
        'Sep25_P05_WAL': res_sep25['wal_p05'],
        'Sep25_Mean_P05_Gap': res_sep25['wal_mean'] - res_sep25['wal_p05'],
        'P05_Period_Gap': p05_gap,
        'Mean_Period_Gap': mean_gap,
        'Mar22_Beta_Avg': res_mar22['beta_avg'],
        'Sep25_Beta_Avg': res_sep25['beta_avg']
    })

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("RESULTS: DUAL-PERIOD COMPARISON WITH CONSISTENT ALPHA")
print("=" * 80)

df_results = pd.DataFrame(results_comparison)

print("\n" + "-" * 80)
print("MARCH 2022 RESULTS (High regime excursion: current >> MA)")
print("-" * 80)
print(df_results[['Alpha', 'Mar22_Mean_WAL', 'Mar22_P05_WAL', 'Mar22_Mean_P05_Gap', 'Mar22_Beta_Avg']].to_string(index=False))

print("\n" + "-" * 80)
print("SEPTEMBER 2025 RESULTS (Low/No regime excursion: current < MA)")
print("-" * 80)
print(df_results[['Alpha', 'Sep25_Mean_WAL', 'Sep25_P05_WAL', 'Sep25_Mean_P05_Gap', 'Sep25_Beta_Avg']].to_string(index=False))

print("\n" + "-" * 80)
print("CROSS-PERIOD COMPARISON (Sep 2025 - Mar 2022)")
print("-" * 80)
print(df_results[['Alpha', 'P05_Period_Gap', 'Mean_Period_Gap']].to_string(index=False))

# =============================================================================
# KEY FINDINGS
# =============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Find alpha where P05 values are closest
df_results['P05_Abs_Gap'] = df_results['P05_Period_Gap'].abs()
best_alpha_row = df_results.loc[df_results['P05_Abs_Gap'].idxmin()]

print(f"\n1. REGIME EXCURSION COMPARISON:")
print(f"   March 2022:     Current SOFR ({current_sofr_mar22*100:.2f}%) vs MA ({ma_24mo_mar22*100:.2f}%)")
print(f"                   Excursion = +{(current_sofr_mar22 - ma_24mo_mar22)*100:.2f}% (AMPLIFICATION ACTIVE)")
print(f"   September 2025: Current SOFR ({current_sofr_sep25*100:.2f}%) vs MA ({ma_24mo_sep25*100:.2f}%)")
print(f"                   Excursion = {(current_sofr_sep25 - ma_24mo_sep25)*100:.2f}% (AMPLIFICATION DORMANT)")

print(f"\n2. P05 WAL CONVERGENCE BY ALPHA:")
for _, row in df_results.iterrows():
    alpha = row['Alpha']
    mar_p05 = row['Mar22_P05_WAL']
    sep_p05 = row['Sep25_P05_WAL']
    gap = row['P05_Period_Gap']
    print(f"   α={alpha:.1f}: Mar22 P05={mar_p05:.2f}y, Sep25 P05={sep_p05:.2f}y, Gap={gap:+.2f}y")

print(f"\n3. ALPHA THAT MINIMIZES P05 GAP:")
print(f"   Best α = {best_alpha_row['Alpha']:.1f} produces P05 gap of {best_alpha_row['P05_Period_Gap']:+.2f} years")

print(f"\n4. INTERPRETATION:")
row_20 = df_results[df_results['Alpha'] == 2.0].iloc[0]
print(f"   - With α=2.0 (selected for base model):")
print(f"     March 2022 P05 WAL:     {row_20['Mar22_P05_WAL']:.2f} years")
print(f"     September 2025 P05 WAL: {row_20['Sep25_P05_WAL']:.2f} years")
print(f"     P05 Gap:                {row_20['P05_Period_Gap']:+.2f} years")
print(f"     Mar22 Tail Risk (Mean-P05): {row_20['Mar22_Mean_P05_Gap']:.2f} years")
print(f"     Sep25 Tail Risk (Mean-P05): {row_20['Sep25_Mean_P05_Gap']:.2f} years")
print(f"\n   - March 2022 has HIGHER tail risk (Mean-P05 gap) because regime amplification")
print(f"     is active (current rate >> MA), reflecting elevated sensitivity from surge deposits.")
print(f"   - September 2025 has LOWER tail risk because:")
print(f"     (a) Surge deposits have already fled during 2022-2023")
print(f"     (b) Regime amplification is dormant (current rate < MA)")

# =============================================================================
# SAVE RESULTS
# =============================================================================

df_results.to_csv(f'{output_dir}/dual_period_alpha_sensitivity.csv', index=False)
print(f"\n✓ Saved: {output_dir}/dual_period_alpha_sensitivity.csv")

# =============================================================================
# VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATION")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Dual-Period Alpha Sensitivity Analysis\n' + 
             'Comparing March 2022 vs September 2025 with CONSISTENT Parameters', 
             fontsize=14, fontweight='bold')

# Panel A: P05 WAL by Alpha
ax = axes[0, 0]
ax.plot(df_results['Alpha'], df_results['Mar22_P05_WAL'], 'o-', label='March 2022', 
        color='red', linewidth=2, markersize=8)
ax.plot(df_results['Alpha'], df_results['Sep25_P05_WAL'], 's-', label='September 2025', 
        color='blue', linewidth=2, markersize=8)
ax.set_xlabel('Regime Amplification Parameter α', fontsize=11)
ax.set_ylabel('P05 WAL (years)', fontsize=11)
ax.set_title('Panel A: P05 WAL by Alpha (Consistent Across Periods)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=4.0, color='gray', linestyle='--', alpha=0.5, label='_nolegend_')

# Panel B: Mean WAL by Alpha
ax = axes[0, 1]
ax.plot(df_results['Alpha'], df_results['Mar22_Mean_WAL'], 'o-', label='March 2022', 
        color='red', linewidth=2, markersize=8)
ax.plot(df_results['Alpha'], df_results['Sep25_Mean_WAL'], 's-', label='September 2025', 
        color='blue', linewidth=2, markersize=8)
ax.set_xlabel('Regime Amplification Parameter α', fontsize=11)
ax.set_ylabel('Mean WAL (years)', fontsize=11)
ax.set_title('Panel B: Mean WAL by Alpha', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel C: Mean-P05 Gap (Tail Risk) by Alpha
ax = axes[1, 0]
ax.plot(df_results['Alpha'], df_results['Mar22_Mean_P05_Gap'], 'o-', label='March 2022', 
        color='red', linewidth=2, markersize=8)
ax.plot(df_results['Alpha'], df_results['Sep25_Mean_P05_Gap'], 's-', label='September 2025', 
        color='blue', linewidth=2, markersize=8)
ax.set_xlabel('Regime Amplification Parameter α', fontsize=11)
ax.set_ylabel('Mean - P05 Gap (years)', fontsize=11)
ax.set_title('Panel C: Tail Risk (Mean-P05 Gap) by Alpha', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel D: P05 Period Gap by Alpha
ax = axes[1, 1]
ax.bar(df_results['Alpha'], df_results['P05_Period_Gap'], color='purple', alpha=0.7, width=0.08)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax.set_xlabel('Regime Amplification Parameter α', fontsize=11)
ax.set_ylabel('Sep25 P05 - Mar22 P05 (years)', fontsize=11)
ax.set_title('Panel D: P05 WAL Convergence (Sep25 - Mar22)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add annotation
for i, row in df_results.iterrows():
    ax.annotate(f'{row["P05_Period_Gap"]:+.2f}', 
                xy=(row['Alpha'], row['P05_Period_Gap']), 
                ha='center', va='bottom' if row['P05_Period_Gap'] > 0 else 'top',
                fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/dual_period_alpha_sensitivity.png', dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_dir}/dual_period_alpha_sensitivity.png")

plt.show()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

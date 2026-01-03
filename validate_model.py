"""Quick validation script for model validation team."""
import pandas as pd

# Read the WAL comparison output from the model run
wal_df = pd.read_csv('model_outputs/06_wal_comparison.csv')

# Extract key metrics
mc_mean = wal_df[wal_df['Profile_Type'] == 'Mean (Total)']['WAL_Years'].values[0]
analytical = wal_df[wal_df['Profile_Type'] == 'Expected (Total)']['WAL_Years'].values[0]
gap = abs(mc_mean - analytical)

print("=" * 60)
print("MODEL VALIDATION - KEY METRICS")
print("=" * 60)
print(f"Monte Carlo Mean WAL:  {mc_mean:.2f} years")
print(f"Analytical WAL:        {analytical:.2f} years")
print(f"Methodological Gap:    {gap:.2f} years")
print("=" * 60)

# Verify against expected values
expected_mc = 4.85
expected_gap = 0.17
tolerance = 0.10

mc_pass = abs(mc_mean - expected_mc) < tolerance
gap_pass = abs(gap - expected_gap) < tolerance

print(f"\nVALIDATION CHECKS:")
print(f"  MC WAL ≈ {expected_mc}: {'✓ PASS' if mc_pass else '✗ FAIL'} (actual: {mc_mean:.2f})")
print(f"  Gap ≈ {expected_gap}: {'✓ PASS' if gap_pass else '✗ FAIL'} (actual: {gap:.2f})")

"""
Market Data Loader for SOFR OIS Curve and Volatility Surfaces
Loads data from Excel and prepares it for the comprehensive NMD model
"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

class MarketDataLoader:
    """Load and process SOFR market data from Excel"""
    
    def __init__(self, excel_path='SOFR_Market_Data_20250930.xlsx'):
        self.excel_path = excel_path
        self.ois_curve = None
        self.cap_vol_surface = None
        self.swaption_vol_surface = None
        
    def load_ois_curve(self):
        """Load SOFR OIS swap rates"""
        df = pd.read_excel(self.excel_path, sheet_name='SOFR_OIS_Curve')
        
        # Parse tenor to years
        tenors_years = []
        for term in df['Term']:
            term_str = str(term).strip().upper()
            if 'D' in term_str:  # Days
                days = int(term_str.replace('D', '').strip())
                tenors_years.append(days / 365.0)
            elif 'MO' in term_str:  # Months
                months = int(term_str.replace('MO', '').strip())
                tenors_years.append(months / 12.0)
            elif 'YR' in term_str or 'Y' in term_str:  # Years
                years = int(term_str.replace('YR', '').replace('Y', '').strip())
                tenors_years.append(float(years))
            else:
                raise ValueError(f"Cannot parse tenor: {term}")
        
        # Convert to arrays
        tenors_np = np.array(tenors_years)
        rates_np = df['Mid'].values
        
        # Add intermediate tenors to reduce interpolation gaps
        # Add 7Y (between 5Y and 10Y), 15Y (between 10Y and 20Y), 25Y (between 20Y and 30Y)
        intermediate_tenors = [7.0, 15.0, 25.0]
        intermediate_rates = []
        
        for t_interp in intermediate_tenors:
            # Linear interpolation
            rate_interp = np.interp(t_interp, tenors_np, rates_np)
            intermediate_rates.append(rate_interp)
        
        # Combine original and intermediate points
        all_tenors = np.concatenate([tenors_np, intermediate_tenors])
        all_rates = np.concatenate([rates_np, intermediate_rates])
        all_terms = np.concatenate([df['Term'].values, ['7 YR', '15 YR', '25 YR']])
        
        # Sort by tenor
        sort_idx = np.argsort(all_tenors)
        all_tenors = all_tenors[sort_idx]
        all_rates = all_rates[sort_idx]
        all_terms = all_terms[sort_idx]
        
        self.ois_curve = {
            'tenors_years': all_tenors,
            'swap_rates': all_rates,  # Already in decimal format (e.g., 0.04 = 4%)
            'terms': all_terms
        }
        
        return self.ois_curve
    
    def load_cap_volatilities(self):
        """Load cap volatility surface with strikes"""
        df = pd.read_excel(self.excel_path, sheet_name='Cap_Volatility')
        
        # First row contains strike levels
        strikes_row = df.iloc[0, 2:].values  # Skip first two columns
        strikes = []
        for s in strikes_row:
            if pd.notna(s) and isinstance(s, str) and '%' in s:
                strikes.append(float(s.replace('%', '')) / 100)  # Convert to decimal
        
        # Parse expiries and volatilities
        expiries = []
        expiries_years = []
        vol_matrix = []
        
        for idx in range(1, len(df)):
            expiry_str = str(df.iloc[idx, 0]).strip().upper()
            
            if pd.isna(expiry_str) or expiry_str == 'NAN':
                continue
                
            # Parse expiry to years
            if 'MO' in expiry_str or 'M' in expiry_str:
                months = int(expiry_str.replace('MO', '').replace('M', '').replace('YR', '').strip())
                expiry_years = months / 12.0
            elif 'YR' in expiry_str or 'Y' in expiry_str:
                years = int(expiry_str.replace('YR', '').replace('Y', '').strip())
                expiry_years = float(years)
            else:
                continue
            
            # Get volatilities for this expiry
            vols = df.iloc[idx, 2:2+len(strikes)].values
            vols = np.array([float(v) if pd.notna(v) else np.nan for v in vols])
            
            expiries.append(expiry_str)
            expiries_years.append(expiry_years)
            vol_matrix.append(vols)
        
        self.cap_vol_surface = {
            'expiries': np.array(expiries),
            'expiries_years': np.array(expiries_years),
            'strikes': np.array(strikes),
            'volatilities': np.array(vol_matrix),  # Already in decimal (e.g., 0.01 = 1% normal vol)
        }
        
        return self.cap_vol_surface
    
    def load_swaption_volatilities(self):
        """Load ATM swaption volatility surface"""
        df = pd.read_excel(self.excel_path, sheet_name='ATM_Swaption_Volatility')
        
        # First row contains tenors
        tenors_row = df.iloc[0, 2:].values
        tenors = []
        tenors_years = []
        for t in tenors_row:
            if pd.notna(t) and isinstance(t, str):
                t_str = str(t).strip().upper()
                if 'YR' in t_str or 'Y' in t_str:
                    years = int(t_str.replace('YR', '').replace('Y', '').strip())
                    tenors.append(t_str)
                    tenors_years.append(float(years))
        
        # Parse expiries and volatilities
        expiries = []
        expiries_years = []
        vol_matrix = []
        
        for idx in range(1, len(df)):
            expiry_str = str(df.iloc[idx, 0]).strip().upper()
            
            if pd.isna(expiry_str) or expiry_str == 'NAN':
                continue
            
            # Parse expiry to years
            if 'MO' in expiry_str or 'M' in expiry_str:
                months = int(expiry_str.replace('MO', '').replace('M', '').strip())
                expiry_years = months / 12.0
            elif 'YR' in expiry_str or 'Y' in expiry_str:
                years = int(expiry_str.replace('YR', '').replace('Y', '').strip())
                expiry_years = float(years)
            else:
                continue
            
            # Get volatilities for this expiry
            vols = df.iloc[idx, 2:2+len(tenors)].values
            vols = np.array([float(v) if pd.notna(v) else np.nan for v in vols])
            
            expiries.append(expiry_str)
            expiries_years.append(expiry_years)
            vol_matrix.append(vols)
        
        self.swaption_vol_surface = {
            'expiries': np.array(expiries),
            'expiries_years': np.array(expiries_years),
            'tenors': np.array(tenors),
            'tenors_years': np.array(tenors_years),
            'volatilities': np.array(vol_matrix),  # Already in decimal
        }
        
        return self.swaption_vol_surface
    
    def get_atm_cap_vols_for_model(self):
        """
        Extract ATM cap volatilities for various expiries
        Returns arrays suitable for the main model
        """
        if self.cap_vol_surface is None:
            self.load_cap_volatilities()
        
        # Find ATM strike (closest to 0%)
        strikes = self.cap_vol_surface['strikes']
        atm_idx = np.argmin(np.abs(strikes))
        
        # Extract ATM volatilities
        atm_vols = self.cap_vol_surface['volatilities'][:, atm_idx]
        
        return {
            'expiries_years': self.cap_vol_surface['expiries_years'],
            'atm_volatilities_bps': atm_vols * 10000,  # Convert to bps
            'atm_volatilities_decimal': atm_vols,  # Keep decimal for calculations
        }
    
    def load_all(self):
        """Load all market data"""
        print("Loading SOFR market data from Excel...")
        self.load_ois_curve()
        self.load_cap_volatilities()
        self.load_swaption_volatilities()
        print(f"✓ Loaded OIS curve: {len(self.ois_curve['tenors_years'])} tenors")
        print(f"✓ Loaded cap vols: {len(self.cap_vol_surface['expiries_years'])} expiries × {len(self.cap_vol_surface['strikes'])} strikes")
        print(f"✓ Loaded swaption vols: {len(self.swaption_vol_surface['expiries_years'])} expiries × {len(self.swaption_vol_surface['tenors_years'])} tenors")
        return self


# Test the loader
if __name__ == '__main__':
    loader = MarketDataLoader()
    loader.load_all()
    
    print("\n" + "="*80)
    print("OIS CURVE DATA:")
    print("="*80)
    for i, (tenor, rate, term) in enumerate(zip(loader.ois_curve['tenors_years'], 
                                                  loader.ois_curve['swap_rates'],
                                                  loader.ois_curve['terms'])):
        print(f"  {term:>6} ({tenor:6.3f}Y) = {rate*100:6.3f}%")
    
    print("\n" + "="*80)
    print("CAP VOLATILITY SURFACE (ATM):")
    print("="*80)
    atm_data = loader.get_atm_cap_vols_for_model()
    for expiry, vol_bps in zip(atm_data['expiries_years'], atm_data['atm_volatilities_bps']):
        print(f"  {expiry:5.2f}Y = {vol_bps:6.1f} bps")
    
    print("\n" + "="*80)
    print("SWAPTION VOLATILITY SURFACE (Sample - 1Y Tenor):")
    print("="*80)
    # Find 1Y tenor index
    tenor_1y_idx = np.where(loader.swaption_vol_surface['tenors_years'] == 1.0)[0]
    if len(tenor_1y_idx) > 0:
        idx = tenor_1y_idx[0]
        for expiry, vols in zip(loader.swaption_vol_surface['expiries_years'],
                                loader.swaption_vol_surface['volatilities']):
            vol = vols[idx]
            if not np.isnan(vol):
                print(f"  {expiry:5.2f}Y expiry = {vol*10000:6.1f} bps")

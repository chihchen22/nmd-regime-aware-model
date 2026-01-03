"""
SABR Volatility Interpolation for building a complete volatility cube
Interpolates cap volatilities across expiries and strikes using SABR model
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d


class SABRVolatilityCube:
    """
    Build a volatility cube using SABR interpolation
    Handles both expiry and strike dimensions
    """
    
    def __init__(self, cap_vol_surface):
        """
        Parameters:
        -----------
        cap_vol_surface : dict
            Contains 'expiries_years', 'strikes', 'volatilities' from market data
        """
        self.expiries_years = cap_vol_surface['expiries_years']
        self.strikes = cap_vol_surface['strikes']
        self.market_vols = cap_vol_surface['volatilities']
        
        # SABR parameters for each expiry
        self.sabr_params = {}
        
    @staticmethod
    def sabr_normal_vol(forward, strike, expiry, alpha, beta, rho, nu):
        """
        SABR normal (Bachelier) volatility formula
        Hagan et al. (2002) approximation for normal volatilities
        
        Parameters:
        -----------
        forward : float
            Forward rate (ATM)
        strike : float
            Strike rate
        expiry : float
            Time to expiry in years
        alpha : float
            Initial volatility parameter
        beta : float
            CEV parameter (typically 0 for normal, 0.5 for CIR, 1 for lognormal)
        rho : float
            Correlation between forward and volatility
        nu : float
            Volatility of volatility
        
        Returns:
        --------
        float : Normal volatility
        """
        if expiry <= 0:
            return alpha
        
        # For normal SABR, beta should be close to 0
        # Using approximate formula for beta=0 (normal SABR)
        
        # ATM case
        if abs(strike - forward) < 1e-8:
            atm_vol = alpha * (1 + expiry * (
                (1 - beta)**2 * alpha**2 / 24 +
                rho * beta * nu * alpha / 4 +
                (2 - 3*rho**2) * nu**2 / 24
            ))
            return atm_vol
        
        # Off-ATM case - simplified normal SABR formula
        z = (nu / alpha) * (forward - strike)
        
        if abs(z) < 1e-4:
            x_z = 1.0
        else:
            x_z = z / np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho) / (1 - rho))
        
        # Simplified for beta close to 0
        vol = alpha * x_z * (1 + expiry * (
            (1 - beta)**2 * alpha**2 / 24 +
            rho * beta * nu * alpha / 4 +
            (2 - 3*rho**2) * nu**2 / 24
        ))
        
        return max(vol, 1e-6)  # Floor for numerical stability
    
    def calibrate_sabr_slice(self, expiry_idx, forward_rate):
        """
        Calibrate SABR parameters for a single expiry slice
        
        Parameters:
        -----------
        expiry_idx : int
            Index of expiry to calibrate
        forward_rate : float
            Forward rate for this expiry (used as ATM reference)
        
        Returns:
        --------
        dict : Calibrated SABR parameters {alpha, beta, rho, nu}
        """
        expiry = self.expiries_years[expiry_idx]
        market_vols_slice = self.market_vols[expiry_idx]
        valid_strikes = ~np.isnan(market_vols_slice)
        
        if not np.any(valid_strikes):
            # No valid data - return default parameters
            return {'alpha': 0.01, 'beta': 0.0, 'rho': 0.0, 'nu': 0.3}
        
        strikes_valid = self.strikes[valid_strikes]
        vols_valid = market_vols_slice[valid_strikes]
        
        # Find ATM volatility
        atm_idx = np.argmin(np.abs(strikes_valid - forward_rate))
        atm_vol = vols_valid[atm_idx]
        
        # Objective function for SABR calibration
        def objective(params):
            alpha, rho, nu = params
            beta = 0.0  # Fix beta=0 for normal SABR
            
            errors = []
            for strike, market_vol in zip(strikes_valid, vols_valid):
                try:
                    sabr_vol = self.sabr_normal_vol(
                        forward_rate, strike, expiry, alpha, beta, rho, nu
                    )
                    errors.append((sabr_vol - market_vol)**2)
                except:
                    errors.append(100.0)  # Penalty for invalid parameters
            
            return np.sum(errors)
        
        # Initial guess
        x0 = [atm_vol, 0.0, 0.5]  # alpha=ATM vol, rho=0, nu=0.5
        
        # Bounds: alpha > 0, -1 < rho < 1, nu > 0
        bounds = [(atm_vol*0.1, atm_vol*5.0), (-0.99, 0.99), (0.01, 2.0)]
        
        try:
            # Use differential evolution for robustness
            result = differential_evolution(
                objective, bounds, maxiter=100, seed=42, polish=True
            )
            alpha, rho, nu = result.x
        except:
            # Fallback to simple initial guess
            alpha, rho, nu = atm_vol, 0.0, 0.5
        
        return {'alpha': alpha, 'beta': 0.0, 'rho': rho, 'nu': nu}
    
    def calibrate_all_expiries(self, forward_curve_func):
        """
        Calibrate SABR parameters for all expiries
        
        Parameters:
        -----------
        forward_curve_func : callable
            Function that takes expiry (years) and returns forward rate
        """
        print("\nCalibrating SABR volatility cube...")
        
        for i, expiry in enumerate(self.expiries_years):
            forward = forward_curve_func(expiry)
            params = self.calibrate_sabr_slice(i, forward)
            self.sabr_params[expiry] = params
            
            print(f"  {expiry:5.2f}Y: α={params['alpha']*10000:5.1f}bps, "
                  f"ρ={params['rho']:+5.2f}, ν={params['nu']:5.2f}")
        
        print("✓ SABR calibration complete")
    
    def get_volatility(self, expiry, strike, forward_rate):
        """
        Get interpolated volatility for any expiry and strike
        
        Parameters:
        -----------
        expiry : float
            Time to expiry in years
        strike : float
            Strike rate (decimal, e.g., 0.04 = 4%)
        forward_rate : float
            Forward rate at this expiry
        
        Returns:
        --------
        float : Normal volatility (decimal)
        """
        # Interpolate SABR parameters across expiries
        if expiry <= self.expiries_years[0]:
            params = self.sabr_params[self.expiries_years[0]]
        elif expiry >= self.expiries_years[-1]:
            params = self.sabr_params[self.expiries_years[-1]]
        else:
            # Linear interpolation of SABR parameters
            idx_upper = np.searchsorted(self.expiries_years, expiry)
            idx_lower = idx_upper - 1
            
            t_lower = self.expiries_years[idx_lower]
            t_upper = self.expiries_years[idx_upper]
            weight = (expiry - t_lower) / (t_upper - t_lower)
            
            params_lower = self.sabr_params[t_lower]
            params_upper = self.sabr_params[t_upper]
            
            params = {
                'alpha': params_lower['alpha'] * (1-weight) + params_upper['alpha'] * weight,
                'beta': 0.0,
                'rho': params_lower['rho'] * (1-weight) + params_upper['rho'] * weight,
                'nu': params_lower['nu'] * (1-weight) + params_upper['nu'] * weight,
            }
        
        # Calculate SABR volatility
        vol = self.sabr_normal_vol(
            forward_rate, strike, expiry,
            params['alpha'], params['beta'], params['rho'], params['nu']
        )
        
        return vol
    
    def get_atm_vol_term_structure(self, expiries_years, forward_curve_func):
        """
        Get ATM volatility term structure for given expiries
        
        Parameters:
        -----------
        expiries_years : array
            Array of expiries in years
        forward_curve_func : callable
            Function to get forward rates
        
        Returns:
        --------
        array : ATM volatilities
        """
        atm_vols = []
        for expiry in expiries_years:
            forward = forward_curve_func(expiry)
            vol = self.get_volatility(expiry, forward, forward)
            atm_vols.append(vol)
        
        return np.array(atm_vols)


# Test the SABR calibration
if __name__ == '__main__':
    from market_data_loader import MarketDataLoader
    
    # Load market data
    loader = MarketDataLoader()
    loader.load_all()
    
    # Create SABR cube
    sabr = SABRVolatilityCube(loader.cap_vol_surface)
    
    # Create simple forward curve function for testing
    def forward_func(t):
        # Simple interpolation of OIS rates as proxy
        return np.interp(t, loader.ois_curve['tenors_years'], 
                        loader.ois_curve['swap_rates'])
    
    # Calibrate
    sabr.calibrate_all_expiries(forward_func)
    
    # Test interpolation at intermediate expiries
    print("\n" + "="*80)
    print("TESTING SABR INTERPOLATION:")
    print("="*80)
    test_expiries = [0.5, 1.5, 3.5, 7.5, 15.0]
    for exp in test_expiries:
        fwd = forward_func(exp)
        atm_vol = sabr.get_volatility(exp, fwd, fwd)
        print(f"  {exp:5.2f}Y: Forward={fwd*100:5.2f}%, ATM Vol={atm_vol*10000:6.1f} bps")

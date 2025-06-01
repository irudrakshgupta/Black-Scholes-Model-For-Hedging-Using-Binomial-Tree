import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch import arch_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class AdvancedVolatility:
    def __init__(self):
        self.garch_model = None
        self.vol_surface = None
        self.historical_data = None
        
    def fit_garch(self, returns, p=1, q=1):
        """
        Fit GARCH(p,q) model to return series
        """
        self.garch_model = arch_model(returns, vol='Garch', p=p, q=q)
        return self.garch_model.fit(disp='off')
        
    def forecast_volatility(self, horizon=10):
        """
        Generate volatility forecasts using fitted GARCH model
        """
        if self.garch_model is None:
            raise ValueError("Must fit GARCH model first")
        forecast = self.garch_model.forecast(horizon=horizon)
        return np.sqrt(forecast.variance.values[-1])
    
    def calibrate_vol_surface(self, strikes, maturities, market_ivs):
        """
        Calibrate volatility surface using Gaussian Process regression
        """
        X = np.column_stack([strikes, maturities])
        kernel = RBF() + WhiteKernel()
        self.vol_surface = GaussianProcessRegressor(kernel=kernel)
        self.vol_surface.fit(X, market_ivs)
        return self.vol_surface
    
    def get_implied_vol(self, strike, maturity):
        """
        Get implied volatility from calibrated surface
        """
        if self.vol_surface is None:
            raise ValueError("Must calibrate volatility surface first")
        return self.vol_surface.predict([[strike, maturity]])[0]
    
    def compute_vol_cone(self, prices, windows=[5, 10, 21, 63, 126, 252]):
        """
        Compute volatility cone using historical data
        """
        returns = np.log(prices[1:] / prices[:-1])
        vol_cone = {}
        
        for window in windows:
            rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
            vol_cone[window] = {
                'min': rolling_vol.min(),
                'max': rolling_vol.max(),
                'mean': rolling_vol.mean(),
                'current': rolling_vol.iloc[-1]
            }
        return vol_cone
    
    def term_structure(self, maturities, strikes, market_ivs):
        """
        Generate volatility term structure
        """
        term_structure = {}
        for maturity in np.unique(maturities):
            mask = maturities == maturity
            term_structure[maturity] = {
                'strikes': strikes[mask],
                'ivs': market_ivs[mask]
            }
        return term_structure
    
    def compare_historical_vol(self, prices, window=30):
        """
        Compare historical and implied volatilities
        """
        returns = np.log(prices[1:] / prices[:-1])
        hist_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252)
        
        if self.vol_surface is not None:
            atm_iv = self.get_implied_vol(prices[-1], 30/252)  # 30-day ATM IV
            return {
                'historical_vol': hist_vol.iloc[-1],
                'implied_vol': atm_iv,
                'vol_premium': atm_iv - hist_vol.iloc[-1]
            }
        return {'historical_vol': hist_vol.iloc[-1]} 
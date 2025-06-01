"""
Hedge Ratio Calculator Module
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats
from scipy.optimize import minimize

class HedgeRatioCalculator:
    @staticmethod
    def minimum_variance_hedge_ratio(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate minimum variance hedge ratio using regression.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
            
        Returns:
        --------
        Tuple[float, float]
            Hedge ratio and R-squared value
        """
        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            returns_hedge,
            returns_asset
        )
        
        return slope, r_value**2

    @staticmethod
    def rolling_hedge_ratio(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray,
        window: int = 60
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rolling hedge ratios.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
        window : int
            Rolling window size
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Rolling hedge ratios and R-squared values
        """
        hedge_ratios = np.zeros(len(returns_asset) - window + 1)
        r_squared = np.zeros(len(returns_asset) - window + 1)
        
        for i in range(len(hedge_ratios)):
            hr, r2 = HedgeRatioCalculator.minimum_variance_hedge_ratio(
                returns_asset[i:i+window],
                returns_hedge[i:i+window]
            )
            hedge_ratios[i] = hr
            r_squared[i] = r2
            
        return hedge_ratios, r_squared

    @staticmethod
    def optimal_hedge_ratio(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray,
        risk_aversion: float = 1.0,
        transaction_cost: float = 0.0001
    ) -> float:
        """
        Calculate optimal hedge ratio considering risk-return tradeoff.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
        risk_aversion : float
            Risk aversion parameter
        transaction_cost : float
            Transaction cost as fraction of trade value
            
        Returns:
        --------
        float
            Optimal hedge ratio
        """
        def objective(h):
            # Portfolio returns
            portfolio_returns = returns_asset - h * returns_hedge
            
            # Mean and variance
            mean_return = np.mean(portfolio_returns)
            variance = np.var(portfolio_returns)
            
            # Utility function: mean - (risk_aversion/2) * variance - transaction_cost * |h|
            utility = mean_return - (risk_aversion/2) * variance - transaction_cost * abs(h)
            return -utility  # Minimize negative utility
        
        # Optimize
        result = minimize(objective, x0=0.0, method='BFGS')
        return result.x[0]

    @staticmethod
    def cross_hedge_ratio(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray,
        correlation_threshold: float = 0.5
    ) -> Optional[float]:
        """
        Calculate cross-hedge ratio if correlation is sufficient.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
        correlation_threshold : float
            Minimum correlation required for cross-hedging
            
        Returns:
        --------
        Optional[float]
            Cross-hedge ratio if correlation is sufficient, None otherwise
        """
        correlation = np.corrcoef(returns_asset, returns_hedge)[0,1]
        
        if abs(correlation) >= correlation_threshold:
            std_asset = np.std(returns_asset)
            std_hedge = np.std(returns_hedge)
            return correlation * (std_asset / std_hedge)
        
        return None

    @staticmethod
    def hedge_effectiveness(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray,
        hedge_ratio: float
    ) -> Tuple[float, float, float]:
        """
        Calculate hedge effectiveness metrics.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
        hedge_ratio : float
            Applied hedge ratio
            
        Returns:
        --------
        Tuple[float, float, float]
            Variance reduction, correlation, and hedge effectiveness ratio
        """
        # Unhedged variance
        var_unhedged = np.var(returns_asset)
        
        # Hedged portfolio returns and variance
        returns_hedged = returns_asset - hedge_ratio * returns_hedge
        var_hedged = np.var(returns_hedged)
        
        # Calculate metrics
        variance_reduction = 1 - (var_hedged / var_unhedged)
        correlation = np.corrcoef(returns_asset, returns_hedge)[0,1]
        hedge_effectiveness_ratio = correlation**2
        
        return variance_reduction, correlation, hedge_effectiveness_ratio

    @staticmethod
    def dynamic_hedge_ratio(
        returns_asset: np.ndarray,
        returns_hedge: np.ndarray,
        lambda_param: float = 0.94
    ) -> np.ndarray:
        """
        Calculate dynamic hedge ratio using EWMA covariance.
        
        Parameters:
        -----------
        returns_asset : array-like
            Returns of the asset to be hedged
        returns_hedge : array-like
            Returns of the hedging instrument
        lambda_param : float
            EWMA decay factor
            
        Returns:
        --------
        array-like
            Dynamic hedge ratios
        """
        # Initialize covariance and variance series
        cov = np.zeros(len(returns_asset))
        var_hedge = np.zeros(len(returns_asset))
        
        # Initialize with sample estimates
        cov[0] = np.cov(returns_asset[:10], returns_hedge[:10])[0,1]
        var_hedge[0] = np.var(returns_hedge[:10])
        
        # Update EWMA estimates
        for t in range(1, len(returns_asset)):
            cov[t] = lambda_param * cov[t-1] + (1 - lambda_param) * returns_asset[t] * returns_hedge[t]
            var_hedge[t] = lambda_param * var_hedge[t-1] + (1 - lambda_param) * returns_hedge[t]**2
        
        # Calculate dynamic hedge ratios
        hedge_ratios = cov / var_hedge
        
        return hedge_ratios 
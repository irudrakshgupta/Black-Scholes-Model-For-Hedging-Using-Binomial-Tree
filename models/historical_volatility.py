"""
Historical Volatility Calculator Module
"""

import numpy as np
from typing import List, Tuple, Optional
import pandas as pd

class HistoricalVolatility:
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns from price series."""
        return np.log(prices[1:] / prices[:-1])

    @staticmethod
    def simple_volatility(returns: np.ndarray, annualization: float = 252) -> float:
        """Calculate simple volatility from returns."""
        return np.std(returns) * np.sqrt(annualization)

    @staticmethod
    def rolling_volatility(
        returns: np.ndarray,
        window: int = 30,
        annualization: float = 252
    ) -> np.ndarray:
        """Calculate rolling volatility."""
        return (
            pd.Series(returns)
            .rolling(window=window)
            .std()
            .values * np.sqrt(annualization)
        )

    @staticmethod
    def ewma_volatility(
        returns: np.ndarray,
        lambda_param: float = 0.94,
        annualization: float = 252
    ) -> np.ndarray:
        """Calculate EWMA volatility."""
        vol = np.zeros_like(returns)
        vol[0] = returns[0]**2
        
        for t in range(1, len(returns)):
            vol[t] = lambda_param * vol[t-1] + (1 - lambda_param) * returns[t]**2
            
        return np.sqrt(vol * annualization)

    @staticmethod
    def parkinson_volatility(
        high: np.ndarray,
        low: np.ndarray,
        annualization: float = 252
    ) -> float:
        """Calculate Parkinson volatility using high-low prices."""
        return np.sqrt(
            1 / (4 * np.log(2)) *
            np.mean((np.log(high/low))**2) *
            annualization
        )

    @staticmethod
    def garman_klass_volatility(
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        annualization: float = 252
    ) -> float:
        """Calculate Garman-Klass volatility."""
        return np.sqrt(
            (0.5 * np.mean(np.log(high/low)**2) -
             (2 * np.log(2) - 1) * np.mean(np.log(close/open_)**2)) *
            annualization
        )

    @staticmethod
    def regime_detection(
        volatilities: np.ndarray,
        n_regimes: int = 2
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Detect volatility regimes using simple threshold method.
        Returns regime labels and threshold values.
        """
        thresholds = np.percentile(volatilities, np.linspace(0, 100, n_regimes+1)[1:-1])
        regimes = np.zeros_like(volatilities)
        
        for i in range(n_regimes):
            if i == 0:
                mask = volatilities <= thresholds[0]
            elif i == n_regimes - 1:
                mask = volatilities > thresholds[-1]
            else:
                mask = (volatilities > thresholds[i-1]) & (volatilities <= thresholds[i])
            regimes[mask] = i
            
        return regimes, thresholds.tolist()

    @staticmethod
    def forecast_volatility(
        volatilities: np.ndarray,
        horizon: int = 5,
        method: str = 'ewma',
        lambda_param: float = 0.94
    ) -> np.ndarray:
        """
        Forecast volatility using various methods.
        
        Parameters:
        -----------
        volatilities : array-like
            Historical volatility series
        horizon : int
            Forecast horizon
        method : str
            Forecasting method ('ewma' or 'simple')
        lambda_param : float
            EWMA decay factor
            
        Returns:
        --------
        array-like
            Forecasted volatilities
        """
        if method == 'ewma':
            forecast = np.zeros(horizon)
            last_vol = volatilities[-1]
            
            for i in range(horizon):
                forecast[i] = np.sqrt(
                    lambda_param * last_vol**2 +
                    (1 - lambda_param) * np.mean(volatilities[-10:])**2
                )
                last_vol = forecast[i]
                
            return forecast
        
        elif method == 'simple':
            return np.repeat(np.mean(volatilities[-10:]), horizon)
        
        else:
            raise ValueError("Method must be either 'ewma' or 'simple'") 
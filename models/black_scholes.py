"""
Black-Scholes Option Pricing Model
"""

import numpy as np
from dataclasses import dataclass
from scipy.stats import norm

@dataclass
class OptionResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class BlackScholes:
    def __init__(self):
        self.N = norm.cdf   
        self.n = norm.pdf

    def d1(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))

    def d2(self, d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter."""
        return d1 - sigma*np.sqrt(T)

    def price_and_greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: str = "call"
    ) -> OptionResult:
        """
        Calculate option price and Greeks.
        
        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free rate (annual)
        sigma : float
            Volatility (annual)
        option_type : str
            Type of option ("call" or "put")
            
        Returns:
        --------
        OptionResult
            Dataclass containing price and Greeks
        """
        d1 = self.d1(S, K, T, r, sigma)
        d2 = self.d2(d1, sigma, T)
        
        # Calculate option price
        if option_type == "call":
            price = S*self.N(d1) - K*np.exp(-r*T)*self.N(d2)
            delta = self.N(d1)
        else:  # put
            price = K*np.exp(-r*T)*self.N(-d2) - S*self.N(-d1)
            delta = -self.N(-d1)
        
        # Calculate Greeks
        gamma = self.n(d1)/(S*sigma*np.sqrt(T))
        
        if option_type == "call":
            theta = (-S*self.n(d1)*sigma/(2*np.sqrt(T)) - 
                    r*K*np.exp(-r*T)*self.N(d2))
        else:
            theta = (-S*self.n(d1)*sigma/(2*np.sqrt(T)) + 
                    r*K*np.exp(-r*T)*self.N(-d2))
        
        vega = S*np.sqrt(T)*self.n(d1)
        
        if option_type == "call":
            rho = K*T*np.exp(-r*T)*self.N(d2)
        else:
            rho = -K*T*np.exp(-r*T)*self.N(-d2)
        
        return OptionResult(
            price=price,
            delta=delta,
            gamma=gamma,
            theta=theta/365,  # Convert to daily theta
            vega=vega/100,    # Convert to 1% vol change
            rho=rho/100       # Convert to 1% rate change
        )

    def implied_volatility(
        self,
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = "call",
        max_iter: int = 100,
        tolerance: float = 1e-5
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Parameters:
        -----------
        price : float
            Market price of option
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free rate (annual)
        option_type : str
            Type of option ("call" or "put")
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
            
        Returns:
        --------
        float
            Implied volatility
        """
        sigma = 0.3  # Initial guess
        
        for i in range(max_iter):
            result = self.price_and_greeks(S, K, T, r, sigma, option_type)
            diff = result.price - price
            
            if abs(diff) < tolerance:
                return sigma
            
            sigma = sigma - diff/result.vega
            
            if sigma <= 0:
                sigma = 0.0001
            
        return sigma 
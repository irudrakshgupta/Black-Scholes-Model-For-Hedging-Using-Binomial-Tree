import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class HestonParameters:
    kappa: float  # Mean reversion speed
    theta: float  # Long-term variance
    sigma: float  # Volatility of variance
    rho: float    # Correlation
    v0: float     # Initial variance

@dataclass
class JumpDiffusionParameters:
    lambda_: float  # Jump intensity
    mu_j: float    # Jump size mean
    sigma_j: float # Jump size volatility

class AdvancedPricing:
    def __init__(self):
        pass
        
    def heston_characteristic_fn(self, phi: float, S: float, K: float, T: float, 
                               r: float, params: HestonParameters) -> complex:
        """
        Compute Heston characteristic function
        """
        kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
        
        a = kappa * theta
        b = kappa - rho * sigma * phi * 1j
        
        d = np.sqrt(b**2 + sigma**2 * phi * 1j * (phi * 1j - 1))
        g = (b - d) / (b + d)
        
        exp1 = np.exp(r * phi * 1j * T)
        exp2 = np.exp(a * T * (b - d) / sigma**2)
        exp3 = np.exp(v0 * (b - d) * (1 - np.exp(-d * T)) / (sigma**2 * (1 - g * np.exp(-d * T))))
        
        return exp1 * exp2 * exp3
        
    def heston_price(self, S: float, K: float, T: float, r: float, 
                    params: HestonParameters, option_type: str = 'call') -> float:
        """
        Price European options using Heston model
        """
        def integrand(phi: float, j: int) -> float:
            if j == 1:
                return np.real(np.exp(-1j * phi * np.log(K)) * 
                             self.heston_characteristic_fn(phi - 1j, S, K, T, r, params) /
                             (1j * phi))
            else:
                return np.real(np.exp(-1j * phi * np.log(K)) * 
                             self.heston_characteristic_fn(phi, S, K, T, r, params) /
                             (1j * phi))
        
        # Compute probabilities P1 and P2
        P1 = 0.5 + 1/np.pi * quad(lambda x: integrand(x, 1), 0, np.inf)[0]
        P2 = 0.5 + 1/np.pi * quad(lambda x: integrand(x, 2), 0, np.inf)[0]
        
        # Calculate call price
        call = S * P1 - K * np.exp(-r * T) * P2
        
        if option_type == 'call':
            return call
        else:
            return call - S + K * np.exp(-r * T)
            
    def merton_jump_diffusion(self, S: float, K: float, T: float, r: float, sigma: float,
                            params: JumpDiffusionParameters, option_type: str = 'call') -> float:
        """
        Price European options using Merton jump-diffusion model
        """
        lambda_, mu_j, sigma_j = params.lambda_, params.mu_j, params.sigma_j
        
        # Adjust drift for jump component
        adjusted_r = r - lambda_ * (np.exp(mu_j + sigma_j**2/2) - 1)
        
        price = 0
        for n in range(20):  # Truncate infinite sum
            # Probability of n jumps
            p_n = np.exp(-lambda_ * T) * (lambda_ * T)**n / np.math.factorial(n)
            
            # Total volatility including jumps
            total_sigma = np.sqrt(sigma**2 + n * sigma_j**2/T)
            
            # Black-Scholes price with adjusted parameters
            adjusted_r_n = adjusted_r + n * mu_j/T
            d1 = (np.log(S/K) + (adjusted_r_n + total_sigma**2/2) * T) / (total_sigma * np.sqrt(T))
            d2 = d1 - total_sigma * np.sqrt(T)
            
            if option_type == 'call':
                bs_price = S * np.exp((adjusted_r_n - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                bs_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((adjusted_r_n - r) * T) * norm.cdf(-d1)
                
            price += p_n * bs_price
            
        return price
        
    def barrier_option(self, S: float, K: float, B: float, T: float, r: float, sigma: float,
                      barrier_type: str, option_type: str = 'call') -> float:
        """
        Price barrier options using analytical formulas
        Supports: up-and-out, down-and-out, up-and-in, down-and-in
        """
        def d1(x1: float, x2: float) -> float:
            return (np.log(x1/x2) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
            
        def d2(x1: float, x2: float) -> float:
            return d1(x1, x2) - sigma * np.sqrt(T)
            
        lambda_ = (r + sigma**2/2) / sigma**2
        y = np.log(B**2/(S*K)) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
        
        if barrier_type == 'down-and-out':
            if option_type == 'call':
                if B >= K:
                    return (S * norm.cdf(d1(S, K)) - K * np.exp(-r * T) * norm.cdf(d2(S, K)) -
                           S * (B/S)**(2*lambda_) * (norm.cdf(d1(B**2/S, K)) - 
                           K * np.exp(-r * T) * norm.cdf(d2(B**2/S, K))))
                else:
                    return 0
            else:  # put
                if B >= K:
                    return (K * np.exp(-r * T) * norm.cdf(-d2(S, K)) - S * norm.cdf(-d1(S, K)) -
                           S * (B/S)**(2*lambda_) * (norm.cdf(-d1(B**2/S, K)) - 
                           K * np.exp(-r * T) * norm.cdf(-d2(B**2/S, K))))
                else:
                    return 0
                    
        elif barrier_type == 'up-and-out':
            if option_type == 'call':
                if B <= K:
                    return 0
                else:
                    return (S * norm.cdf(d1(S, K)) - K * np.exp(-r * T) * norm.cdf(d2(S, K)) -
                           S * (B/S)**(2*lambda_) * (norm.cdf(d1(B**2/S, K)) - 
                           K * np.exp(-r * T) * norm.cdf(d2(B**2/S, K))))
            else:  # put
                if B <= K:
                    return 0
                else:
                    return (K * np.exp(-r * T) * norm.cdf(-d2(S, K)) - S * norm.cdf(-d1(S, K)) -
                           S * (B/S)**(2*lambda_) * (norm.cdf(-d1(B**2/S, K)) - 
                           K * np.exp(-r * T) * norm.cdf(-d2(B**2/S, K))))
                           
        elif barrier_type == 'down-and-in':
            vanilla = S * norm.cdf(d1(S, K)) - K * np.exp(-r * T) * norm.cdf(d2(S, K))
            if option_type == 'call':
                return vanilla - self.barrier_option(S, K, B, T, r, sigma, 'down-and-out', 'call')
            else:
                return vanilla - self.barrier_option(S, K, B, T, r, sigma, 'down-and-out', 'put')
                
        elif barrier_type == 'up-and-in':
            vanilla = S * norm.cdf(d1(S, K)) - K * np.exp(-r * T) * norm.cdf(d2(S, K))
            if option_type == 'call':
                return vanilla - self.barrier_option(S, K, B, T, r, sigma, 'up-and-out', 'call')
            else:
                return vanilla - self.barrier_option(S, K, B, T, r, sigma, 'up-and-out', 'put')
                
    def local_volatility(self, S: float, K: float, T: float, r: float, 
                        local_vol_surface: callable, option_type: str = 'call',
                        n_steps: int = 1000) -> float:
        """
        Price options using local volatility model via finite difference method
        """
        # Set up grid
        dt = T/n_steps
        n_space = 100
        S_max = 2 * S
        dS = S_max/n_space
        
        # Initialize grid
        grid = np.zeros((n_space+1, n_steps+1))
        S_values = np.linspace(0, S_max, n_space+1)
        
        # Terminal condition
        if option_type == 'call':
            grid[:,-1] = np.maximum(S_values - K, 0)
        else:
            grid[:,-1] = np.maximum(K - S_values, 0)
            
        # Backward induction
        for t in range(n_steps-1, -1, -1):
            for i in range(1, n_space):
                vol = local_vol_surface(S_values[i], (n_steps-t)*dt)
                
                # Finite difference coefficients
                a = 0.5 * dt * (r * i * dS + vol**2 * i**2 * dS**2)
                b = 1 - dt * (vol**2 * i**2 + r)
                c = 0.5 * dt * (-r * i * dS + vol**2 * i**2 * dS**2)
                
                grid[i,t] = a * grid[i+1,t+1] + b * grid[i,t+1] + c * grid[i-1,t+1]
                
        # Interpolate to get price at current spot
        idx = int(S/dS)
        w = (S - idx*dS)/dS
        price = w * grid[idx+1,0] + (1-w) * grid[idx,0]
        
        return price 
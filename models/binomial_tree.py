"""
Cox-Ross-Rubinstein Binomial Tree Model for option pricing.
"""

import numpy as np
from typing import Literal, Tuple
from dataclasses import dataclass

@dataclass
class BinomialTreeResult:
    price: float
    delta: float
    gamma: float
    theta: float

class BinomialTree:
    def __init__(self, steps: int = 100):
        """
        Initialize Binomial Tree calculator.

        Parameters:
        -----------
        steps : int
            Number of time steps in the tree
        """
        self.steps = steps

    def _calculate_parameters(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float
    ) -> Tuple[float, float, float, float]:
        """Calculate up, down, and probability parameters for the tree."""
        dt = T / self.steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        return dt, u, d, p

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal['call', 'put'] = 'call',
        exercise: Literal['european', 'american'] = 'european'
    ) -> BinomialTreeResult:
        """
        Calculate option price and Greeks using binomial tree model.

        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility (annual)
        option_type : str
            Type of option ('call' or 'put')
        exercise : str
            Exercise style ('european' or 'american')

        Returns:
        --------
        BinomialTreeResult
            Dataclass containing price and Greeks
        """
        dt, u, d, p = self._calculate_parameters(S, T, r, sigma)
        
        # Initialize stock price tree
        stock_tree = np.zeros((self.steps + 1, self.steps + 1))
        for i in range(self.steps + 1):
            for j in range(i + 1):
                stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

        # Initialize option value tree
        option_tree = np.zeros((self.steps + 1, self.steps + 1))
        
        # Terminal payoff
        for i in range(self.steps + 1):
            if option_type == 'call':
                option_tree[i, self.steps] = max(0, stock_tree[i, self.steps] - K)
            else:  # put
                option_tree[i, self.steps] = max(0, K - stock_tree[i, self.steps])

        # Backward induction
        for i in range(self.steps - 1, -1, -1):
            for j in range(i + 1):
                # Expected value
                expected = (p * option_tree[j, i + 1] + 
                          (1 - p) * option_tree[j + 1, i + 1]) * np.exp(-r * dt)
                
                if exercise == 'american':
                    # Early exercise value
                    if option_type == 'call':
                        intrinsic = max(0, stock_tree[j, i] - K)
                    else:  # put
                        intrinsic = max(0, K - stock_tree[j, i])
                    option_tree[j, i] = max(expected, intrinsic)
                else:
                    option_tree[j, i] = expected

        # Calculate Greeks
        delta = (option_tree[0, 1] - option_tree[1, 1]) / (stock_tree[0, 1] - stock_tree[1, 1])
        
        gamma = ((option_tree[0, 2] - option_tree[1, 2]) / (stock_tree[0, 2] - stock_tree[1, 2]) -
                (option_tree[1, 2] - option_tree[2, 2]) / (stock_tree[1, 2] - stock_tree[2, 2])) / (
                    (stock_tree[0, 1] - stock_tree[1, 1]) / 2)
        
        theta = (option_tree[1, 2] - option_tree[0, 0]) / (2 * dt)

        return BinomialTreeResult(
            price=option_tree[0, 0],
            delta=delta,
            gamma=gamma,
            theta=theta
        )

    def price_path(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        num_paths: int = 1,
        option_type: Literal['call', 'put'] = 'call'
    ) -> np.ndarray:
        """
        Generate price paths using the binomial model parameters.

        Parameters:
        -----------
        S : float
            Current stock price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate (annual)
        sigma : float
            Volatility (annual)
        num_paths : int
            Number of paths to generate
        option_type : str
            Type of option ('call' or 'put')

        Returns:
        --------
        np.ndarray
            Array of shape (num_paths, steps + 1) containing price paths
        """
        dt, u, d, _ = self._calculate_parameters(S, T, r, sigma)
        
        paths = np.zeros((num_paths, self.steps + 1))
        paths[:, 0] = S
        
        for i in range(num_paths):
            path = [S]
            for _ in range(self.steps):
                if np.random.random() < 0.5:
                    path.append(path[-1] * u)
                else:
                    path.append(path[-1] * d)
            paths[i] = path
            
        return paths 
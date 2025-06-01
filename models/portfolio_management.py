import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PortfolioPosition:
    asset_id: str
    quantity: float
    current_price: float
    target_weight: float
    volatility: float
    beta: float

class PortfolioManager:
    def __init__(self):
        self.positions = {}
        self.correlation_matrix = None
        self.risk_factors = None
        
    def set_positions(self, positions: Dict[str, PortfolioPosition]):
        """
        Set portfolio positions
        """
        self.positions = positions
        
    def calculate_portfolio_value(self) -> float:
        """
        Calculate total portfolio value
        """
        return sum(pos.quantity * pos.current_price for pos in self.positions.values())
        
    def calculate_current_weights(self) -> Dict[str, float]:
        """
        Calculate current portfolio weights
        """
        total_value = self.calculate_portfolio_value()
        return {
            asset_id: (pos.quantity * pos.current_price) / total_value
            for asset_id, pos in self.positions.items()
        }
        
    def generate_rebalancing_trades(self, tolerance: float = 0.01) -> Dict[str, float]:
        """
        Generate trades needed to rebalance portfolio to target weights
        """
        current_weights = self.calculate_current_weights()
        total_value = self.calculate_portfolio_value()
        
        trades = {}
        for asset_id, pos in self.positions.items():
            weight_diff = pos.target_weight - current_weights[asset_id]
            if abs(weight_diff) > tolerance:
                trade_value = weight_diff * total_value
                trades[asset_id] = trade_value / pos.current_price
                
        return trades
        
    def optimize_portfolio(self, expected_returns: Dict[str, float], 
                         risk_aversion: float = 1.0) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization
        """
        n_assets = len(self.positions)
        assets = list(self.positions.keys())
        
        # Construct covariance matrix from correlation matrix and volatilities
        vols = np.array([self.positions[aid].volatility for aid in assets])
        cov_matrix = np.outer(vols, vols) * self.correlation_matrix
        
        # Setup optimization problem
        def objective(weights):
            portfolio_return = sum(weights[i] * expected_returns[aid] 
                                 for i, aid in enumerate(assets))
            portfolio_var = weights.T @ cov_matrix @ weights
            return -portfolio_return + risk_aversion * portfolio_var
            
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},  # weights sum to 1
        ]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]  # long-only constraints
        
        result = minimize(objective, 
                        x0=np.array([1.0/n_assets] * n_assets),
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints)
                        
        return dict(zip(assets, result.x))
        
    def calculate_factor_exposures(self) -> Dict[str, float]:
        """
        Calculate portfolio exposure to risk factors
        """
        exposures = {}
        weights = self.calculate_current_weights()
        
        for factor in self.risk_factors.columns:
            exposure = sum(weights[aid] * self.risk_factors.loc[aid, factor]
                         for aid in self.positions.keys())
            exposures[factor] = exposure
            
        return exposures
        
    def run_scenario_analysis(self, scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Run scenario analysis on portfolio
        """
        results = {}
        base_value = self.calculate_portfolio_value()
        
        for scenario_name, shocks in scenarios.items():
            scenario_value = 0
            for asset_id, pos in self.positions.items():
                # Apply price shock
                shocked_price = pos.current_price * (1 + shocks.get(asset_id, 0))
                scenario_value += pos.quantity * shocked_price
            
            results[scenario_name] = {
                'portfolio_value': scenario_value,
                'pnl': scenario_value - base_value,
                'return': (scenario_value - base_value) / base_value
            }
            
        return results
        
    def analyze_correlation_matrix(self, returns: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Analyze correlation structure of portfolio assets
        """
        self.correlation_matrix = returns.corr().values
        
        # Perform PCA to identify risk factors
        eigenvalues, eigenvectors = np.linalg.eigh(self.correlation_matrix)
        
        # Sort by explained variance
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]
        
        # Calculate explained variance ratio
        explained_var_ratio = eigenvalues / np.sum(eigenvalues)
        
        return eigenvalues, explained_var_ratio
        
    def attribute_performance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Attribute portfolio performance to factors
        """
        weights = self.calculate_current_weights()
        portfolio_return = sum(weights[aid] * returns[aid].mean() 
                             for aid in self.positions.keys())
        
        # Factor attribution
        factor_contrib = {}
        exposures = self.calculate_factor_exposures()
        
        for factor in self.risk_factors.columns:
            factor_return = returns[factor].mean()
            factor_contrib[factor] = exposures[factor] * factor_return
            
        # Residual return
        explained_return = sum(factor_contrib.values())
        residual = portfolio_return - explained_return
        
        return {
            'total_return': portfolio_return,
            'factor_contribution': factor_contrib,
            'residual': residual
        }
        
    def calculate_tracking_error(self, returns: pd.DataFrame, 
                               benchmark_returns: pd.Series) -> Dict[str, float]:
        """
        Calculate and decompose tracking error
        """
        weights = self.calculate_current_weights()
        portfolio_returns = sum(weights[aid] * returns[aid] 
                              for aid in self.positions.keys())
        
        tracking_diff = portfolio_returns - benchmark_returns
        tracking_error = np.std(tracking_diff) * np.sqrt(252)  # Annualized
        
        beta = np.cov(portfolio_returns, benchmark_returns)[0,1] / np.var(benchmark_returns)
        
        return {
            'tracking_error': tracking_error,
            'beta': beta,
            'correlation': np.corrcoef(portfolio_returns, benchmark_returns)[0,1]
        } 
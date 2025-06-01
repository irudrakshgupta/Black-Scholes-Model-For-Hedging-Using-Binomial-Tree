import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import cvxopt as cv
from cvxopt import solvers

class RiskManagement:
    def __init__(self):
        self.portfolio = None
        self.returns = None
        self.positions = None
        
    def set_portfolio(self, positions, returns):
        """
        Set portfolio positions and historical returns
        """
        self.positions = positions
        self.returns = returns
        
    def calculate_var(self, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk using various methods
        """
        if method == 'historical':
            return np.percentile(self.returns, (1 - confidence_level) * 100)
        elif method == 'parametric':
            z_score = stats.norm.ppf(1 - confidence_level)
            return np.mean(self.returns) + z_score * np.std(self.returns)
        elif method == 'monte_carlo':
            n_simulations = 10000
            mu = np.mean(self.returns)
            sigma = np.std(self.returns)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            return np.percentile(simulated_returns, (1 - confidence_level) * 100)
            
    def calculate_es(self, confidence_level=0.95):
        """
        Calculate Expected Shortfall (Conditional VaR)
        """
        var = self.calculate_var(confidence_level)
        return -np.mean(self.returns[self.returns <= var])
        
    def stress_test(self, scenarios):
        """
        Perform stress testing under different scenarios
        """
        results = {}
        for scenario_name, scenario in scenarios.items():
            shocked_returns = self.returns * scenario['shock_factor']
            results[scenario_name] = {
                'var': self.calculate_var(returns=shocked_returns),
                'es': self.calculate_es(returns=shocked_returns),
                'max_loss': np.min(shocked_returns)
            }
        return results
        
    def correlation_analysis(self, option_returns):
        """
        Analyze correlations between different options
        """
        return np.corrcoef(option_returns.T)
        
    def optimize_portfolio(self, expected_returns, risk_aversion=1):
        """
        Optimize portfolio using mean-variance optimization
        """
        n = len(expected_returns)
        returns = np.matrix(expected_returns)
        cov = np.matrix(np.cov(self.returns.T))
        
        P = cv.matrix(risk_aversion * cov)
        q = cv.matrix(-returns.T)
        G = cv.matrix(-np.eye(n))
        h = cv.matrix(np.zeros(n))
        A = cv.matrix(np.ones((1, n)))
        b = cv.matrix(1.0)
        
        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h, A, b)
        return np.array(sol['x']).flatten()
        
    def calculate_performance_metrics(self, returns):
        """
        Calculate various risk-adjusted performance metrics
        """
        excess_returns = returns - 0.02/252  # Assuming 2% risk-free rate
        
        sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
        
        downside_returns = returns[returns < 0]
        sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
        
        max_drawdown = self.calculate_max_drawdown(returns)
        
        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'annualized_return': np.mean(returns) * 252,
            'annualized_vol': np.std(returns) * np.sqrt(252)
        }
        
    def calculate_max_drawdown(self, returns):
        """
        Calculate maximum drawdown
        """
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = cum_returns / running_max - 1
        return np.min(drawdowns)
        
    def risk_attribution(self, weights, factor_returns):
        """
        Perform risk attribution analysis
        """
        factor_cov = np.cov(factor_returns.T)
        portfolio_risk = np.sqrt(weights.T @ factor_cov @ weights)
        
        # Component contribution to risk
        marginal_contrib = (factor_cov @ weights) / portfolio_risk
        component_contrib = weights * marginal_contrib
        
        return {
            'total_risk': portfolio_risk,
            'marginal_contribution': marginal_contrib,
            'component_contribution': component_contrib,
            'pct_contribution': component_contrib / portfolio_risk
        } 
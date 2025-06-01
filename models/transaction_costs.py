"""
Transaction Cost Analysis Module
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

@dataclass
class TransactionCostResult:
    total_cost: float
    spread_cost: float
    impact_cost: float
    commission: float
    rebalancing_cost: float
    total_trades: int
    average_trade_size: float

@dataclass
class TransactionCostParams:
    commission_rate: float
    bid_ask_spread: float
    market_impact_factor: float
    fixed_costs: float
    min_commission: float

class TransactionCostAnalyzer:
    def __init__(self, params: TransactionCostParams):
        self.params = params
        
    def calculate_commission(self, notional_value: float) -> float:
        """
        Calculate commission costs
        """
        commission = max(
            self.params.min_commission,
            notional_value * self.params.commission_rate
        )
        return commission
        
    def estimate_spread_cost(self, notional_value: float, bid_ask_spread: Optional[float] = None) -> float:
        """
        Estimate cost from bid-ask spread
        """
        spread = bid_ask_spread if bid_ask_spread is not None else self.params.bid_ask_spread
        return notional_value * spread / 2
        
    def estimate_market_impact(self, notional_value: float, adv: float) -> float:
        """
        Estimate market impact cost based on trade size relative to ADV
        """
        participation_rate = notional_value / adv
        impact = self.params.market_impact_factor * np.sqrt(participation_rate)
        return notional_value * impact
        
    def calculate_total_cost(self, 
                           notional_value: float,
                           adv: float,
                           bid_ask_spread: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate total transaction cost breakdown
        """
        commission = self.calculate_commission(notional_value)
        spread_cost = self.estimate_spread_cost(notional_value, bid_ask_spread)
        market_impact = self.estimate_market_impact(notional_value, adv)
        
        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'market_impact': market_impact,
            'fixed_costs': self.params.fixed_costs,
            'total_cost': commission + spread_cost + market_impact + self.params.fixed_costs
        }
        
    def analyze_slippage(self, 
                        executed_price: float,
                        arrival_price: float,
                        notional_value: float) -> Dict[str, float]:
        """
        Analyze execution slippage
        """
        implementation_shortfall = (executed_price - arrival_price) * notional_value
        
        expected_costs = self.calculate_total_cost(notional_value, adv=1e6)  # Example ADV
        unexpected_slippage = implementation_shortfall - expected_costs['total_cost']
        
        return {
            'implementation_shortfall': implementation_shortfall,
            'expected_costs': expected_costs['total_cost'],
            'unexpected_slippage': unexpected_slippage
        }
        
    def optimize_trade_schedule(self,
                              total_size: float,
                              adv: float,
                              urgency: float = 0.5,
                              max_participation: float = 0.3) -> Dict[str, List[float]]:
        """
        Optimize trading schedule to minimize costs
        """
        # Simple implementation of VWAP-based schedule
        daily_participation = min(max_participation, urgency)
        days_to_complete = total_size / (adv * daily_participation)
        
        # Create daily schedule
        n_days = int(np.ceil(days_to_complete))
        daily_sizes = [min(adv * daily_participation, total_size - i * adv * daily_participation)
                      for i in range(n_days)]
        
        # Calculate costs for each day
        daily_costs = [self.calculate_total_cost(size, adv)['total_cost']
                      for size in daily_sizes]
        
        return {
            'daily_sizes': daily_sizes,
            'daily_costs': daily_costs,
            'total_days': n_days,
            'total_cost': sum(daily_costs)
        }
        
    def estimate_hedging_costs(self,
                             delta: float,
                             spot_price: float,
                             adv: float,
                             rebalance_frequency: str = 'daily') -> Dict[str, float]:
        """
        Estimate costs of delta hedging
        """
        # Convert delta to shares
        shares_to_hedge = abs(delta) * 100  # Assuming standard option contract size
        notional_value = shares_to_hedge * spot_price
        
        # Estimate number of rebalances
        rebalances_per_year = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12
        }[rebalance_frequency]
        
        # Calculate costs per rebalance
        avg_rebalance_size = notional_value * 0.1  # Assume 10% adjustment per rebalance
        costs_per_rebalance = self.calculate_total_cost(avg_rebalance_size, adv)
        
        annual_costs = costs_per_rebalance['total_cost'] * rebalances_per_year
        
        return {
            'initial_hedge_cost': self.calculate_total_cost(notional_value, adv)['total_cost'],
            'estimated_annual_rebalance_cost': annual_costs,
            'costs_per_rebalance': costs_per_rebalance['total_cost']
        }
        
    def analyze_portfolio_turnover(self,
                                 portfolio_value: float,
                                 annual_turnover: float,
                                 adv_ratio: float) -> Dict[str, float]:
        """
        Analyze transaction costs from portfolio turnover
        """
        annual_traded_value = portfolio_value * annual_turnover
        avg_trade_size = annual_traded_value / 252  # Assume daily trading
        
        daily_costs = self.calculate_total_cost(
            avg_trade_size,
            adv=avg_trade_size / adv_ratio
        )
        
        annual_costs = {k: v * 252 for k, v in daily_costs.items()}
        annual_costs['turnover_rate'] = annual_turnover
        annual_costs['cost_per_turnover'] = annual_costs['total_cost'] / annual_turnover
        
        return annual_costs

    def calculate_spread_cost(self, trade_value: float) -> float:
        """Calculate cost due to bid-ask spread."""
        return trade_value * self.params.bid_ask_spread

    def calculate_price_impact(self, trade_value: float, adv: float) -> float:
        """
        Calculate price impact cost.
        
        Parameters:
        -----------
        trade_value : float
            Value of the trade
        adv : float
            Average daily volume in value terms
        """
        participation_rate = trade_value / adv
        return trade_value * self.params.market_impact_factor * np.sqrt(participation_rate)

    def calculate_total_cost_result(
        self,
        trades: np.ndarray,
        prices: np.ndarray,
        adv: Optional[float] = None
    ) -> TransactionCostResult:
        """
        Calculate total transaction costs for a series of trades.
        
        Parameters:
        -----------
        trades : array-like
            Array of trade sizes (positive for buys, negative for sells)
        prices : array-like
            Array of prices
        adv : float, optional
            Average daily volume in value terms
        
        Returns:
        --------
        TransactionCostResult
            Detailed breakdown of transaction costs
        """
        trade_values = np.abs(trades * prices)
        total_value = np.sum(trade_values)
        n_trades = np.sum(trade_values > 0)
        
        # Calculate individual cost components
        spread_costs = np.sum([self.calculate_spread_cost(tv) for tv in trade_values])
        commission_costs = np.sum([self.calculate_commission(tv) for tv in trade_values])
        
        # Calculate price impact if ADV is provided
        if adv is not None:
            impact_costs = np.sum([self.calculate_price_impact(tv, adv) for tv in trade_values])
        else:
            impact_costs = 0.0
        
        # Calculate rebalancing costs (additional trades due to delta changes)
        rebalancing_costs = np.sum(np.abs(np.diff(trades)) * prices[1:] * self.params.commission_rate)
        
        return TransactionCostResult(
            total_cost=spread_costs + commission_costs + impact_costs + rebalancing_costs,
            spread_cost=spread_costs,
            impact_cost=impact_costs,
            commission=commission_costs,
            rebalancing_cost=rebalancing_costs,
            total_trades=n_trades,
            average_trade_size=total_value / n_trades if n_trades > 0 else 0
        )

    def optimize_rebalancing(
        self,
        deltas: np.ndarray,
        prices: np.ndarray,
        threshold: float = 0.1
    ) -> Tuple[np.ndarray, TransactionCostResult]:
        """
        Optimize rebalancing by implementing a threshold strategy.
        
        Parameters:
        -----------
        deltas : array-like
            Array of target delta positions
        prices : array-like
            Array of prices
        threshold : float
            Rebalancing threshold as fraction of position
            
        Returns:
        --------
        Tuple[np.ndarray, TransactionCostResult]
            Optimized trades and transaction costs
        """
        actual_position = np.zeros_like(deltas)
        trades = np.zeros_like(deltas)
        
        # Initial position
        actual_position[0] = deltas[0]
        trades[0] = deltas[0]
        
        # Implement threshold rebalancing
        for i in range(1, len(deltas)):
            delta_diff = deltas[i] - actual_position[i-1]
            
            # Only trade if difference exceeds threshold
            if abs(delta_diff) > threshold * abs(actual_position[i-1]):
                trades[i] = delta_diff
                actual_position[i] = deltas[i]
            else:
                actual_position[i] = actual_position[i-1]
        
        # Calculate costs for optimized trading schedule
        costs = self.calculate_total_cost_result(trades, prices)
        
        return trades, costs

    def calculate_break_even_time(
        self,
        option_premium: float,
        hedge_costs: TransactionCostResult,
        daily_theta: float
    ) -> float:
        """
        Calculate break-even time considering transaction costs.
        
        Parameters:
        -----------
        option_premium : float
            Premium received from selling the option
        hedge_costs : TransactionCostResult
            Expected hedging costs
        daily_theta : float
            Option's daily theta value
            
        Returns:
        --------
        float
            Number of days to break even
        """
        daily_cost = hedge_costs.total_cost / 30  # Assume 30-day hedging period
        net_daily_profit = daily_theta - daily_cost
        
        if net_daily_profit <= 0:
            return float('inf')
        
        return option_premium / net_daily_profit 
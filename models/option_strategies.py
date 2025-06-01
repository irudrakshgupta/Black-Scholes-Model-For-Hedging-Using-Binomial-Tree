import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from .black_scholes import BlackScholes

@dataclass
class OptionPosition:
    strike: float
    maturity: float
    option_type: str
    quantity: int
    premium: float

class OptionStrategies:
    def __init__(self):
        self.bs_calculator = BlackScholes()
        
    def iron_condor(self, spot, put_strikes, call_strikes, maturity, volatility, rate=0.05):
        """
        Create Iron Condor strategy
        """
        if len(put_strikes) != 2 or len(call_strikes) != 2:
            raise ValueError("Need exactly 2 strikes each for puts and calls")
            
        positions = [
            OptionPosition(
                strike=put_strikes[0],
                maturity=maturity,
                option_type='put',
                quantity=-1,
                premium=self.bs_calculator.price_and_greeks(spot, put_strikes[0], maturity, rate, volatility, 'put').price
            ),
            OptionPosition(
                strike=put_strikes[1],
                maturity=maturity,
                option_type='put',
                quantity=1,
                premium=self.bs_calculator.price_and_greeks(spot, put_strikes[1], maturity, rate, volatility, 'put').price
            ),
            OptionPosition(
                strike=call_strikes[0],
                maturity=maturity,
                option_type='call',
                quantity=1,
                premium=self.bs_calculator.price_and_greeks(spot, call_strikes[0], maturity, rate, volatility, 'call').price
            ),
            OptionPosition(
                strike=call_strikes[1],
                maturity=maturity,
                option_type='call',
                quantity=-1,
                premium=self.bs_calculator.price_and_greeks(spot, call_strikes[1], maturity, rate, volatility, 'call').price
            )
        ]
        return positions
        
    def butterfly_spread(self, spot, center_strike, wing_width, maturity, volatility, rate=0.05):
        """
        Create Butterfly spread strategy
        """
        lower_strike = center_strike - wing_width
        upper_strike = center_strike + wing_width
        
        positions = [
            OptionPosition(
                strike=lower_strike,
                maturity=maturity,
                option_type='call',
                quantity=1,
                premium=self.bs_calculator.price_and_greeks(spot, lower_strike, maturity, rate, volatility, 'call').price
            ),
            OptionPosition(
                strike=center_strike,
                maturity=maturity,
                option_type='call',
                quantity=-2,
                premium=self.bs_calculator.price_and_greeks(spot, center_strike, maturity, rate, volatility, 'call').price
            ),
            OptionPosition(
                strike=upper_strike,
                maturity=maturity,
                option_type='call',
                quantity=1,
                premium=self.bs_calculator.price_and_greeks(spot, upper_strike, maturity, rate, volatility, 'call').price
            )
        ]
        return positions
        
    def calculate_payoff(self, positions: List[OptionPosition], spot_range):
        """
        Calculate strategy payoff across a range of spot prices
        """
        total_payoff = np.zeros_like(spot_range)
        initial_premium = 0
        
        for position in positions:
            for i, spot in enumerate(spot_range):
                if position.option_type == 'stock':
                    payoff = spot - position.strike  # Stock P&L
                elif position.option_type == 'call':
                    payoff = max(0, spot - position.strike)
                else:  # put
                    payoff = max(0, position.strike - spot)
                total_payoff[i] += position.quantity * payoff
                initial_premium += position.quantity * position.premium
                
        return total_payoff - initial_premium
        
    def calculate_greeks(self, positions: List[OptionPosition], spot, rate=0.05):
        """
        Calculate aggregate Greeks for the strategy
        """
        total_delta = 0
        total_gamma = 0
        total_theta = 0
        total_vega = 0
        
        for position in positions:
            result = self.bs_calculator.price_and_greeks(
                spot, position.strike, position.maturity, rate,
                volatility=0.2,  # Using a default vol, should be adjusted
                option_type=position.option_type
            )
            
            total_delta += position.quantity * result.delta
            total_gamma += position.quantity * result.gamma
            total_theta += position.quantity * result.theta
            total_vega += position.quantity * result.vega
            
        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }
        
    def analyze_breakeven(self, positions: List[OptionPosition], spot, precision=0.01):
        """
        Find break-even points for the strategy
        """
        spot_range = np.arange(spot * 0.5, spot * 1.5, precision)
        payoff = self.calculate_payoff(positions, spot_range)
        
        breakeven_points = []
        for i in range(1, len(payoff)):
            if payoff[i-1] * payoff[i] <= 0:
                breakeven_points.append(spot_range[i])
                
        return breakeven_points
        
    def roll_analysis(self, current_positions: List[OptionPosition], 
                     new_maturity: float, 
                     spot: float,
                     rate=0.05,
                     volatility=0.2):
        """
        Analyze rolling existing positions to a new maturity
        """
        current_value = 0
        new_value = 0
        
        for position in current_positions:
            current_price = self.bs_calculator.price_and_greeks(
                spot, position.strike, position.maturity,
                rate, volatility, position.option_type
            ).price
            new_price = self.bs_calculator.price_and_greeks(
                spot, position.strike, new_maturity,
                rate, volatility, position.option_type
            ).price
            
            current_value += position.quantity * current_price
            new_value += position.quantity * new_price
            
        roll_cost = new_value - current_value
        
        return {
            'current_value': current_value,
            'new_value': new_value,
            'roll_cost': roll_cost
        }
        
    def pnl_attribution(self, positions: List[OptionPosition], 
                       old_spot: float,
                       new_spot: float,
                       old_vol: float,
                       new_vol: float,
                       time_decay: float):
        """
        Break down P&L into components
        """
        delta_pnl = 0
        gamma_pnl = 0
        theta_pnl = 0
        vega_pnl = 0
        
        for position in positions:
            # Calculate Greeks at initial point
            result = self.bs_calculator.price_and_greeks(
                old_spot, position.strike, position.maturity,
                0.05, old_vol, position.option_type
            )
            
            # Approximate component P&Ls
            spot_change = new_spot - old_spot
            vol_change = new_vol - old_vol
            
            delta_pnl += position.quantity * result.delta * spot_change
            gamma_pnl += 0.5 * position.quantity * result.gamma * spot_change**2
            theta_pnl += position.quantity * result.theta * time_decay
            vega_pnl += position.quantity * result.vega * vol_change
            
        return {
            'delta_pnl': delta_pnl,
            'gamma_pnl': gamma_pnl,
            'theta_pnl': theta_pnl,
            'vega_pnl': vega_pnl,
            'total_pnl': delta_pnl + gamma_pnl + theta_pnl + vega_pnl
        }
        
    def covered_call(self, spot, strike, maturity, volatility, rate=0.05):
        """
        Create Covered Call strategy (long stock + short call)
        """
        positions = [
            OptionPosition(
                strike=spot,  # For stock position
                maturity=maturity,
                option_type='stock',
                quantity=100,  # Standard lot size
                premium=spot
            ),
            OptionPosition(
                strike=strike,
                maturity=maturity,
                option_type='call',
                quantity=-1,
                premium=self.bs_calculator.price_and_greeks(spot, strike, maturity, rate, volatility, 'call').price
            )
        ]
        return positions
        
    def protective_put(self, spot, strike, maturity, volatility, rate=0.05):
        """
        Create Protective Put strategy (long stock + long put)
        """
        positions = [
            OptionPosition(
                strike=spot,  # For stock position
                maturity=maturity,
                option_type='stock',
                quantity=100,  # Standard lot size
                premium=spot
            ),
            OptionPosition(
                strike=strike,
                maturity=maturity,
                option_type='put',
                quantity=1,
                premium=self.bs_calculator.price_and_greeks(spot, strike, maturity, rate, volatility, 'put').price
            )
        ]
        return positions 
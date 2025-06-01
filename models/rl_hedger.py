"""
Reinforcement Learning Environment for Option Hedging.
"""

import gym
import numpy as np
from typing import Dict, Tuple, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from .black_scholes import BlackScholes

class HedgingEnv(gym.Env):
    """Custom Environment for option hedging that follows gym interface."""
    
    def __init__(
        self,
        S0: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        r: float = 0.05,
        sigma: float = 0.2,
        dt: float = 1/252,  # Daily rebalancing
        transaction_cost: float = 0.001,  # 10 bps
        bid_ask_spread: float = 0.001,  # 10 bps
        option_type: str = 'call'
    ):
        super(HedgingEnv, self).__init__()

        # Market parameters
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.dt = dt
        self.transaction_cost = transaction_cost
        self.bid_ask_spread = bid_ask_spread
        self.option_type = option_type
        self.n_steps = int(self.T / self.dt)
        
        # State space: [stock_price, time_to_maturity, current_hedge_ratio]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -2]),
            high=np.array([np.inf, self.T, 2]),
            dtype=np.float32
        )
        
        # Action space: continuous hedge ratio adjustment [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )
        
        # Initialize Black-Scholes calculator
        self.bs = BlackScholes()
        
        # Initialize state variables
        self.reset()

    def _get_state(self) -> np.ndarray:
        """Get current state observation."""
        return np.array([
            self.S,
            self.time_to_maturity,
            self.current_hedge
        ])

    def _calculate_transaction_cost(self, old_hedge: float, new_hedge: float) -> float:
        """Calculate transaction cost for hedge rebalancing."""
        hedge_change = abs(new_hedge - old_hedge)
        return self.S * hedge_change * self.transaction_cost

    def _calculate_reward(
        self,
        old_hedge: float,
        new_hedge: float,
        old_price: float,
        new_price: float
    ) -> float:
        """Calculate reward based on P&L and transaction costs."""
        # P&L from stock position
        stock_pnl = old_hedge * (self.S - self.prev_S)
        
        # P&L from option position
        option_pnl = new_price - old_price
        
        # Transaction costs
        transaction_cost = self._calculate_transaction_cost(old_hedge, new_hedge)
        
        # Total P&L
        total_pnl = stock_pnl - option_pnl - transaction_cost
        
        # Add variance penalty to encourage smoother hedging
        variance_penalty = 0.1 * (new_hedge - old_hedge)**2
        
        return -(abs(total_pnl) + variance_penalty)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one time step within the environment.
        
        Parameters:
        -----------
        action : np.ndarray
            The hedge ratio adjustment to make
            
        Returns:
        --------
        observation : np.ndarray
            The current state observation
        reward : float
            The reward achieved by the previous action
        done : bool
            Whether the episode has ended
        info : dict
            Additional information
        """
        # Store previous values
        self.prev_S = self.S
        old_hedge = self.current_hedge
        
        # Calculate old option price
        old_result = self.bs.price_and_greeks(
            self.S, self.K, self.time_to_maturity, self.r, self.sigma, self.option_type
        )
        old_price = old_result.price
        
        # Update stock price using GBM
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * np.random.normal()
        self.S *= np.exp(drift + diffusion)
        
        # Update time
        self.time_to_maturity -= self.dt
        self.current_step += 1
        
        # Apply hedge ratio adjustment
        self.current_hedge = np.clip(
            self.current_hedge + action[0],
            -1.0,  # Allow short selling up to 100%
            2.0    # Allow leverage up to 200%
        )
        
        # Calculate new option price
        new_result = self.bs.price_and_greeks(
            self.S, self.K, self.time_to_maturity, self.r, self.sigma, self.option_type
        )
        new_price = new_result.price
        
        # Calculate reward
        reward = self._calculate_reward(old_hedge, self.current_hedge, old_price, new_price)
        
        # Check if episode is done
        done = self.current_step >= self.n_steps or self.time_to_maturity <= 0
        
        # Get new state
        observation = self._get_state()
        
        # Additional info
        info = {
            'stock_price': self.S,
            'option_price': new_price,
            'hedge_ratio': self.current_hedge,
            'theoretical_delta': new_result.delta
        }
        
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.S = self.S0
        self.prev_S = self.S0
        self.time_to_maturity = self.T
        self.current_step = 0
        self.current_hedge = 0.0
        return self._get_state()

class RLHedger:
    """Reinforcement Learning based option hedger using PPO."""
    
    def __init__(
        self,
        env_params: Optional[Dict] = None,
        model_path: Optional[str] = None
    ):
        """
        Initialize the RL Hedger.
        
        Parameters:
        -----------
        env_params : dict, optional
            Parameters for the hedging environment
        model_path : str, optional
            Path to load a pre-trained model
        """
        # Default environment parameters
        default_params = {
            'S0': 100.0,
            'K': 100.0,
            'T': 1.0,
            'r': 0.05,
            'sigma': 0.2,
            'dt': 1/252,
            'transaction_cost': 0.001,
            'bid_ask_spread': 0.001,
            'option_type': 'call'
        }
        
        # Update with provided parameters
        self.env_params = default_params.copy()
        if env_params is not None:
            self.env_params.update(env_params)
            
        # Create environment
        self.env = DummyVecEnv([lambda: HedgingEnv(**self.env_params)])
        
        # Create or load model
        if model_path is not None:
            self.model = PPO.load(model_path, env=self.env)
        else:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=None,
            )

    def train(
        self,
        total_timesteps: int = 100000,
        save_path: Optional[str] = None
    ) -> None:
        """
        Train the RL hedging model.
        
        Parameters:
        -----------
        total_timesteps : int
            Total number of timesteps to train for
        save_path : str, optional
            Path to save the trained model
        """
        self.model.learn(total_timesteps=total_timesteps)
        
        if save_path is not None:
            self.model.save(save_path)

    def get_hedge_ratio(self, observation: np.ndarray) -> float:
        """
        Get the hedge ratio for a given market state.
        
        Parameters:
        -----------
        observation : np.ndarray
            Current market state observation
            
        Returns:
        --------
        float
            Optimal hedge ratio according to the model
        """
        action, _ = self.model.predict(observation, deterministic=True)
        return float(action[0])

    def simulate_hedging(
        self,
        n_episodes: int = 1,
        render: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Simulate hedging strategy over multiple episodes.
        
        Parameters:
        -----------
        n_episodes : int
            Number of episodes to simulate
        render : bool
            Whether to render the environment
            
        Returns:
        --------
        dict
            Dictionary containing simulation results
        """
        results = {
            'stock_prices': [],
            'option_prices': [],
            'hedge_ratios': [],
            'theoretical_deltas': [],
            'rewards': []
        }
        
        for _ in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_rewards = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                results['stock_prices'].append(info[0]['stock_price'])
                results['option_prices'].append(info[0]['option_price'])
                results['hedge_ratios'].append(info[0]['hedge_ratio'])
                results['theoretical_deltas'].append(info[0]['theoretical_delta'])
                episode_rewards.append(reward[0])
                
                if render:
                    self.env.render()
            
            results['rewards'].extend(episode_rewards)
        
        return {k: np.array(v) for k, v in results.items()} 
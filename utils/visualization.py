"""
Visualization utilities for option pricing and hedging analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_option_price(
    stock_prices: np.ndarray,
    option_prices: np.ndarray,
    title: str = "Option Price vs Stock Price",
    save_path: Optional[str] = None
) -> None:
    """
    Plot option price against stock price.
    
    Parameters:
    -----------
    stock_prices : np.ndarray
        Array of stock prices
    option_prices : np.ndarray
        Array of corresponding option prices
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(stock_prices, option_prices, 'b-', label='Option Price')
    plt.xlabel('Stock Price')
    plt.ylabel('Option Price')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_greeks(
    stock_prices: np.ndarray,
    greeks: Dict[str, np.ndarray],
    title: str = "Option Greeks",
    save_path: Optional[str] = None
) -> None:
    """
    Plot option Greeks against stock price.
    
    Parameters:
    -----------
    stock_prices : np.ndarray
        Array of stock prices
    greeks : dict
        Dictionary containing arrays for each Greek
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Delta', 'Gamma', 'Theta', 'Vega')
    )
    
    greek_names = ['delta', 'gamma', 'theta', 'vega']
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for greek, pos in zip(greek_names, positions):
        if greek in greeks:
            fig.add_trace(
                go.Scatter(
                    x=stock_prices,
                    y=greeks[greek],
                    name=greek.capitalize()
                ),
                row=pos[0], col=pos[1]
            )
    
    fig.update_layout(
        height=800,
        width=1000,
        title_text=title,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_hedging_performance(
    results: Dict[str, np.ndarray],
    title: str = "Hedging Performance",
    save_path: Optional[str] = None
) -> None:
    """
    Plot hedging performance metrics.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing hedging simulation results
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Stock & Option Prices',
            'Hedge Ratio vs Theoretical Delta',
            'Cumulative P&L',
            'Reward Evolution'
        )
    )
    
    # Stock and Option Prices
    fig.add_trace(
        go.Scatter(
            y=results['stock_prices'],
            name='Stock Price'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            y=results['option_prices'],
            name='Option Price'
        ),
        row=1, col=1
    )
    
    # Hedge Ratio vs Delta
    fig.add_trace(
        go.Scatter(
            y=results['hedge_ratios'],
            name='Hedge Ratio'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            y=results['theoretical_deltas'],
            name='Theoretical Delta'
        ),
        row=1, col=2
    )
    
    # Cumulative P&L
    cumulative_rewards = np.cumsum(results['rewards'])
    fig.add_trace(
        go.Scatter(
            y=cumulative_rewards,
            name='Cumulative P&L'
        ),
        row=2, col=1
    )
    
    # Reward Evolution
    fig.add_trace(
        go.Scatter(
            y=results['rewards'],
            name='Reward'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        width=1200,
        title_text=title,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_model_comparison(
    stock_prices: np.ndarray,
    model_results: Dict[str, Dict[str, np.ndarray]],
    metrics: List[str] = ['price', 'delta'],
    title: str = "Model Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between different pricing models.
    
    Parameters:
    -----------
    stock_prices : np.ndarray
        Array of stock prices
    model_results : dict
        Dictionary containing results from different models
    metrics : list
        List of metrics to compare
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    n_metrics = len(metrics)
    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        subplot_titles=[m.capitalize() for m in metrics]
    )
    
    colors = ['blue', 'red', 'green']
    
    for i, metric in enumerate(metrics, 1):
        for (model_name, results), color in zip(model_results.items(), colors):
            if metric in results:
                fig.add_trace(
                    go.Scatter(
                        x=stock_prices,
                        y=results[metric],
                        name=f"{model_name} {metric}",
                        line=dict(color=color)
                    ),
                    row=i,
                    col=1
                )
    
    fig.update_layout(
        height=300 * n_metrics,
        width=1000,
        title_text=title,
        showlegend=True
    )
    
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

def plot_training_progress(
    training_rewards: List[float],
    window_size: int = 100,
    title: str = "Training Progress",
    save_path: Optional[str] = None
) -> None:
    """
    Plot training progress of the RL model.
    
    Parameters:
    -----------
    training_rewards : list
        List of rewards during training
    window_size : int
        Window size for moving average
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    rewards = np.array(training_rewards)
    moving_avg = np.convolve(
        rewards,
        np.ones(window_size) / window_size,
        mode='valid'
    )
    
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, 'b-', alpha=0.3, label='Raw Rewards')
    plt.plot(
        np.arange(window_size-1, len(rewards)),
        moving_avg,
        'r-',
        label=f'{window_size}-Episode Moving Average'
    )
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.close() 
"""
PyQt5-based GUI for the Deep Hedger application.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QTabWidget,
    QGroupBox, QFormLayout, QMessageBox
)
from PyQt5.QtCore import Qt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.black_scholes import BlackScholes
from models.binomial_tree import BinomialTree
from models.rl_hedger import RLHedger
from utils.visualization import (
    plot_option_price, plot_greeks, plot_hedging_performance,
    plot_model_comparison
)

class HedgerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Hedger")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize models
        self.bs_model = BlackScholes()
        self.binomial_model = BinomialTree(steps=100)
        self.rl_model = RLHedger()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create tabs
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Add tabs
        tabs.addTab(self._create_pricing_tab(), "Option Pricing")
        tabs.addTab(self._create_hedging_tab(), "Hedging Simulation")
        tabs.addTab(self._create_training_tab(), "RL Training")
        
    def _create_pricing_tab(self) -> QWidget:
        """Create the option pricing tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Parameters group
        param_group = QGroupBox("Option Parameters")
        param_layout = QFormLayout()
        
        # Create input fields
        self.stock_price = QLineEdit("100.0")
        self.strike_price = QLineEdit("100.0")
        self.time_to_maturity = QLineEdit("1.0")
        self.risk_free_rate = QLineEdit("0.05")
        self.volatility = QLineEdit("0.2")
        
        # Add fields to form layout
        param_layout.addRow("Stock Price:", self.stock_price)
        param_layout.addRow("Strike Price:", self.strike_price)
        param_layout.addRow("Time to Maturity (years):", self.time_to_maturity)
        param_layout.addRow("Risk-free Rate:", self.risk_free_rate)
        param_layout.addRow("Volatility:", self.volatility)
        
        # Option type selection
        self.option_type = QComboBox()
        self.option_type.addItems(["call", "put"])
        param_layout.addRow("Option Type:", self.option_type)
        
        # Model selection
        self.model_type = QComboBox()
        self.model_type.addItems(["Black-Scholes", "Binomial Tree", "RL Hedger"])
        param_layout.addRow("Model:", self.model_type)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        calculate_btn = QPushButton("Calculate")
        calculate_btn.clicked.connect(self._calculate_option)
        button_layout.addWidget(calculate_btn)
        
        plot_btn = QPushButton("Plot Greeks")
        plot_btn.clicked.connect(self._plot_greeks)
        button_layout.addWidget(plot_btn)
        
        compare_btn = QPushButton("Compare Models")
        compare_btn.clicked.connect(self._compare_models)
        button_layout.addWidget(compare_btn)
        
        layout.addLayout(button_layout)
        
        # Results group
        results_group = QGroupBox("Results")
        results_layout = QFormLayout()
        
        self.price_result = QLabel("--")
        self.delta_result = QLabel("--")
        self.gamma_result = QLabel("--")
        self.theta_result = QLabel("--")
        self.vega_result = QLabel("--")
        
        results_layout.addRow("Price:", self.price_result)
        results_layout.addRow("Delta:", self.delta_result)
        results_layout.addRow("Gamma:", self.gamma_result)
        results_layout.addRow("Theta:", self.theta_result)
        results_layout.addRow("Vega:", self.vega_result)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        return tab
        
    def _create_hedging_tab(self) -> QWidget:
        """Create the hedging simulation tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Simulation parameters group
        sim_group = QGroupBox("Simulation Parameters")
        sim_layout = QFormLayout()
        
        self.n_steps = QLineEdit("252")  # Daily steps for 1 year
        self.transaction_cost = QLineEdit("0.001")
        self.n_simulations = QLineEdit("1")
        
        sim_layout.addRow("Number of Steps:", self.n_steps)
        sim_layout.addRow("Transaction Cost:", self.transaction_cost)
        sim_layout.addRow("Number of Simulations:", self.n_simulations)
        
        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        simulate_btn = QPushButton("Run Simulation")
        simulate_btn.clicked.connect(self._run_simulation)
        button_layout.addWidget(simulate_btn)
        
        plot_perf_btn = QPushButton("Plot Performance")
        plot_perf_btn.clicked.connect(self._plot_performance)
        button_layout.addWidget(plot_perf_btn)
        
        layout.addLayout(button_layout)
        
        return tab
        
    def _create_training_tab(self) -> QWidget:
        """Create the RL training tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Training parameters group
        train_group = QGroupBox("Training Parameters")
        train_layout = QFormLayout()
        
        self.n_episodes = QLineEdit("1000")
        self.learning_rate = QLineEdit("0.0003")
        self.batch_size = QLineEdit("64")
        
        train_layout.addRow("Number of Episodes:", self.n_episodes)
        train_layout.addRow("Learning Rate:", self.learning_rate)
        train_layout.addRow("Batch Size:", self.batch_size)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        train_btn = QPushButton("Train Model")
        train_btn.clicked.connect(self._train_model)
        button_layout.addWidget(train_btn)
        
        save_btn = QPushButton("Save Model")
        save_btn.clicked.connect(self._save_model)
        button_layout.addWidget(save_btn)
        
        load_btn = QPushButton("Load Model")
        load_btn.clicked.connect(self._load_model)
        button_layout.addWidget(load_btn)
        
        layout.addLayout(button_layout)
        
        return tab
    
    def _get_option_params(self) -> dict:
        """Get option parameters from input fields."""
        try:
            params = {
                'S': float(self.stock_price.text()),
                'K': float(self.strike_price.text()),
                'T': float(self.time_to_maturity.text()),
                'r': float(self.risk_free_rate.text()),
                'sigma': float(self.volatility.text()),
                'option_type': self.option_type.currentText()
            }
            return params
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", "Please enter valid numeric values.")
            return None
            
    def _calculate_option(self) -> None:
        """Calculate option price and Greeks."""
        params = self._get_option_params()
        if not params:
            return
            
        model_type = self.model_type.currentText()
        
        try:
            if model_type == "Black-Scholes":
                result = self.bs_model.price_and_greeks(**params)
            elif model_type == "Binomial Tree":
                result = self.binomial_model.price(**params)
            else:  # RL Hedger
                # For RL model, we'll use Black-Scholes as benchmark
                result = self.bs_model.price_and_greeks(**params)
            
            # Update results
            self.price_result.setText(f"{result.price:.4f}")
            self.delta_result.setText(f"{result.delta:.4f}")
            self.gamma_result.setText(f"{result.gamma:.4f}")
            self.theta_result.setText(f"{result.theta:.4f}")
            if hasattr(result, 'vega'):
                self.vega_result.setText(f"{result.vega:.4f}")
            else:
                self.vega_result.setText("N/A")
                
        except Exception as e:
            QMessageBox.warning(self, "Calculation Error", str(e))
            
    def _plot_greeks(self) -> None:
        """Plot option Greeks."""
        params = self._get_option_params()
        if not params:
            return
            
        # Generate stock price range
        S = params['S']
        stock_prices = np.linspace(0.5 * S, 1.5 * S, 100)
        
        # Calculate Greeks for each stock price
        greeks = {
            'delta': [],
            'gamma': [],
            'theta': [],
            'vega': []
        }
        
        for s in stock_prices:
            params['S'] = s
            result = self.bs_model.price_and_greeks(**params)
            greeks['delta'].append(result.delta)
            greeks['gamma'].append(result.gamma)
            greeks['theta'].append(result.theta)
            greeks['vega'].append(result.vega)
        
        # Convert to numpy arrays
        for key in greeks:
            greeks[key] = np.array(greeks[key])
            
        # Plot
        plot_greeks(stock_prices, greeks)
        
    def _compare_models(self) -> None:
        """Compare different pricing models."""
        params = self._get_option_params()
        if not params:
            return
            
        # Generate stock price range
        S = params['S']
        stock_prices = np.linspace(0.5 * S, 1.5 * S, 100)
        
        # Calculate prices and deltas for each model
        results = {
            'Black-Scholes': {'price': [], 'delta': []},
            'Binomial Tree': {'price': [], 'delta': []},
            'RL Hedger': {'price': [], 'delta': []}
        }
        
        for s in stock_prices:
            params['S'] = s
            
            # Black-Scholes
            bs_result = self.bs_model.price_and_greeks(**params)
            results['Black-Scholes']['price'].append(bs_result.price)
            results['Black-Scholes']['delta'].append(bs_result.delta)
            
            # Binomial Tree
            bin_result = self.binomial_model.price(**params)
            results['Binomial Tree']['price'].append(bin_result.price)
            results['Binomial Tree']['delta'].append(bin_result.delta)
            
            # RL Hedger (using Black-Scholes as benchmark for now)
            results['RL Hedger']['price'].append(bs_result.price)
            results['RL Hedger']['delta'].append(bs_result.delta)
        
        # Convert to numpy arrays
        for model in results:
            for key in results[model]:
                results[model][key] = np.array(results[model][key])
                
        # Plot comparison
        plot_model_comparison(stock_prices, results)
        
    def _run_simulation(self) -> None:
        """Run hedging simulation."""
        params = self._get_option_params()
        if not params:
            return
            
        try:
            n_steps = int(self.n_steps.text())
            transaction_cost = float(self.transaction_cost.text())
            n_simulations = int(self.n_simulations.text())
            
            # Update environment parameters
            env_params = {
                'S0': params['S'],
                'K': params['K'],
                'T': params['T'],
                'r': params['r'],
                'sigma': params['sigma'],
                'dt': params['T'] / n_steps,
                'transaction_cost': transaction_cost,
                'option_type': params['option_type']
            }
            
            # Create new environment with updated parameters
            self.rl_model = RLHedger(env_params=env_params)
            
            # Run simulation
            results = self.rl_model.simulate_hedging(n_episodes=n_simulations)
            
            # Plot results
            plot_hedging_performance(results)
            
        except Exception as e:
            QMessageBox.warning(self, "Simulation Error", str(e))
            
    def _train_model(self) -> None:
        """Train the RL model."""
        try:
            n_episodes = int(self.n_episodes.text())
            learning_rate = float(self.learning_rate.text())
            batch_size = int(self.batch_size.text())
            
            # Update model parameters
            self.rl_model.model.learning_rate = learning_rate
            self.rl_model.model.batch_size = batch_size
            
            # Train model
            self.rl_model.train(total_timesteps=n_episodes)
            
            QMessageBox.information(self, "Training Complete", 
                                  f"Model trained for {n_episodes} episodes.")
            
        except Exception as e:
            QMessageBox.warning(self, "Training Error", str(e))
            
    def _save_model(self) -> None:
        """Save the trained RL model."""
        try:
            self.rl_model.model.save("trained_model")
            QMessageBox.information(self, "Save Complete", 
                                  "Model saved to 'trained_model'")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))
            
    def _load_model(self) -> None:
        """Load a trained RL model."""
        try:
            self.rl_model = RLHedger(model_path="trained_model")
            QMessageBox.information(self, "Load Complete", 
                                  "Model loaded from 'trained_model'")
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

def main():
    app = QApplication(sys.argv)
    window = HedgerGUI()
    window.show()
    sys.exit(app.exec_()) 
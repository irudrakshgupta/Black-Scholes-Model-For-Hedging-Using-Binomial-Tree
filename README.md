# Black-Scholes-Merton Binomial Tree Model for Option Pricing and Hedging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive financial modeling tool that implements advanced option pricing and hedging strategies using the Black-Scholes-Merton framework and binomial tree methods. This interactive application provides sophisticated analytics for derivatives pricing, risk management, and dynamic hedging simulations.

## 📚 Table of Contents

- [Core Financial Concepts](#core-financial-concepts)
- [Mathematical Models](#mathematical-models)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Technical Implementation](#technical-implementation)
- [Contributing](#contributing)
- [License](#license)

## 🎓 Core Financial Concepts

### Option Basics
- **Options**: Financial derivatives giving the right (not obligation) to buy (call) or sell (put) an asset at a predetermined price
- **Strike Price (K)**: The predetermined price at which the option can be exercised
- **Maturity (T)**: Time until option expiration
- **Option Types**:
  - European: Can only be exercised at maturity
  - American: Can be exercised at any time until maturity

### Risk Measures (Greeks)
- **Delta (Δ)**: Measures change in option price relative to underlying asset price
- **Gamma (Γ)**: Rate of change in delta (second derivative)
- **Theta (Θ)**: Time decay of option value
- **Vega (V)**: Sensitivity to volatility changes
- **Rho (ρ)**: Sensitivity to interest rate changes

### Volatility Concepts
- **Historical Volatility**: Calculated from past price movements
- **Implied Volatility**: Derived from market prices using BSM model
- **Volatility Surface**: 3D representation of implied volatility across strikes and maturities
- **Volatility Smile**: Pattern of implied volatility varying with strike price

## 📐 Mathematical Models

### Black-Scholes-Merton Model
The fundamental equation for option pricing:
```math
∂V/∂t + (1/2)σ²S²(∂²V/∂S²) + rS(∂V/∂S) - rV = 0
```
Where:
- V: Option value
- S: Stock price
- σ: Volatility
- r: Risk-free rate
- t: Time

### Binomial Tree Model
- Based on Cox-Ross-Rubinstein framework
- Discrete-time approximation of continuous BSM model
- Parameters:
  - u = e^(σ√Δt): Up factor
  - d = 1/u: Down factor
  - p = (e^(rΔt) - d)/(u - d): Risk-neutral probability

### Advanced Models
- **Monte Carlo Simulation**: For complex path-dependent options
- **Reinforcement Learning**: Dynamic hedging optimization
- **Transaction Cost Analysis**: Real-world trading friction modeling

## 🚀 Features

### 1. Option Price Calculator
- Black-Scholes-Merton pricing
- Binomial tree implementation
- Greeks calculation
- Implied volatility solver

### 2. Greeks Visualization
- Interactive 2D/3D plots
- Sensitivity analysis
- Time decay visualization
- Greek surfaces

### 3. Volatility Analysis
- Historical volatility calculation
- Implied volatility surface
- Volatility smile fitting
- Term structure analysis

### 4. Option Strategy Builder
- Multiple legs support
- P&L analysis
- Risk metrics
- Strategy visualization

### 5. Hedging Tools
- Delta-neutral strategies
- Dynamic rebalancing
- Transaction cost optimization
- Hedge effectiveness metrics

### 6. Risk Management
- VaR calculation
- Stress testing
- Scenario analysis
- Portfolio optimization

### 7. Machine Learning Integration
- RL-based hedging
- Volatility prediction
- Risk factor analysis
- Pattern recognition

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BSM-Binomial-Tree-Model.git
cd BSM-Binomial-Tree-Model
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Usage Examples

### Basic Option Pricing
```python
from models.black_scholes import BlackScholes

bs = BlackScholes()
result = bs.price_and_greeks(
    S=100,    # Stock price
    K=100,    # Strike price
    T=1,      # Time to maturity (years)
    r=0.05,   # Risk-free rate
    sigma=0.2 # Volatility
)
print(f"Option Price: {result.price:.2f}")
print(f"Delta: {result.delta:.4f}")
```

### Binomial Tree Analysis
```python
from models.binomial_tree import BinomialTree

bt = BinomialTree(steps=100)
result = bt.price(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    option_type='american'
)
```

## 🗂 Project Structure

```
├── models/
│   ├── black_scholes.py       # Core BSM implementation
│   ├── binomial_tree.py       # Binomial model
│   ├── advanced_pricing.py    # Complex derivatives
│   ├── advanced_volatility.py # Vol surface/smile
│   ├── hedge_ratio.py        # Hedging calculations
│   ├── portfolio_management.py # Portfolio tools
│   ├── risk_management.py    # Risk metrics
│   ├── rl_hedger.py         # ML-based hedging
│   └── transaction_costs.py  # Trading frictions
├── utils/
│   └── visualization.py      # Plotting tools
├── gui/
│   └── hedger_gui.py        # Streamlit interface
├── templates/
│   └── index.html           # Web templates
├── streamlit_app.py         # Main application
├── requirements.txt         # Dependencies
└── README.md               # Documentation
```

## 🔧 Technical Implementation

### Core Components

1. **Black-Scholes Engine**
   - Analytical solutions for European options
   - Greeks calculation using partial derivatives
   - Implied volatility solver using Newton-Raphson

2. **Binomial Tree Engine**
   - Flexible n-step implementation
   - American option early exercise
   - Path-dependent option support

3. **Volatility Module**
   - GARCH model implementation
   - Surface fitting using cubic splines
   - Volatility forecasting

4. **Risk Management**
   - Historical VaR calculation
   - Monte Carlo VaR simulation
   - Conditional VaR (Expected Shortfall)

5. **Machine Learning Integration**
   - Deep Q-Learning for hedging
   - Neural network architecture
   - Experience replay buffer

### Performance Optimizations
- Vectorized numpy operations
- Cython for intensive calculations
- Parallel processing for simulations
- Memory-efficient data structures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/BSM-Binomial-Tree-Model](https://github.com/yourusername/BSM-Binomial-Tree-Model)

## 🙏 Acknowledgments

- Black-Scholes-Merton model theory
- Cox-Ross-Rubinstein binomial model
- Modern derivatives pricing literature
- Open-source financial libraries 
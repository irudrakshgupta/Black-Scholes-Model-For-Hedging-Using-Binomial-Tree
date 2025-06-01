# Black-Scholes Option Pricing & Hedging Tool

A comprehensive options analysis and trading platform built with Python and Streamlit. This tool provides advanced option pricing, risk analysis, and portfolio management capabilities.

## Features

### Core Option Pricing
- Black-Scholes option pricing
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- Interactive price and Greeks visualization
- Implied volatility calculator

### Advanced Volatility Analysis
- GARCH model for volatility forecasting
- Volatility surface calibration
- Volatility cone analysis
- Term structure visualization
- Historical vs implied volatility comparison
- Machine learning-based volatility prediction

### Risk Management
- Value at Risk (VaR) calculator
- Expected Shortfall (ES) metrics
- Stress testing scenarios
- Correlation analysis
- Portfolio optimization
- Risk-adjusted performance metrics
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown

### Option Strategies
- Complex spread strategies:
  - Iron Condor
  - Butterfly spreads
  - Calendar spreads
- Strategy comparison tool
- P&L attribution analysis
- Break-even calculator
- Roll analysis
- Options chain visualization

### Transaction Cost Analysis
- Bid-ask spread impact calculator
- Commission cost analysis
- Slippage estimation
- Market impact modeling
- Total cost of trading analysis

### Advanced Pricing Models
- Heston stochastic volatility model
- Merton jump diffusion model
- Local volatility model
- American option pricing
- Barrier option pricing
- Monte Carlo simulation

### Portfolio Management
- Portfolio rebalancing calculator
- Correlation matrix visualization
- Factor analysis
- Portfolio attribution
- Scenario analysis
- Dynamic hedging simulation

### Machine Learning Features
- Volatility forecasting
- Regime detection
- Anomaly detection
- Trading signal generation
- Feature importance analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/options-analysis-tool.git
cd options-analysis-tool
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Analysis Tools

### Volatility Analysis
- Historical volatility calculation with configurable lookback periods
- Implied volatility surface fitting and visualization
- GARCH model parameter estimation and forecasting
- Volatility regime detection using machine learning

### Risk Metrics
- Portfolio VaR calculation using multiple methods:
  - Historical simulation
  - Parametric VaR
  - Monte Carlo simulation
- Expected Shortfall computation
- Stress testing with customizable scenarios
- Correlation analysis and risk decomposition

### Option Strategies
- Pre-built option strategies with customizable parameters
- Strategy P&L visualization
- Greeks analysis for complex positions
- Roll analysis for position management
- Break-even probability calculation

### Portfolio Management
- Mean-variance optimization
- Risk factor analysis
- Performance attribution
- Rebalancing recommendations
- Transaction cost optimization

## Data Sources
- Real-time market data integration
- Historical price data analysis
- Volatility surface calibration
- Options chain data
- Risk factor data

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Black-Scholes model implementation
- Heston model calibration
- GARCH model implementation
- Portfolio optimization algorithms
- Risk management tools

## Contact
For questions and feedback, please open an issue on GitHub. 
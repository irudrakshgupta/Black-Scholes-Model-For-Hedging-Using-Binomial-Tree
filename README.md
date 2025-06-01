# Black-Scholes-Merton Binomial Tree Model for Option Pricing and Hedging

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive financial modeling tool that implements advanced option pricing and hedging strategies using the Black-Scholes-Merton framework and binomial tree methods. This interactive application provides sophisticated analytics for derivatives pricing, risk management, and dynamic hedging simulations.

## 📚 Table of Contents

- [Core Financial Concepts](#core-financial-concepts)
- [Mathematical Foundations](#mathematical-foundations)
- [Implementation Details](#implementation-details)
- [Step-by-Step Guides](#step-by-step-guides)
- [Advanced Topics](#advanced-topics)
- [API Reference](#api-reference)
- [Installation & Setup](#installation--setup)
- [Contributing](#contributing)
- [License](#license)

## 🎓 Core Financial Concepts

### 1. Options Fundamentals

#### 1.1 Basic Definitions
- **Option Contract**: A financial derivative giving the right (not obligation) to buy/sell an asset
- **Call Option**: Right to buy the underlying asset
- **Put Option**: Right to sell the underlying asset
- **Strike Price (K)**: Predetermined exercise price
- **Maturity (T)**: Time until expiration
- **Premium**: Price paid for the option

#### 1.2 Option Classifications
- **By Exercise Rights**:
  - European: Exercise only at maturity
  - American: Exercise any time until maturity
  - Bermudan: Exercise on specific dates
- **By Market Position**:
  - Long: Buyer of the option
  - Short: Seller of the option
- **By Strike Relation**:
  - In-the-money (ITM)
  - At-the-money (ATM)
  - Out-of-the-money (OTM)

### 2. Risk Measures (Greeks)

#### 2.1 First-Order Greeks
- **Delta (Δ)**:
  ```math
  Δ = ∂V/∂S
  ```
  - Measures price sensitivity to underlying
  - Used for delta-hedging strategies
  - Range: [-1, 1] for vanilla options

- **Theta (Θ)**:
  ```math
  Θ = ∂V/∂t
  ```
  - Time decay of option value
  - Usually negative for bought options
  - Expressed in value/day

- **Rho (ρ)**:
  ```math
  ρ = ∂V/∂r
  ```
  - Interest rate sensitivity
  - Usually larger for longer-dated options

#### 2.2 Second-Order Greeks
- **Gamma (Γ)**:
  ```math
  Γ = ∂²V/∂S² = ∂Δ/∂S
  ```
  - Rate of change of delta
  - Key for dynamic hedging
  - Always positive for vanilla options

- **Vega (V)**:
  ```math
  V = ∂V/∂σ
  ```
  - Volatility sensitivity
  - Important for volatility trading
  - Always positive for vanilla options

### 3. Volatility Concepts

#### 3.1 Historical Volatility
- **Definition**: Standard deviation of returns
  ```math
  σ_hist = \sqrt{\frac{1}{n-1}\sum_{i=1}^n(r_i - \bar{r})²}
  ```
  where:
  - r_i: logarithmic returns
  - n: number of observations

#### 3.2 Implied Volatility
- **Definition**: Volatility implied by market prices
- **Black-Scholes Inversion**:
  ```math
  Market Price = BS(S, K, T, r, σ_imp)
  ```
- **Volatility Surface**:
  - 3D representation across strikes and maturities
  - Shows market pricing of volatility risk

## 📐 Mathematical Foundations

### 1. Black-Scholes-Merton Model

#### 1.1 Core Assumptions
- Log-normal price distribution
- No arbitrage
- Continuous trading
- No transaction costs
- Risk-free rate constant
- Volatility constant

#### 1.2 Fundamental Equation
```math
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma²S²\frac{\partial²V}{\partial S²} + rS\frac{\partial V}{\partial S} - rV = 0
```

#### 1.3 Solution for European Options
- **Call Option Price**:
  ```math
  C = SN(d₁) - Ke^{-rT}N(d₂)
  ```
- **Put Option Price**:
  ```math
  P = Ke^{-rT}N(-d₂) - SN(-d₁)
  ```
where:
```math
d₁ = \frac{\ln(S/K) + (r + \sigma²/2)T}{\sigma\sqrt{T}}
```
```math
d₂ = d₁ - \sigma\sqrt{T}
```

### 2. Binomial Tree Model

#### 2.1 Model Parameters
- Up factor: `u = e^(σ√Δt)`
- Down factor: `d = 1/u`
- Risk-neutral probability:
  ```math
  p = \frac{e^{rΔt} - d}{u - d}
  ```

#### 2.2 Price Evolution
```math
S_{i,j} = S_0 u^j d^{i-j}
```
where:
- i: time step
- j: up movements

#### 2.3 Option Value
```math
V_{i,j} = e^{-rΔt}[pV_{i+1,j+1} + (1-p)V_{i+1,j}]
```

## 🛠️ Implementation Details

### 1. Black-Scholes Implementation

```python
def price_european_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return call
```

### 2. Binomial Tree Implementation

```python
def build_tree(S, K, T, r, sigma, n):
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt) - d)/(u - d)
    
    # Build stock price tree
    stock = np.zeros((n+1, n+1))
    for i in range(n+1):
        for j in range(i+1):
            stock[j,i] = S * (u**j) * (d**(i-j))
            
    return stock, p, dt
```

## 📚 Step-by-Step Guides

### 1. Basic Option Pricing

#### 1.1 Using Black-Scholes
```python
from models.black_scholes import BlackScholes

# Initialize calculator
bs = BlackScholes()

# Price a European call
result = bs.price_and_greeks(
    S=100,    # Current stock price
    K=100,    # Strike price
    T=1,      # Time to maturity (years)
    r=0.05,   # Risk-free rate (5%)
    sigma=0.2 # Volatility (20%)
)

print(f"Option Price: ${result.price:.2f}")
print(f"Delta: {result.delta:.4f}")
print(f"Gamma: {result.gamma:.4f}")
print(f"Theta: ${result.theta:.4f}/day")
print(f"Vega: ${result.vega:.4f}/%vol")
```

#### 1.2 Using Binomial Tree
```python
from models.binomial_tree import BinomialTree

# Initialize with 100 steps
bt = BinomialTree(steps=100)

# Price an American option
result = bt.price(
    S=100,
    K=100,
    T=1,
    r=0.05,
    sigma=0.2,
    option_type='american',
    exercise='american'
)
```

### 2. Volatility Analysis

#### 2.1 Historical Volatility
```python
from models.historical_volatility import HistoricalVolatility

# Calculate historical volatility
hv = HistoricalVolatility()
sigma = hv.calculate(
    prices=price_series,
    window=30,  # 30-day rolling window
    annualize=True
)
```

#### 2.2 Implied Volatility Surface
```python
from models.advanced_volatility import ImpliedVolatility

# Create volatility surface
iv = ImpliedVolatility()
surface = iv.create_surface(
    option_chain=market_data,
    method='cubic'  # Cubic spline interpolation
)
```

### 3. Risk Management

#### 3.1 VaR Calculation
```python
from models.risk_management import VaRCalculator

# Calculate Value at Risk
var = VaRCalculator()
result = var.historical_var(
    returns=portfolio_returns,
    confidence_level=0.95,
    time_horizon=10
)
```

#### 3.2 Portfolio Hedging
```python
from models.portfolio_management import PortfolioHedger

# Optimize hedge ratios
hedger = PortfolioHedger()
hedge_ratios = hedger.optimize_hedge(
    portfolio=positions,
    hedge_instruments=available_options,
    objective='minimize_variance'
)
```

## 🔬 Advanced Topics

### 1. Machine Learning Integration

#### 1.1 Reinforcement Learning for Hedging
```python
from models.rl_hedger import RLHedger

# Train RL hedging agent
agent = RLHedger(
    state_dim=4,
    action_dim=1,
    learning_rate=0.001
)

# Train the agent
agent.train(
    episodes=1000,
    market_data=training_data
)
```

### 2. Transaction Cost Analysis

#### 2.1 Cost-Adjusted Pricing
```python
from models.transaction_costs import TransactionCostModel

# Initialize cost model
tcm = TransactionCostModel(
    spread=0.01,  # 1% bid-ask spread
    commission=0.001  # 0.1% commission
)

# Get cost-adjusted prices
adjusted_price = tcm.adjust_option_price(
    price=theoretical_price,
    volume=trade_size
)
```

## 📊 API Reference

### Black-Scholes Module
```python
class BlackScholes:
    def price_and_greeks(
        S: float,    # Stock price
        K: float,    # Strike price
        T: float,    # Time to maturity
        r: float,    # Risk-free rate
        sigma: float # Volatility
    ) -> OptionResult:
        """
        Calculate option price and Greeks
        Returns: OptionResult(price, delta, gamma, theta, vega, rho)
        """
```

### Binomial Tree Module
```python
class BinomialTree:
    def price(
        S: float,    # Stock price
        K: float,    # Strike price
        T: float,    # Time to maturity
        r: float,    # Risk-free rate
        sigma: float,# Volatility
        steps: int   # Number of time steps
    ) -> BinomialTreeResult:
        """
        Price options using binomial tree
        Returns: BinomialTreeResult(price, delta, gamma, theta)
        """
```

## 💻 Installation & Setup

### 1. System Requirements
- Python 3.8+
- 4GB RAM minimum
- CUDA-capable GPU (optional, for ML features)

### 2. Installation Steps
```bash
# Clone repository
git clone https://github.com/yourusername/BSM-Binomial-Tree-Model.git
cd BSM-Binomial-Tree-Model

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Start the application
streamlit run streamlit_app.py
```

### 3. Configuration
Edit `.streamlit/config.toml` for UI customization:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
font = "sans serif"
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/BSM-Binomial-Tree-Model](https://github.com/yourusername/BSM-Binomial-Tree-Model)

## 🙏 Acknowledgments

- Black-Scholes-Merton model theory
- Cox-Ross-Rubinstein binomial model
- Modern derivatives pricing literature
- Open-source financial libraries 
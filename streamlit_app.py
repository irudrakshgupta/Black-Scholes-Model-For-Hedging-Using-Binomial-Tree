import streamlit as st
import numpy as np
import plotly.graph_objects as go
from models.black_scholes import BlackScholes
from models.option_strategies import OptionStrategies
import plotly.express as px
from datetime import datetime, timedelta

# Force dark theme and configure page
st.set_page_config(
    page_title="Black-Scholes Calculator & Hedging Tools",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS to enforce dark mode and improve UI
st.markdown("""
    <style>
    /* Dark theme colors */
    :root {
        --background-color: #0E1117;
        --secondary-background-color: #1A1C24;
        --text-color: #F8F9FA;
        --font-family: 'Source Sans Pro', sans-serif;
    }
    
    /* Main container */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    /* Inputs and Widgets */
    .stTextInput > div > div,
    .stNumberInput > div > div,
    .stSelectbox > div > div {
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
    }
    
    /* Hide unnecessary UI elements */
    #MainMenu, footer, header {
        display: none !important;
    }
    
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"],
    div[data-testid="stStatusWidget"] {
        display: none;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #00ADB5 !important;
        color: var(--text-color) !important;
    }
    
    /* Plots */
    .stPlotlyChart {
        background-color: var(--secondary-background-color);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Black-Scholes calculator
bs_calculator = BlackScholes()
option_strategies = OptionStrategies()

# Sidebar navigation
with st.sidebar:
    st.title('Black-Scholes Analysis Tools')
    analysis_type = st.selectbox(
        'Select Analysis Type',
        [
            'Option Calculator',
            'Greeks Visualization',
            'Implied Volatility Surface',
            'Option Strategy Builder',
            'Monte Carlo Simulation',
            'Delta Hedging Calculator',
            'Portfolio Risk Analysis',
            'Dynamic Hedging Simulator',
            'Hedge Effectiveness'
        ],
        key='analysis_type'
    )

if analysis_type == "Option Calculator":
    # Original calculator layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        
        stock_price = st.number_input(
            "Stock Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Current price of the underlying stock"
        )
        
        strike_price = st.number_input(
            "Strike Price ($)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            help="Strike price of the option"
        )
        
        time_to_maturity = st.number_input(
            "Time to Maturity (years)",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Time until option expiration in years"
        )
        
        risk_free_rate = st.number_input(
            "Risk-free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Annual risk-free interest rate"
        ) / 100
        
        volatility = st.number_input(
            "Volatility (%)",
            min_value=1.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
            help="Annual volatility of the underlying stock"
        ) / 100
        
        option_type = st.radio(
            "Option Type",
            options=["call", "put"],
            horizontal=True,
            help="Choose between Call and Put options"
        )

    try:
        result = bs_calculator.price_and_greeks(
            S=stock_price,
            K=strike_price,
            T=time_to_maturity,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type
        )
        
        with col2:
            st.subheader("Results")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Option Price", f"${result.price:.4f}")
                st.metric("Delta", f"{result.delta:.4f}")
                st.metric("Gamma", f"{result.gamma:.4f}")
            
            with metrics_col2:
                st.metric("Theta", f"{result.theta:.4f}")
                st.metric("Vega", f"{result.vega:.4f}")
                st.metric("Rho", f"{result.rho:.4f}")

        # Price Sensitivity Analysis
        st.subheader("Price Sensitivity Analysis")
        
        price_range = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
        prices = []
        deltas = []
        
        for price in price_range:
            res = bs_calculator.price_and_greeks(
                S=price,
                K=strike_price,
                T=time_to_maturity,
                r=risk_free_rate,
                sigma=volatility,
                option_type=option_type
            )
            prices.append(res.price)
            deltas.append(res.delta)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=prices,
            name="Option Price",
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=price_range,
            y=deltas,
            name="Delta",
            line=dict(color='red', dash='dash'),
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Option Price and Delta vs Stock Price",
            xaxis_title="Stock Price ($)",
            yaxis_title="Option Price ($)",
            yaxis2=dict(
                title="Delta",
                overlaying="y",
                side="right"
            ),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error in calculation: {str(e)}")

elif analysis_type == "Greeks Visualization":
    st.subheader("Greeks Surface Analysis")
    
    # Input parameters for Greeks visualization
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Stock Price ($)", value=100.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=1.0) / 100
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    with col2:
        strike_range = st.slider("Strike Price Range", 50, 150, (80, 120))
        time_range = st.slider("Time to Maturity Range (years)", 0.1, 2.0, (0.1, 1.0))
    
    # Create meshgrid for surface plot
    strikes = np.linspace(strike_range[0], strike_range[1], 50)
    times = np.linspace(time_range[0], time_range[1], 50)
    K, T = np.meshgrid(strikes, times)
    
    # Calculate Greeks for each point
    Z_delta = np.zeros_like(K)
    Z_gamma = np.zeros_like(K)
    Z_theta = np.zeros_like(K)
    
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            result = bs_calculator.price_and_greeks(
                S=stock_price,
                K=K[i,j],
                T=T[i,j],
                r=0.05,
                sigma=volatility,
                option_type=option_type
            )
            Z_delta[i,j] = result.delta
            Z_gamma[i,j] = result.gamma
            Z_theta[i,j] = result.theta
    
    # Create surface plots
    greek_tabs = st.tabs(["Delta Surface", "Gamma Surface", "Theta Surface"])
    
    with greek_tabs[0]:
        fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z_delta)])
        fig.update_layout(
            title="Delta Surface",
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Time to Maturity",
                zaxis_title="Delta"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with greek_tabs[1]:
        fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z_gamma)])
        fig.update_layout(
            title="Gamma Surface",
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Time to Maturity",
                zaxis_title="Gamma"
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with greek_tabs[2]:
        fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z_theta)])
        fig.update_layout(
            title="Theta Surface",
            scene=dict(
                xaxis_title="Strike Price",
                yaxis_title="Time to Maturity",
                zaxis_title="Theta"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Implied Volatility Surface":
    st.subheader("Implied Volatility Surface")
    
    # Input parameters for IV surface
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Stock Price ($)", value=100.0, min_value=0.01)
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    with col2:
        strike_range = st.slider("Strike Price Range", 50, 150, (80, 120))
        time_range = st.slider("Time to Maturity Range (years)", 0.1, 2.0, (0.1, 1.0))
    
    # Create sample market prices with smile effect
    strikes = np.linspace(strike_range[0], strike_range[1], 20)
    times = np.linspace(time_range[0], time_range[1], 20)
    K, T = np.meshgrid(strikes, times)
    
    # Simulate market prices with volatility smile
    def simulate_market_price(K, S, T):
        moneyness = K/S
        base_vol = 0.2
        smile = 0.1 * (moneyness - 1)**2
        vol = base_vol + smile
        result = bs_calculator.price_and_greeks(S=S, K=K, T=T, r=0.05, sigma=vol, option_type=option_type)
        return result.price
    
    Z_iv = np.zeros_like(K)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            market_price = simulate_market_price(K[i,j], stock_price, T[i,j])
            iv = bs_calculator.implied_volatility(
                price=market_price,
                S=stock_price,
                K=K[i,j],
                T=T[i,j],
                r=0.05,
                option_type=option_type
            )
            Z_iv[i,j] = iv if iv is not None else 0.2
    
    fig = go.Figure(data=[go.Surface(x=K, y=T, z=Z_iv*100)])
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Time to Maturity",
            zaxis_title="Implied Volatility (%)"
        )
    )
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Option Strategy Builder":
    st.subheader("Option Strategy Builder")
    
    # Strategy selection
    strategy = st.selectbox(
        "Select Strategy",
        ["Bull Call Spread", "Bear Put Spread", "Straddle", "Strangle", "Covered Call", "Protective Put"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Stock Price ($)", value=100.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=1.0) / 100
        time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
    
    with col2:
        if strategy in ["Bull Call Spread", "Bear Put Spread"]:
            strike1 = st.number_input("First Strike Price ($)", value=95.0, min_value=0.01)
            strike2 = st.number_input("Second Strike Price ($)", value=105.0, min_value=0.01)
        elif strategy == "Straddle":
            strike = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        elif strategy in ["Covered Call", "Protective Put"]:
            strike = st.number_input("Option Strike Price ($)", value=105.0, min_value=0.01)
        else:  # Strangle
            strike_call = st.number_input("Call Strike Price ($)", value=105.0, min_value=0.01)
            strike_put = st.number_input("Put Strike Price ($)", value=95.0, min_value=0.01)
    
    # Calculate strategy payoff
    price_range = np.linspace(stock_price * 0.5, stock_price * 1.5, 100)
    payoff = np.zeros_like(price_range)
    
    for i, price in enumerate(price_range):
        if strategy == "Bull Call Spread":
            long_call = bs_calculator.price_and_greeks(S=price, K=strike1, T=time_to_maturity, r=0.05, sigma=volatility, option_type="call").price
            short_call = bs_calculator.price_and_greeks(S=price, K=strike2, T=time_to_maturity, r=0.05, sigma=volatility, option_type="call").price
            payoff[i] = long_call - short_call
        elif strategy == "Bear Put Spread":
            long_put = bs_calculator.price_and_greeks(S=price, K=strike1, T=time_to_maturity, r=0.05, sigma=volatility, option_type="put").price
            short_put = bs_calculator.price_and_greeks(S=price, K=strike2, T=time_to_maturity, r=0.05, sigma=volatility, option_type="put").price
            payoff[i] = long_put - short_put
        elif strategy == "Straddle":
            call = bs_calculator.price_and_greeks(S=price, K=strike, T=time_to_maturity, r=0.05, sigma=volatility, option_type="call").price
            put = bs_calculator.price_and_greeks(S=price, K=strike, T=time_to_maturity, r=0.05, sigma=volatility, option_type="put").price
            payoff[i] = call + put
        elif strategy == "Covered Call":
            positions = option_strategies.covered_call(stock_price, strike, time_to_maturity, volatility)
            payoff[i] = option_strategies.calculate_payoff(positions, [price])[0]
        elif strategy == "Protective Put":
            positions = option_strategies.protective_put(stock_price, strike, time_to_maturity, volatility)
            payoff[i] = option_strategies.calculate_payoff(positions, [price])[0]
        else:  # Strangle
            call = bs_calculator.price_and_greeks(S=price, K=strike_call, T=time_to_maturity, r=0.05, sigma=volatility, option_type="call").price
            put = bs_calculator.price_and_greeks(S=price, K=strike_put, T=time_to_maturity, r=0.05, sigma=volatility, option_type="put").price
            payoff[i] = call + put
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_range,
        y=payoff,
        name="Strategy Payoff",
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title=f"{strategy} Payoff Profile",
        xaxis_title="Stock Price ($)",
        yaxis_title="Payoff ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Monte Carlo Simulation":
    st.subheader("Monte Carlo Price Simulation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Initial Stock Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=1.0) / 100
    
    with col2:
        time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
        n_simulations = st.number_input("Number of Simulations", value=1000, min_value=100)
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    # Generate price paths
    dt = 1/252  # Daily steps
    n_steps = int(time_to_maturity * 252)
    times = np.linspace(0, time_to_maturity, n_steps)
    
    paths = np.zeros((n_simulations, n_steps))
    paths[:,0] = stock_price
    
    for i in range(1, n_steps):
        z = np.random.standard_normal(n_simulations)
        paths[:,i] = paths[:,i-1] * np.exp((0.05 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
    
    # Plot some sample paths
    fig = go.Figure()
    
    for i in range(min(10, n_simulations)):  # Plot first 10 paths
        fig.add_trace(go.Scatter(
            x=times,
            y=paths[i,:],
            mode='lines',
            line=dict(width=1),
            showlegend=False
        ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=np.mean(paths, axis=0),
        mode='lines',
        name='Mean Path',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Monte Carlo Stock Price Paths",
        xaxis_title="Time (years)",
        yaxis_title="Stock Price ($)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate option prices from simulations
    final_prices = paths[:,-1]
    if option_type == "call":
        payoffs = np.maximum(final_prices - strike_price, 0)
    else:
        payoffs = np.maximum(strike_price - final_prices, 0)
    
    mc_price = np.exp(-0.05 * time_to_maturity) * np.mean(payoffs)
    bs_price = bs_calculator.price_and_greeks(
        S=stock_price,
        K=strike_price,
        T=time_to_maturity,
        r=0.05,
        sigma=volatility,
        option_type=option_type
    ).price
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monte Carlo Price", f"${mc_price:.4f}")
    with col2:
        st.metric("Black-Scholes Price", f"${bs_price:.4f}")

elif analysis_type == "Delta Hedging Calculator":
    st.subheader("Delta Hedging Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        portfolio_value = st.number_input("Portfolio Value ($)", value=100000.0, min_value=0.01)
        stock_price = st.number_input("Stock Price ($)", value=100.0, min_value=0.01)
        stock_position = st.number_input("Current Stock Position (shares)", value=0)
        
    with col2:
        option_position = st.number_input("Number of Options", value=100)
        strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
        
    volatility = st.slider("Volatility (%)", min_value=1.0, max_value=100.0, value=20.0) / 100
    option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    # Calculate option Greeks
    result = bs_calculator.price_and_greeks(
        S=stock_price,
        K=strike_price,
        T=time_to_maturity,
        r=0.05,
        sigma=volatility,
        option_type=option_type
    )
    
    # Calculate portfolio Greeks
    portfolio_delta = stock_position + option_position * result.delta
    portfolio_gamma = option_position * result.gamma
    portfolio_theta = option_position * result.theta
    portfolio_vega = option_position * result.vega
    
    # Calculate hedge requirements
    shares_to_hedge = -option_position * result.delta
    hedge_cost = abs(shares_to_hedge * stock_price)
    
    # Display results
    st.subheader("Portfolio Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Portfolio Delta", f"{portfolio_delta:.2f}")
        st.metric("Required Stock Position for Delta-Neutral", f"{shares_to_hedge:.0f} shares")
        st.metric("Estimated Hedge Cost", f"${hedge_cost:,.2f}")
    
    with col2:
        st.metric("Portfolio Gamma", f"{portfolio_gamma:.4f}")
        st.metric("Portfolio Theta", f"{portfolio_theta:.4f}")
        st.metric("Portfolio Vega", f"{portfolio_vega:.4f}")
    
    # Add sensitivity analysis
    st.subheader("Hedge Sensitivity Analysis")
    
    price_range = np.linspace(stock_price * 0.8, stock_price * 1.2, 100)
    unhedged_pnl = []
    hedged_pnl = []
    
    for price in price_range:
        # Calculate option P&L
        new_result = bs_calculator.price_and_greeks(
            S=price,
            K=strike_price,
            T=time_to_maturity,
            r=0.05,
            sigma=volatility,
            option_type=option_type
        )
        option_pnl = option_position * (new_result.price - result.price)
        
        # Calculate stock P&L for hedged position
        stock_pnl = shares_to_hedge * (price - stock_price)
        
        unhedged_pnl.append(option_pnl)
        hedged_pnl.append(option_pnl + stock_pnl)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=price_range,
        y=unhedged_pnl,
        name="Unhedged P&L",
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        x=price_range,
        y=hedged_pnl,
        name="Hedged P&L",
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="P&L Analysis: Hedged vs Unhedged Position",
        xaxis_title="Stock Price ($)",
        yaxis_title="Profit/Loss ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Portfolio Risk Analysis":
    st.subheader("Portfolio Risk Analysis")
    
    # Portfolio composition
    st.subheader("Portfolio Composition")
    
    num_options = st.number_input("Number of Different Options", min_value=1, max_value=5, value=2)
    
    total_delta = 0
    total_gamma = 0
    total_theta = 0
    total_vega = 0
    
    options_data = []
    
    for i in range(num_options):
        st.write(f"Option {i+1}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position = st.number_input(f"Position Size (Option {i+1})", value=100)
            strike = st.number_input(f"Strike Price (Option {i+1})", value=100.0, min_value=0.01)
        
        with col2:
            maturity = st.number_input(f"Time to Maturity (Option {i+1}, years)", value=1.0, min_value=0.01)
            option_type = st.selectbox(f"Option Type (Option {i+1})", ["call", "put"])
        
        with col3:
            vol = st.number_input(f"Volatility % (Option {i+1})", value=20.0, min_value=1.0) / 100
        
        result = bs_calculator.price_and_greeks(
            S=100.0,  # Using 100 as reference price
            K=strike,
            T=maturity,
            r=0.05,
            sigma=vol,
            option_type=option_type
        )
        
        total_delta += position * result.delta
        total_gamma += position * result.gamma
        total_theta += position * result.theta
        total_vega += position * result.vega
        
        options_data.append({
            'position': position,
            'strike': strike,
            'maturity': maturity,
            'type': option_type,
            'vol': vol,
            'delta': result.delta,
            'gamma': result.gamma,
            'theta': result.theta,
            'vega': result.vega
        })
    
    # Display portfolio Greeks
    st.subheader("Portfolio Risk Metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio Delta", f"{total_delta:.4f}")
        st.metric("Portfolio Gamma", f"{total_gamma:.4f}")
    
    with col2:
        st.metric("Portfolio Theta", f"{total_theta:.4f}")
        st.metric("Portfolio Vega", f"{total_vega:.4f}")
    
    # Risk visualization
    st.subheader("Risk Visualization")
    
    # Create risk contribution chart
    risk_data = {
        'Option': [f"Option {i+1}" for i in range(num_options)],
        'Delta Contribution': [opt['position'] * opt['delta'] for opt in options_data],
        'Gamma Contribution': [opt['position'] * opt['gamma'] for opt in options_data],
        'Theta Contribution': [opt['position'] * opt['theta'] for opt in options_data],
        'Vega Contribution': [opt['position'] * opt['vega'] for opt in options_data]
    }
    
    risk_metric = st.selectbox("Select Risk Metric", ['Delta', 'Gamma', 'Theta', 'Vega'])
    
    fig = px.bar(
        risk_data,
        x='Option',
        y=f'{risk_metric} Contribution',
        title=f'{risk_metric} Risk Contribution by Option'
    )
    
    st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Dynamic Hedging Simulator":
    st.subheader("Dynamic Hedging Simulator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        initial_stock_price = st.number_input("Initial Stock Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=1.0) / 100
        
    with col2:
        time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
        rebalance_freq = st.selectbox("Rebalancing Frequency", ["Daily", "Weekly", "Monthly"])
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    # Simulation parameters
    n_paths = st.slider("Number of Simulation Paths", min_value=1, max_value=100, value=10)
    
    # Calculate number of steps based on rebalancing frequency
    steps_per_year = {
        "Daily": 252,
        "Weekly": 52,
        "Monthly": 12
    }[rebalance_freq]
    
    n_steps = int(time_to_maturity * steps_per_year)
    dt = time_to_maturity / n_steps
    
    # Generate price paths
    times = np.linspace(0, time_to_maturity, n_steps)
    paths = np.zeros((n_paths, n_steps))
    hedging_errors = np.zeros((n_paths, n_steps))
    paths[:,0] = initial_stock_price
    
    for path in range(n_paths):
        # Generate stock price path
        for i in range(1, n_steps):
            z = np.random.standard_normal()
            paths[path,i] = paths[path,i-1] * np.exp((0.05 - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z)
        
        # Calculate hedging error
        portfolio_value = 0
        prev_delta = 0
        
        for i in range(n_steps):
            result = bs_calculator.price_and_greeks(
                S=paths[path,i],
                K=strike_price,
                T=time_to_maturity - times[i],
                r=0.05,
                sigma=volatility,
                option_type=option_type
            )
            
            if i > 0:
                # Calculate P&L from delta hedging
                stock_pnl = prev_delta * (paths[path,i] - paths[path,i-1])
                portfolio_value += stock_pnl
            
            # Update delta hedge
            prev_delta = -result.delta
            hedging_errors[path,i] = portfolio_value
    
    # Plot results
    fig = go.Figure()
    
    # Plot stock price paths
    for i in range(n_paths):
        fig.add_trace(go.Scatter(
            x=times,
            y=paths[i,:],
            name=f"Path {i+1}",
            line=dict(width=1),
            opacity=0.5
        ))
    
    fig.update_layout(
        title="Stock Price Paths",
        xaxis_title="Time (years)",
        yaxis_title="Stock Price ($)",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot hedging errors
    fig2 = go.Figure()
    
    for i in range(n_paths):
        fig2.add_trace(go.Scatter(
            x=times,
            y=hedging_errors[i,:],
            name=f"Path {i+1}",
            line=dict(width=1),
            opacity=0.5
        ))
    
    fig2.add_trace(go.Scatter(
        x=times,
        y=np.mean(hedging_errors, axis=0),
        name="Mean Hedging Error",
        line=dict(color='red', width=2)
    ))
    
    fig2.update_layout(
        title="Hedging Error Paths",
        xaxis_title="Time (years)",
        yaxis_title="Cumulative Hedging Error ($)",
        showlegend=True
    )
    
    st.plotly_chart(fig2, use_container_width=True)

elif analysis_type == "Hedge Effectiveness":
    st.subheader("Hedge Effectiveness Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stock_price = st.number_input("Current Stock Price ($)", value=100.0, min_value=0.01)
        strike_price = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        volatility = st.number_input("Volatility (%)", value=20.0, min_value=1.0) / 100
        
    with col2:
        time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0, min_value=0.01)
        hedge_period = st.number_input("Hedge Holding Period (days)", value=30, min_value=1)
        option_type = st.radio("Option Type", ["call", "put"], horizontal=True)
    
    # Calculate hedge effectiveness metrics
    price_changes = np.linspace(-0.2, 0.2, 41)  # -20% to +20%
    hedge_effectiveness = []
    unhedged_pnl = []
    hedged_pnl = []
    
    initial_result = bs_calculator.price_and_greeks(
        S=stock_price,
        K=strike_price,
        T=time_to_maturity,
        r=0.05,
        sigma=volatility,
        option_type=option_type
    )
    
    for change in price_changes:
        new_stock_price = stock_price * (1 + change)
        new_time = time_to_maturity - hedge_period/365
        
        new_result = bs_calculator.price_and_greeks(
            S=new_stock_price,
            K=strike_price,
            T=new_time,
            r=0.05,
            sigma=volatility,
            option_type=option_type
        )
        
        # Calculate P&L for one contract
        option_pnl = new_result.price - initial_result.price
        stock_hedge_pnl = -initial_result.delta * (new_stock_price - stock_price)
        
        unhedged_pnl.append(option_pnl)
        hedged_pnl.append(option_pnl + stock_hedge_pnl)
        
        # Calculate hedge effectiveness
        if abs(option_pnl) > 0:
            effectiveness = 1 - abs(option_pnl + stock_hedge_pnl) / abs(option_pnl)
        else:
            effectiveness = 1
        hedge_effectiveness.append(effectiveness)
    
    # Plot results
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_changes * 100,
        y=unhedged_pnl,
        name="Unhedged P&L",
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=price_changes * 100,
        y=hedged_pnl,
        name="Hedged P&L",
        line=dict(color='green')
    ))
    
    fig.update_layout(
        title="P&L Analysis: Hedged vs Unhedged Position",
        xaxis_title="Stock Price Change (%)",
        yaxis_title="Profit/Loss ($)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot hedge effectiveness
    fig2 = go.Figure()
    
    fig2.add_trace(go.Scatter(
        x=price_changes * 100,
        y=np.array(hedge_effectiveness) * 100,
        name="Hedge Effectiveness",
        line=dict(color='blue')
    ))
    
    fig2.update_layout(
        title="Hedge Effectiveness vs Stock Price Change",
        xaxis_title="Stock Price Change (%)",
        yaxis_title="Hedge Effectiveness (%)",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Display summary metrics
    st.subheader("Hedge Effectiveness Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Average Hedge Effectiveness",
            f"{np.mean(hedge_effectiveness)*100:.1f}%"
        )
        st.metric(
            "Maximum Absolute Hedged P&L",
            f"${max(abs(min(hedged_pnl)), abs(max(hedged_pnl))):.2f}"
        )
    
    with col2:
        st.metric(
            "Minimum Hedge Effectiveness",
            f"{min(hedge_effectiveness)*100:.1f}%"
        )
        st.metric(
            "Maximum Absolute Unhedged P&L",
            f"${max(abs(min(unhedged_pnl)), abs(max(unhedged_pnl))):.2f}"
        )

# Add explanation of Greeks
with st.expander("Understanding the Greeks"):
    st.markdown("""
    ### Option Greeks Explained
    
    - **Delta (Δ)**: Measures the rate of change in the option price with respect to the underlying asset's price.
    - **Gamma (Γ)**: Measures the rate of change in delta with respect to the underlying asset's price.
    - **Theta (Θ)**: Measures the rate of change in the option price with respect to time (time decay).
    - **Vega (V)**: Measures the rate of change in the option price with respect to volatility.
    - **Rho (ρ)**: Measures the rate of change in the option price with respect to the risk-free interest rate.
    """) 
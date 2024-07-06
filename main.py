import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps):
    dt = T / num_steps
    paths = np.zeros((num_steps + 1, num_simulations))
    paths[0] = S0
    np.random.seed(42)  # For reproducibility

    for t in range(1, num_steps + 1):
        z = np.random.standard_normal(num_simulations)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

    call_payoffs = np.maximum(paths[-1] - K, 0)
    put_payoffs = np.maximum(K - paths[-1], 0)

    call_price_mc = np.exp(-r * T) * np.mean(call_payoffs)
    put_price_mc = np.exp(-r * T) * np.mean(put_payoffs)

    return call_price_mc, put_price_mc, paths

def black_scholes_option_pricing(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price_bs = S0 * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    put_price_bs = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S0 * stats.norm.cdf(-d1)

    return call_price_bs, put_price_bs, d1, d2

def calculate_greeks(S0, K, T, r, sigma, d1, d2):
    delta_call = stats.norm.cdf(d1)
    delta_put = stats.norm.cdf(d1) - 1

    gamma = stats.norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

    theta_call = (-S0 * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * stats.norm.cdf(d2))
    theta_put = (-S0 * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * stats.norm.cdf(-d2))

    vega = S0 * np.sqrt(T) * stats.norm.pdf(d1)

    rho_call = K * T * np.exp(-r * T) * stats.norm.cdf(d2)
    rho_put = -K * T * np.exp(-r * T) * stats.norm.cdf(-d2)

    return delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put

st.title("Option Pricing and Greeks Calculator")

# Input parameters
S0 = st.number_input("Initial stock price (S0)", value=100.0, step=1.0)
K = st.number_input("Strike price (K)", value=100.0, step=1.0)
T = st.number_input("Time to maturity (T in years)", value=1.0, step=0.1)
r = st.number_input("Risk-free rate (r)", value=0.05, step=0.01)
sigma = st.number_input("Volatility (σ)", value=0.2, step=0.01)
num_simulations = st.number_input("Number of simulations for Monte Carlo", value=10000, step=1000)
num_steps = st.number_input("Number of time steps for Monte Carlo", value=252, step=10)

if st.button("Calculate"):
    # Monte Carlo simulation
    call_price_mc, put_price_mc, paths = monte_carlo_option_pricing(S0, K, T, r, sigma, num_simulations, num_steps)
    # Black-Scholes formula
    call_price_bs, put_price_bs, d1, d2 = black_scholes_option_pricing(S0, K, T, r, sigma)
    # Greeks
    delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put = calculate_greeks(S0, K, T, r, sigma, d1, d2)

    # Display prices in a row with borders
    st.markdown("""
    <div style="display: flex; justify-content: space-around;">
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
            <strong>Monte Carlo Call Price</strong>
            <p style="margin: 0; font-size: 1.2em;">{:.2f}</p>
        </div>
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
            <strong>Monte Carlo Put Price</strong>
            <p style="margin: 0; font-size: 1.2em;">{:.2f}</p>
        </div>
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
            <strong>Black-Scholes Call Price</strong>
            <p style="margin: 0; font-size: 1.2em;">{:.2f}</p>
        </div>
        <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center; margin: 5px;">
            <strong>Black-Scholes Put Price</strong>
            <p style="margin: 0; font-size: 1.2em;">{:.2f}</p>
        </div>
    </div>
    """.format(call_price_mc, put_price_mc, call_price_bs, put_price_bs), unsafe_allow_html=True)

    # Display Greeks in small boxes with borders
    st.markdown("### Option Greeks")
    greek_labels = ["Δ (Delta Call)", "Δ (Delta Put)", "Γ (Gamma)", "Θ (Theta Call)", "Θ (Theta Put)", "V (Vega)", "ρ (Rho Call)", "ρ (Rho Put)"]
    greek_values = [delta_call, delta_put, gamma, theta_call, theta_put, vega, rho_call, rho_put]
    for i in range(0, len(greek_labels), 4):
        greek_cols = st.columns(4)
        for col, label, value in zip(greek_cols, greek_labels[i:i+4], greek_values[i:i+4]):
            col.markdown(f"""
            <div style="border: 2px solid #4CAF50; padding: 10px; border-radius: 5px; text-align: center;">
                <strong>{label}</strong>
                <p style="margin: 0; font-size: 1.2em;">{value:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    # Plot Monte Carlo simulation paths
    st.markdown("### Monte Carlo Simulation Paths")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(num_simulations):  
        ax.plot(paths[:, i], lw=1.5)
    ax.set_title('Monte Carlo Simulation of Stock Price Paths')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Stock Price')
    ax.grid(True)
    ax.set_frame_on(True)
    ax.spines['top'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    st.pyplot(fig)

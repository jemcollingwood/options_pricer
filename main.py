import numpy as np
from scipy.stats import norm


rng = np.random.default_rng(seed=42)

S0    = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset


# ===================================================================================
# Helper functions
# ===================================================================================
def calculate_put_price(S0, K, T, r, C):
    """Calculates the put price P of an option given it has call price C."""
    P = C - S0 + K * np.exp((-r * T))

    return P


def simulate_terminal_prices(S0, T, r, sigma, Z):
    """
    Simulates terminal asset prices S(T) using Geometric Brownian Motion, given
    a numpy array of standard normal draws Z. Vectorised over Z.
    """
    ST = S0 * np.e ** ((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    return ST


# ===================================================================================
# Black-Scholes formula
# ===================================================================================
def black_scholes_call(S0, K, T, r, sigma):
    """Calculates the European call price using the Black-Scholes formula."""
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2)) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    C = S0 * norm.cdf(d_1) - K * np.exp((-r * T)) * norm.cdf(d_2)

    return C


def black_scholes_put(S0, K, T, r, sigma):
    """Calculates the European put price using the Black-Scholes formula."""
    C = black_scholes_put(S0, K, T, r, sigma)
    P = calculate_put_price(S0, K, T, r, C)

    return P


# ===================================================================================
# Naive Monte Carlo option pricer
# ===================================================================================
def naive_mc_call(S0, K, T, r, sigma, N, rng):
    """Estimates the European call price using Monte Carlo methods with *N* paths."""
    Z = rng.standard_normal(N)
    ST = simulate_terminal_prices(S0, T, r, sigma, Z)
    payoffs = np.maximum(ST - K, 0)

    # Find average of all price paths, then apply discount due to interest
    expected_payoff = np.mean(payoffs)
    C = np.exp((-r * T)) * expected_payoff
    
    return C, payoffs

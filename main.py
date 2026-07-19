import numpy as np
from scipy.stats import norm


rng = np.random.default_rng(seed=42)


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
    d_1 = (np.log(S0 / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    C = S0 * norm.cdf(d_1) - K * np.exp((-r * T)) * norm.cdf(d_2)

    return C


def black_scholes_put(S0, K, T, r, sigma):
    """Calculates the European put price using the Black-Scholes formula."""
    C = black_scholes_call(S0, K, T, r, sigma)
    P = calculate_put_price(S0, K, T, r, C)

    return P


# ===================================================================================
# Monte Carlo option pricers
# ===================================================================================
def naive_mc_call(S0, K, T, r, sigma, N, rng: np.random):
    """Estimates the European call price using Monte Carlo methods with *N* paths."""
    Z = rng.standard_normal(N)
    ST = simulate_terminal_prices(S0, T, r, sigma, Z)
    payoffs = np.maximum(ST - K, 0)
    discounted = np.exp((-r * T)) * payoffs
    C = np.mean(discounted)

    return C, payoffs


def control_var_mc_call(S0, K, T, r, sigma, N, rng: np.random):
    """
    Control variates estimator. Uses S(T) as the control, since E(S(T)) =
    S0 * exp(rT) is known exactly.
    """
    Z = rng.standard_normal(N)
    ST = simulate_terminal_prices(S0, T, r, sigma, Z)
    payoffs = np.maximum(ST - K, 0)
    discounted_payoffs = np.exp((-r * T)) * payoffs

    control = ST
    control_mean = S0 * np.exp(r * T) # We know this analytically (see func docstr)

    # Finding optimal c value
    cov = np.cov(discounted_payoffs, control, ddof=1)[0, 1]
    var_control = np.var(control, ddof=1)
    c = cov / var_control

    # Apply control variate
    adjusted_payoffs = discounted_payoffs - c * (control - control_mean)

    C = np.mean(adjusted_payoffs)
    
    return C, adjusted_payoffs


def antithetic_var_call(S0, K, T, r, sigma, N, rng: np.random):
    """
    Antithetic variates estimator. Draws N//2 samples of Z, pairs each with -Z,
    and returns the N//2 paired (independent) discounted payoffs so variance
    calculations in tests.py remain valid.
    """
    Z_half = rng.standard_normal(N // 2)

    ST_pos = simulate_terminal_prices(S0, T, r, sigma, Z_half)
    ST_neg = simulate_terminal_prices(S0, T, r, sigma, -Z_half)

    payoffs_pos = np.maximum(ST_pos - K, 0)
    payoffs_neg = np.maximum(ST_neg - K, 0)

    discounted_pos = np.exp((-r * T)) * payoffs_pos
    discounted_neg = np.exp((-r * T)) * payoffs_neg

    paired_discounted_payoffs = 0.5 * (discounted_pos + discounted_neg)
    C = np.mean(paired_discounted_payoffs)

    # Only return payoff_pos since variance tests assume payoffs are independent,
    # but payoff_pos and payoff_neg are not independent, so var test will be wrong
    return C, paired_discounted_payoffs

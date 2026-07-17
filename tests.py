"""tests.py - Tests and benchmarks for Monte Carlo estimators"""

import numpy as np

from main import black_scholes_call, naive_mc_call


S0    = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

TRUE_PRICE = black_scholes_call(S0, K, T, r, sigma)


# ===================================================================================
# Single method benchmarks
# ===================================================================================
def benchmark_single_method_variance(N, method):
    """Benchmarks the variance of a single method's estimator for the call price."""
    rng_method = np.random.default_rng(seed=123)
    _, payoffs = method(S0, K, T, r, sigma, N, rng_method)

    return np.var(payoffs, ddof = 1) / N


# ===================================================================================
# Benchmark summaries
# ===================================================================================
def benchmark_variance(N=1000):
    """Benchmarks the variance of every method's estimator for the call price."""
    methods = [naive_mc_call]
    result = "--- Variances ---\n"

    for method in methods:
        variance = benchmark_single_method_variance(N, method)
        result += f"{str(method.__name__)}: {variance:.5f}\n"

    print(result)


if __name__ == "__main__":
    benchmark_variance()
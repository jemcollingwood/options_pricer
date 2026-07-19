"""tests.py - Tests and benchmarks for Monte Carlo estimators"""

import numpy as np
from tabulate import tabulate

from main import black_scholes_call, naive_mc_call, control_var_mc_call


S0    = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

TRUE_PRICE = black_scholes_call(S0, K, T, r, sigma)


# ===================================================================================
# Single method benchmarks
# ===================================================================================
def benchmark_single_method_variance(N, method, n_reps=30, base_seed=123):
    """
    Benchmarks the variance of a single method's estimator for the call price.
    
    Averages over *n_reps* to smooth over statistical noise.
    """
    variances = []
    
    for i in range(n_reps):
        rng_method = np.random.default_rng(seed=base_seed + i)
        _, payoffs = method(S0, K, T, r, sigma, N, rng_method)
        variances.append(np.var(payoffs, ddof = 1) / N)
    
    mean_var = np.mean(variances)
    se_var = np.std(variances, ddof=1) / np.sqrt(n_reps)

    return mean_var, se_var


def benchmark_single_method_error(N, method, n_reps=30, base_seed=123):
    """
    Benchmarks the error of a single method's estimator for the call price.
    Compares estimated value to true value, derived from Black-Scholes formula.

    Averages over *n_reps* to smooth over statistical noise.
    """
    prices, errors, ses = [], [], []

    for i in range(n_reps):
        rng_method = np.random.default_rng(seed=base_seed + i)
        C, payoffs = method(S0, K, T, r, sigma, N, rng_method)
        se = np.std(payoffs) / np.sqrt(N)

        prices.append(C)
        errors.append(abs(C - TRUE_PRICE))
        ses.append(se)
    
    mean_price = np.mean(prices)
    mean_error = np.mean(errors)
    se_error = np.std(errors, ddof=1) / np.sqrt(n_reps)

    return mean_price, mean_error, se_error


# ===================================================================================
# Benchmark summaries
# ===================================================================================
def benchmark_variance(N=1000, n_reps=30):
    """Benchmarks the variance of every method's estimator for the call price."""
    methods = [naive_mc_call, control_var_mc_call]
    headers = ["Method", "Mean variance", "Std. error"]
    rows = []

    for method in methods:
        mean_var, se_var = benchmark_single_method_variance(N, method, n_reps)
        rows.append([method.__name__, f"{mean_var:.5f}", f"{se_var:.5f}"])

    print("=" * 61)
    print(f"VARIANCE (mean over {n_reps} reps)")
    print(tabulate(rows, headers, tablefmt="simple"))
    print("=" * 61)


def benchmark_error(N=1000, n_reps=30):
    """
    Benchmarks the error of every method's estimator for the call price.
    Compares estimated value to true value, derived from Black-Scholes formula.
    """
    methods = [naive_mc_call, control_var_mc_call]
    headers = ["Method", "Mean error", "Std. error", "Mean price"]
    rows = []

    for method in methods:
        mean_price, mean_error, se_error = benchmark_single_method_error(N, method)
        rows.append([method.__name__, f"{mean_error:.5f}",
                    f"{se_error:.5f}", f"{mean_price:.5f}"])

    print("=" * 61)      
    print(f"ERROR (mean over {n_reps} reps)")
    print(tabulate(rows, headers, tablefmt="simple"))
    print("=" * 61)


if __name__ == "__main__":
    benchmark_error()
    print()
    benchmark_variance()
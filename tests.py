"""tests.py - Tests and benchmarks for Monte Carlo estimators"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from main import (
    black_scholes_call,
    naive_mc_call,
    control_var_call,
    antithetic_var_call,
    strat_sample_call,
    imp_sample_call
)

S0    = 100  # Current price of underlying asset in GBP
K     = 120  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

TRUE_PRICE = black_scholes_call(S0, K, T, r, sigma)

# List of methods to be benchmarked
methods = [
    naive_mc_call,
    antithetic_var_call,
    control_var_call,
    strat_sample_call,
    imp_sample_call
]

# Used in convergence benchmarks, list of the various N's tested
Ns = [10, 100, 1000, 10000, 100000]


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
        variances.append(np.var(payoffs, ddof = 1) / len(payoffs))
    
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
        se = np.std(payoffs) / np.sqrt(len(payoffs))

        prices.append(C)
        errors.append(abs(C - TRUE_PRICE))
        ses.append(se)
    
    mean_price = np.mean(prices)
    mean_error = np.mean(errors)
    se_error = np.std(errors, ddof=1) / np.sqrt(n_reps)

    return mean_price, mean_error, se_error


def benchmark_single_convergence(method, Ns, n_reps=30, base_seed=123):
    """
    Returns a list containing the mean error of a method (over 30 reps) for each N.
    """
    errors = []

    for N in Ns:
        _, mean_error, _ = benchmark_single_method_error(N, method, n_reps, base_seed)
        errors.append(mean_error)

    return errors


# ===================================================================================
# Benchmark summaries
# ===================================================================================
def benchmark_variance(methods, N=1000, n_reps=30):
    """Benchmarks the variance of every method's estimator for the call price."""
    headers = ["Method", "Mean variance", "Std. error"]
    rows = []

    for method in methods:
        mean_var, se_var = benchmark_single_method_variance(N, method, n_reps)
        rows.append([method.__name__, f"{mean_var:.5f}", f"{se_var:.5f}"])

    print("=" * 61)
    print(f"VARIANCE (mean over {n_reps} reps)")
    print(tabulate(rows, headers, tablefmt="simple"))
    print("=" * 61)


def benchmark_error(methods, N=1000, n_reps=30):
    """
    Benchmarks the error of every method's estimator for the call price.
    Compares estimated value to true value, derived from Black-Scholes formula.
    """
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


def benchmark_convergence(methods, Ns, n_reps=30):
    """
    Benchmarks the convergence rate (error vs number of paths simulated N) of each
    method and plots them together on a loglog graph for comparison.
    """
    fig, ax = plt.subplots()

    for method in methods:
        errors = benchmark_single_convergence(method, Ns, n_reps)
        ax.plot(Ns, errors, label=method.__name__)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    ax.set_xlabel("Number of paths simulated per rep N")
    ax.set_ylabel(f"Mean error over {n_reps} reps")
    ax.set_title("Comparing convergence rates of Monte Carlo methods")

    plt.show()
    

if __name__ == "__main__":
    benchmark_error(methods)
    print()
    benchmark_variance(methods)
    benchmark_convergence(methods, Ns)
"""tests.py - Tests and benchmarks for Monte Carlo estimators"""

import statistics

from main import call_naive_mc


S_0   = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset


def benchmark_variance(N, num_paths, method):
    """Benchmarks a method's estimator variance for *N* trials, each trial using *num_paths* paths."""
    estimator_list = []

    for _ in range(N):
        estimator_list.append(method(S_0, K, T, r, sigma, num_paths))

    return statistics.variance(estimator_list)


print(benchmark_variance(1000, 1000, call_naive_mc))
"""tests.py - Tests and benchmarks for Monte Carlo estimators"""

import statistics

from main import call_naive_mc


S_0   = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset


def benchmark_single_method_variance(N, num_estimators, method):
    """Benchmarks a method's estimator variance for *num_estimators*, each estimator using *N* paths."""
    estimator_list = []

    for _ in range(num_estimators):
        C = method(S_0, K, T, r, sigma, N)
        estimator_list.append(C)

    return statistics.variance(estimator_list)


def benchmark_single_method_bias(method):
    """Benchmarks a method's estimator bias compared to Black-Scholes formula."""
    pass


print(benchmark_single_method_variance(1000, 2000, call_naive_mc))
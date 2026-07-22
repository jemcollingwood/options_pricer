"""benchmarks.py - Benchmarks for Monte Carlo estimators."""

import matplotlib.pyplot as plt
import numpy as np
import time
from tabulate import tabulate

from main import (
    black_scholes_call,
    naive_mc_call,
    control_var_call,
    antithetic_var_call,
    strat_sample_call,
    imp_sample_call,
)

S0 = 100  # Current price of underlying asset in GBP
K = 120  # Strike price of the contract in GBP
T = 1  # Time to maturity in years
r = 0.05  # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

TRUE_PRICE = black_scholes_call(S0, K, T, r, sigma)


# List of methods to be benchmarked
methods = [
    naive_mc_call,
    antithetic_var_call,
    control_var_call,
    strat_sample_call,
    imp_sample_call,
]

# Used in convergence benchmarks, list of the various N's tested
Ns = [10, 100, 1000, 10000, 100000]


# ===================================================================================
# Single method benchmarks
# ===================================================================================
def benchmark_single_method_variance(
    N: int, method: function, n_reps: int = 300, base_seed: int = 123
) -> tuple[float, float]:
    """
    Benchmarks the variance of a single method's estimator for the call price.

    Averages over *n_reps* to smooth over statistical noise.
    """
    prices = []

    for i in range(n_reps):
        rng_method = np.random.default_rng(seed=base_seed + i)
        C, _ = method(S0, K, T, r, sigma, N, rng_method)
        prices.append(C)

    mean_var = np.var(prices, ddof=1)
    se_mean_price = np.std(prices, ddof=1) / np.sqrt(n_reps)

    return mean_var, se_mean_price


def benchmark_single_method_error(
    N: int, method: function, n_reps: int = 300, base_seed: int = 123
) -> tuple[float, float, float, float]:
    """
    Benchmarks the error of a single method's estimator for the call price.
    Compares estimated value to true value, derived from Black-Scholes formula.

    Averages over *n_reps* to smooth over statistical noise.
    """
    prices, errors, ses = [], [], []

    for i in range(n_reps):
        rng_method = np.random.default_rng(seed=base_seed + i)
        C, payoffs = method(S0, K, T, r, sigma, N, rng_method)
        se_theoretical = np.std(payoffs, ddof=1) / np.sqrt(len(payoffs))

        prices.append(C)
        errors.append(abs(C - TRUE_PRICE))
        ses.append(se_theoretical)

    mean_price = np.mean(prices)
    mean_error = np.mean(errors)
    se_error = np.std(errors, ddof=1) / np.sqrt(n_reps)
    mean_se_theoretical = np.mean(ses)

    return mean_price, mean_error, se_error, mean_se_theoretical


def benchmark_single_convergence(
    method: function, Ns: list[int], n_reps: int = 300, base_seed: int = 123
) -> list[float]:
    """
    Returns a list containing the mean error of a method (over 30 reps) for each N.
    """
    errors = []

    for N in Ns:
        _, mean_error, _, _ = benchmark_single_method_error(
            N, method, n_reps, base_seed
        )
        errors.append(mean_error)

    return errors


def benchmark_single_efficiency(
    N: int, method: function, n_reps: int = 300, base_seed: int = 123
) -> float:
    """
    Benchmarks the efficiency of a single method i.e. mean variance * mean time.
    """
    mean_time = benchmark_mean_time(N, method, n_reps, base_seed)
    mean_var, _ = benchmark_single_method_variance(N, method, n_reps, base_seed)

    return mean_var * mean_time


def benchmark_mean_time(
    N: int, method: function, n_reps: int = 300, base_seed: int = 123
) -> float:
    """
    Returns the mean time of a method to estimate an option price with *N* paths.
    """
    times = []

    for i in range(n_reps):
        rng_method = np.random.default_rng(seed=base_seed + i)
        start = time.perf_counter()
        method(S0, K, T, r, sigma, N, rng_method)
        times.append(time.perf_counter() - start)

    return np.mean(times)


# ===================================================================================
# Benchmark summaries
# ===================================================================================
def benchmark_variance(
    methods: list[function], N: int = 1000, n_reps: int = 300
) -> None:
    """Benchmarks the variance of every method's estimator for the call price."""
    headers = ["Method", "Mean variance"]
    rows = []

    for method in methods:
        mean_var, _ = benchmark_single_method_variance(N, method, n_reps)
        rows.append([method.__name__, f"{mean_var:.4f}"])

    print(f"=== VARIANCE (mean over {n_reps} reps) ===")
    print(tabulate(rows, headers, tablefmt="simple"))


def benchmark_error(
        methods: list[function], N: int = 1000, n_reps: int = 300
) -> None:
    """
    Benchmarks the error of every method's estimator for the call price.
    Compares estimated value to true value, derived from Black-Scholes formula.
    """
    headers = ["Method", "Mean price", "Mean error", "Std. error"]
    rows = []

    for method in methods:
        mean_price, mean_error, se_error, _ = benchmark_single_method_error(
            N, method, n_reps
        )

        rows.append(
            [
                method.__name__,
                f"{mean_price:.4f}",
                f"{mean_error:.4f}",
                f"{se_error:.4f}",
            ]
        )

    print(f"=== ERROR (mean over {n_reps} reps) ===")
    print(tabulate(rows, headers, tablefmt="simple"))


def benchmark_convergence(
    methods: list[function], Ns: list[int], n_reps: int = 300
) -> None:
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

    fig.savefig("convergence.png", dpi=150, bbox_inches="tight")
    plt.show()


def benchmark_efficiency(
    methods: list[function], N: int = 1000, n_reps: int = 300
) -> None:
    """
    Benchmarks the efficiency (mean_var * mean_time) of each method. Compares relative
    to naive Monte Carlo.
    """
    headers = ["Method", "Mean variance", "Mean time (s)", "Efficiency (rel. to naive)"]
    rows = []

    naive_var, _ = benchmark_single_method_variance(N, naive_mc_call, n_reps)
    naive_time = benchmark_mean_time(N, naive_mc_call, n_reps)
    naive_eff = naive_var * naive_time

    for method in methods:
        mean_var, _ = benchmark_single_method_variance(N, method, n_reps)
        mean_time = benchmark_mean_time(N, method, n_reps)
        efficiency = mean_var * mean_time

        rows.append(
            [
                method.__name__,
                f"{mean_var:.4f}",
                f"{mean_time:.7f}",
                f"{naive_eff / efficiency:.2f}",  # >1 means more efficient than naive
            ]
        )

    print(f"=== EFFICIENCY (mean over {n_reps} reps) ===")
    print(tabulate(rows, headers, tablefmt="simple"))


def compare_theoretical_vs_empirical_se(
    methods: list[function], N: int = 1000, n_reps: int = 300
) -> None:
    """
    Compares the theoretical vs empirical std. error for each method.
    """
    headers = [
        "Method",
        "Theoretical SE (single-rep)",
        "Empirical SE (cross-reps)",
        "Ratio",
    ]
    rows = []

    for method in methods:
        _, _, _, mean_se_theoretical = benchmark_single_method_error(N, method, n_reps)
        mean_var, _ = benchmark_single_method_variance(N, method, n_reps)
        empirical_se = np.sqrt(mean_var)
        ratio = mean_se_theoretical / empirical_se

        rows.append(
            [
                method.__name__,
                f"{mean_se_theoretical:.4f}",
                f"{empirical_se:.4f}",
                f"{ratio:.2f}",
            ]
        )

    print(f"=== THEORETICAL vs EMPIRICAL SE OF VARIANCE (N = {N}, {n_reps} reps) ===")
    print(tabulate(rows, headers, tablefmt="simple"))


if __name__ == "__main__":
    benchmark_error(methods, n_reps=300)
    benchmark_variance(methods, n_reps=300)
    compare_theoretical_vs_empirical_se(methods)
    benchmark_efficiency(methods, n_reps=300)

    benchmark_convergence(methods, Ns, n_reps=300)

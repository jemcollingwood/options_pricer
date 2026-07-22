"""test_main.py - Simple tests for the Black-Scholes formula and MC estimators."""

import numpy as np

from main import (
    black_scholes_call,
    black_scholes_put,
    calculate_put_price,
    simulate_terminal_prices,
    stratified_normal,
    naive_mc_call,
    control_var_call,
    antithetic_var_call,
    strat_sample_call,
    imp_sample_call,
)

S0, K, T, r, sigma = 100, 120, 1, 0.05, 0.2
TRUE_PRICE = black_scholes_call(S0, K, T, r, sigma)

# Methods whose raw payoffs are always non-negative (control variates is
# excluded, since its adjusted payoffs can legitimately go negative)
methods = [naive_mc_call, control_var_call, antithetic_var_call, strat_sample_call, imp_sample_call]
non_negative_methods = [naive_mc_call, antithetic_var_call, strat_sample_call, imp_sample_call]


def test_black_scholes_put_call_parity():
    C = black_scholes_call(S0, K, T, r, sigma)
    P = black_scholes_put(S0, K, T, r, sigma)
    expected_P = calculate_put_price(S0, K, T, r, C)
    assert abs(P - expected_P) < 1e-9


def test_simulate_terminal_prices_zero_shock():
    # With Z=0, S(T) should equal the deterministic drift term
    ST = simulate_terminal_prices(S0, T, r, sigma, np.array([0.0]))
    expected = S0 * np.exp((r - 0.5 * sigma**2) * T)
    assert abs(ST[0] - expected) < 1e-9


def test_stratified_normal_is_increasing_and_centred():
    rng = np.random.default_rng(0)
    Z = stratified_normal(10_000, rng)
    assert np.all(np.diff(Z) > 0)
    assert abs(Z.mean()) < 0.05


def test_antithetic_output_length():
    rng = np.random.default_rng(0)
    C, payoffs = antithetic_var_call(S0, K, T, r, sigma, 10_000, rng)
    assert len(payoffs) == 5_000


def test_estimators_converge_to_black_scholes():
    for method in methods:
        rng = np.random.default_rng(1)
        C, _ = method(S0, K, T, r, sigma, 200_000, rng)
        assert abs(C - TRUE_PRICE) < 0.1


def test_estimators_are_reproducible_with_fixed_seed():
    for method in methods:
        rng1 = np.random.default_rng(7)
        rng2 = np.random.default_rng(7)
        C1, _ = method(S0, K, T, r, sigma, 5_000, rng1)
        C2, _ = method(S0, K, T, r, sigma, 5_000, rng2)
        assert C1 == C2


def test_prices_are_non_negative():
    for method in methods:
        rng = np.random.default_rng(2)
        C, _ = method(S0, K, T, r, sigma, 5_000, rng)
        assert C >= 0


def test_raw_payoffs_are_non_negative():
    for method in non_negative_methods:
        rng = np.random.default_rng(2)
        _, payoffs = method(S0, K, T, r, sigma, 5_000, rng)
        assert np.all(payoffs >= -1e-9)


def test_variance_reduction_vs_naive():
    n_reps, N = 100, 1_000

    for method in [control_var_call, antithetic_var_call, strat_sample_call, imp_sample_call]:
        naive_prices, method_prices = [], []

        for i in range(n_reps):
            seed = 100 + i
            rng_a = np.random.default_rng(seed)
            rng_b = np.random.default_rng(seed)
            naive_prices.append(naive_mc_call(S0, K, T, r, sigma, N, rng_a)[0])
            method_prices.append(method(S0, K, T, r, sigma, N, rng_b)[0])

        assert np.var(method_prices, ddof=1) < np.var(naive_prices, ddof=1)

import statistics
import math


S_0   = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

norm = statistics.NormalDist()


# ===================================================================================
# Helper functions
# ===================================================================================
def calculate_put_price(S_0, K, T, r, C):
    """Calculates the put price P of an option given it has call price C."""
    P = C - S_0 + K * math.e ** (-r * T)

    return P


# ===================================================================================
# Black-Scholes formula
# ===================================================================================
def call_black_scholes(S_0, K, T, r, sigma):
    """Calculates the European call price using the Black-Scholes formula."""
    d_1 = (math.log(S_0 / K) + (r + sigma ** 2 / 2)) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)
    C = S_0 * norm.cdf(d_1) - K * (math.e ** (-r * T)) * norm.cdf(d_2)

    return C


def put_black_scholes(S_0, K, T, r, sigma):
    """Calculates the European put price using the Black-Scholes formula."""
    C = call_black_scholes(S_0, K, T, r, sigma)
    P = calculate_put_price(S_0, K, T, r, C)

    return P


# ===================================================================================
# Naive Monte Carlo option pricer
# ===================================================================================
def call_naive_mc(S_0, K, T, r, sigma, N):
    """Estimates the European call price using Monte Carlo methods with *N* paths."""
    payoff_list = []

    # Simulate N possible price paths, then calculate payoff for each path
    for _ in range(N):
        Z = norm.samples(1)[0]
        S_T = S_0 * math.e ** ((r - sigma ** 2 / 2) * T + sigma * math.sqrt(T) * Z)
        payoff = max(S_T - K, 0)

        payoff_list.append(payoff)

    # Find average of all price paths, then apply discount due to interest
    expected_payoff = sum(payoff_list) / len(payoff_list)
    C = math.e ** (-r * T) * expected_payoff

    return C

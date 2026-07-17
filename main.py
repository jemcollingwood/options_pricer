import statistics
import math


S_0   = 100  # Current price of underlying asset in GBP
K     = 100  # Strike price of the contract in GBP
T     = 1    # Time to maturity in years
r     = 0.05 # Risk free interest rate
sigma = 0.2  # Volatility of underlying asset

N = statistics.NormalDist()


def call_black_scholes(S_0, K, T, r, sigma):
    """Calculates the European call price using the Black-Scholes formula."""
    d_1 = (math.log(S_0 / K) + (r + (sigma ** 2) / 2)) / (sigma * math.sqrt(T))
    d_2 = d_1 - sigma * math.sqrt(T)
    C = S_0 * N.cdf(d_1) - K * (math.e ** (-r * T)) * N.cdf(d_2)

    return C


def put_black_scholes(S_0, K, T, r, sigma):
    """Calculates the European put price using the Black-Scholes formula."""
    C = call_black_scholes(S_0, K, T, r, sigma)
    P = C - S_0 + K * math.e ** (-r * T)

    return P


print(f"European call price is: {call_black_scholes(S_0, K, T, r, sigma)}")
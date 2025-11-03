import numpy as np
from scipy.stats import norm

# Standard normal cumulative distribution function
N = norm.cdf

class BlackScholes:
    """
    A simple implementation of the Black-Scholes-Merton model for pricing European options
    and calculating Greeks. This is a sample and can be replaced with more advanced models.
    """

    @staticmethod
    def calculate_d1(S, K, T, r, sigma):
        """Calculate d1 term."""
        if T == 0 or sigma == 0:
            return float('inf') if S > K else float('-inf')
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def calculate_d2(d1, T, sigma):
        """Calculate d2 term."""
        return d1 - sigma * np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Calculate European call option price."""
        if T <= 0:
            return max(0, S - K)
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholes.calculate_d2(d1, T, sigma)
        return S * N(d1) - K * np.exp(-r * T) * N(d2)

    @staticmethod
    def put_price(S, K, T, r, sigma):
        """Calculate European put option price."""
        if T <= 0:
            return max(0, K - S)
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholes.calculate_d2(d1, T, sigma)
        return K * np.exp(-r * T) * N(-d2) - S * N(-d1)

    # --- GREEKS ---
    
    @staticmethod
    def delta(S, K, T, r, sigma, option_type='call'):
        """Calculate option Delta."""
        if T <= 0:
            if option_type == 'call':
                return 1 if S > K else 0
            else: # put
                return -1 if S < K else 0
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        if option_type == 'call':
            return N(d1)
        else: # put
            return N(d1) - 1

    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Calculate option Gamma."""
        if T <= 0 or sigma == 0:
            return 0
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Calculate option Vega."""
        if T <= 0:
            return 0
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100 # per 1% change in IV

    @staticmethod
    def theta(S, K, T, r, sigma, option_type='call'):
        """Calculate option Theta."""
        if T <= 0:
            return 0
        d1 = BlackScholes.calculate_d1(S, K, T, r, sigma)
        d2 = BlackScholes.calculate_d2(d1, T, sigma)
        p1 = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
        if option_type == 'call':
            p2 = r * K * np.exp(-r * T) * N(d2)
            return (p1 - p2) / 365 # per day
        else: # put
            p2 = r * K * np.exp(-r * T) * N(-d2)
            return (p1 + p2) / 365 # per day

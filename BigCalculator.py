import numpy as np
from scipy.stats import norm

def calculate_d1(S, K, T, r, sigma, q):
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def calculate_d2(d1, sigma, T):
    return d1 - sigma * np.sqrt(T)

def forward_price(S0, r, T, q=0.0):
    """
    Compute forward price of an asset under continuous compounding.
    """
    return S0 * np.exp((r - q) * T)

def bs_price(S, K, T, r, sigma, q=0.0, option_type='call', position='long'):
    """
    Black-Scholes price of a European call or put.
    """
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price if position == 'long' else -price

def delta(S, K, T, r, sigma, q=0.0, option_type='call', position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    if option_type == 'call':
        delta_val = np.exp(-q * T) * norm.cdf(d1)
    elif option_type == 'put':
        delta_val = -np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return delta_val if position == 'long' else -delta_val

def gamma(S, K, T, r, sigma, q=0.0, position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    gamma_val = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma_val if position == 'long' else -gamma_val

def vega(S, K, T, r, sigma, q=0.0, position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    vega_val = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100  # per 1%
    return vega_val if position == 'long' else -vega_val

def theta(S, K, T, r, sigma, q=0.0, option_type='call', position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    term1 = -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
    if option_type == 'call':
        term2 = q * S * norm.cdf(d1) * np.exp(-q * T)
        term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_val = term1 + term2 + term3
    elif option_type == 'put':
        term2 = -q * S * norm.cdf(-d1) * np.exp(-q * T)
        term3 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_val = term1 + term2 + term3
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    theta_val /= 365  # Per day
    return theta_val if position == 'long' else -theta_val

def vanna(S, K, T, r, sigma, q=0.0, position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    vanna_val = np.exp(-q * T) * norm.pdf(d1) * (-d2) / sigma / 100
    return vanna_val if position == 'long' else -vanna_val

def volga(S, K, T, r, sigma, q=0.0, position='long'):
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(d1, sigma, T)
    raw_vega = vega(S, K, T, r, sigma, q, position='long') * 100  # undo /100
    volga_val = raw_vega * d1 * d2 / sigma / 100
    return volga_val if position == 'long' else -volga_val

# Example usage
if __name__ == "__main__":
    S = 750       # Spot price
    K = 700       # Strike price
    T = 1         # Time to maturity (in years)
    r = 0.00      # Risk-free rate
    sigma = 0.20  # Volatility
    q = 0.00      # Dividend yield

    print(f"Forward Price: {forward_price(S, r, T, q):.2f}\n")

    for opt_type in ['call', 'put']:
        for pos in ['long', 'short']:
            price = bs_price(S, K, T, r, sigma, q, option_type=opt_type, position=pos)
            d = delta(S, K, T, r, sigma, q, option_type=opt_type, position=pos)
            g = gamma(S, K, T, r, sigma, q, position=pos)
            v = vega(S, K, T, r, sigma, q, position=pos)
            t = theta(S, K, T, r, sigma, q, option_type=opt_type, position=pos)
            va = vanna(S, K, T, r, sigma, q, position=pos)
            vo = volga(S, K, T, r, sigma, q, position=pos)
            print(f"{pos.capitalize()} {opt_type.capitalize()} Greeks and Price:")
            print(f"  BS Price: {price:.4f}")
            print(f"  Delta   : {d:.4f}")
            print(f"  Gamma   : {g:.6f}")
            print(f"  Vega    : {v:.4f}")
            print(f"  Theta   : {t:.4f} per day")
            print(f"  Vanna   : {va:.4f}")
            print(f"  Volga   : {vo:.4f}")
            print("-" * 40)

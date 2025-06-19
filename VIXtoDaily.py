# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 16:07:29 2025

@author: ward_
"""

import math
from scipy.stats import norm

def vix_to_daily_volatility(vix):
    """
    Convert VIX from annualized % to daily standard deviation (in decimal).
    Assumes 252 trading days per year.
    """
    return (vix / 100) / math.sqrt(252)

def probability_less_than_return(vix, daily_return):
    """
    Calculate the probability that the daily log return is less than the given return,
    assuming normal distribution centered at 0 with std dev from VIX.
    
    Parameters:
    - vix: VIX level (e.g., 20 means 20% annualized volatility)
    - daily_return: daily **log return** (e.g., -0.01 for -1%)
    
    Returns:
    - probability: float between 0 and 1
    """
    daily_vol = vix_to_daily_volatility(vix)
    prob = norm.cdf(daily_return, loc=0, scale=daily_vol)
    return prob

# Example usage
if __name__ == "__main__":
    vix = float(input("Enter VIX (% annualized volatility): "))
    r = float(input("Enter daily log return (e.g., -0.01 for -1%): "))
    
    prob = probability_less_than_return(vix, r)
    print(f"Probability that log return is less than {r:.8f} given VIX={vix}%: {prob:.8%}")

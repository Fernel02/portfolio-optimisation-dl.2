import numpy as np
import pandas as pd

def sortino_ratio(returns, required_return=0):
    """Calculates the Sortino ratio."""
    # Ensure returns is a pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
        
    downside_returns = returns[returns < required_return]
    if len(downside_returns) == 0:
        return np.nan
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    if downside_deviation == 0:
        return np.nan
    # Annualize the ratio (assuming monthly data)
    return (returns.mean() * 12) / (downside_deviation * np.sqrt(12))

def max_drawdown(returns):
    """Calculates the maximum drawdown."""
    # Ensure returns is a pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
        
    cumulative = np.exp(returns.cumsum())
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def calmar_ratio(returns):
    """Calculates the Calmar ratio."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    # Annualize the ratio (assuming monthly data)
    return (returns.mean() * 12) / abs(mdd)

def cvar(returns, alpha=0.95):
    """Calculates the Conditional Value at Risk (CVaR)."""
    # Ensure returns is a pandas Series
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
        
    var = returns.quantile(1 - alpha)
    return returns[returns <= var].mean()
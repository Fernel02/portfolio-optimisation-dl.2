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

def sharpe_ratio(returns, risk_free=0):
    """Sharpe ratio: excess return / total risk."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    excess = returns - risk_free/12  # monthly adjustment
    return (excess.mean() * 12) / (excess.std() * np.sqrt(12))

def treynor_ratio(returns, benchmark, risk_free=0):
    """Treynor ratio: excess return / beta."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(benchmark, pd.Series):
        benchmark = pd.Series(benchmark)
    excess = returns - risk_free/12
    bench_excess = benchmark - risk_free/12
    cov = np.cov(excess, bench_excess)[0, 1]
    var_bench = np.var(bench_excess)
    beta = cov / var_bench if var_bench != 0 else np.nan
    return (excess.mean() * 12) / beta if beta != 0 else np.nan

def information_ratio(returns, benchmark):
    """Information ratio: active return / active risk."""
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    if not isinstance(benchmark, pd.Series):
        benchmark = pd.Series(benchmark)
    active = returns - benchmark
    return (active.mean() * 12) / (active.std() * np.sqrt(12))
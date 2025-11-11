"""
Standardized annualized metrics for portfolio evaluation.

All functions assume daily return data and use 252 trading days per year.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Union


def annualized_metrics(
    returns: Union[pd.Series, pd.DataFrame],
    weights: Union[pd.Series, dict, None] = None,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Compute standardized annualized metrics from daily returns.
    
    Args:
        returns: Daily returns as Series (single asset) or DataFrame (multiple assets)
        weights: Optional portfolio weights (if returns is DataFrame). Dict or Series with asset names as keys/index.
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default: 0.0)
    
    Returns:
        dict with keys:
            - CAGR: Compound Annual Growth Rate
            - Volatility: Annualized volatility (std dev)
            - Sharpe: Annualized Sharpe ratio
            - MaxDD: Maximum drawdown (as negative percentage)
            - N: Number of observations
            - Start: First date
            - End: Last date
    
    Notes:
        - CAGR: (final_value / initial_value) ** (252 / N) - 1
        - Volatility: daily_std * sqrt(252)
        - Sharpe: (CAGR - rf) / Volatility
        - MaxDD: min((cumulative - running_max) / running_max)
    """
    if returns is None or len(returns) == 0:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "N": 0,
            "Start": None,
            "End": None,
        }
    
    # Handle DataFrame with weights
    if isinstance(returns, pd.DataFrame):
        if weights is None:
            # Equal weight
            weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
        elif isinstance(weights, dict):
            weights = pd.Series(weights)
        
        # Align weights with returns columns
        weights = weights.reindex(returns.columns).fillna(0.0)
        weights = weights / weights.sum() if weights.sum() > 0 else weights
        
        # Compute portfolio returns
        port_returns = (returns * weights).sum(axis=1)
    else:
        port_returns = returns.copy()
    
    # Drop NaN
    port_returns = port_returns.dropna()
    
    if len(port_returns) == 0:
        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "MaxDD": np.nan,
            "N": 0,
            "Start": None,
            "End": None,
        }
    
    N = len(port_returns)
    
    # CAGR: Compound annual growth rate
    cumulative = (1 + port_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1.0
    years = N / 252.0
    if years > 0 and cumulative.iloc[-1] > 0:
        cagr = (cumulative.iloc[-1]) ** (1.0 / years) - 1.0
    else:
        cagr = np.nan
    
    # Volatility: Annualized standard deviation
    vol = port_returns.std() * np.sqrt(252)
    
    # Sharpe: (CAGR - rf) / Volatility
    if vol > 0 and not np.isnan(cagr):
        sharpe = (cagr - risk_free_rate) / vol
    else:
        sharpe = np.nan
    
    # MaxDD: Maximum drawdown
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else np.nan
    
    return {
        "CAGR": float(cagr),
        "Volatility": float(vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "N": int(N),
        "Start": port_returns.index[0] if hasattr(port_returns.index[0], 'strftime') else str(port_returns.index[0]),
        "End": port_returns.index[-1] if hasattr(port_returns.index[-1], 'strftime') else str(port_returns.index[-1]),
    }


def beta_to_benchmark(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    min_overlap: int = 252
) -> float:
    """
    Compute portfolio beta relative to benchmark.
    
    Args:
        returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns
        min_overlap: Minimum overlapping observations required
    
    Returns:
        Beta coefficient (covariance / benchmark variance)
    """
    # Align and drop NaN
    aligned = pd.DataFrame({"port": returns, "bench": benchmark_returns}).dropna()
    
    if len(aligned) < min_overlap:
        return np.nan
    
    cov = aligned["port"].cov(aligned["bench"])
    var = aligned["bench"].var()
    
    if var == 0 or np.isnan(var):
        return np.nan
    
    return float(cov / var)


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Compute Value at Risk (VaR) at given confidence level.
    
    Args:
        returns: Daily returns series
        confidence: Confidence level (default: 0.95 for 95% VaR)
        method: "historical" (empirical quantile) or "parametric" (normal assumption)
    
    Returns:
        VaR as a negative percentage (e.g., -0.02 means 2% loss at confidence level)
    """
    returns = returns.dropna()
    
    if len(returns) == 0:
        return np.nan
    
    if method == "historical":
        var = returns.quantile(1 - confidence)
    elif method == "parametric":
        from scipy import stats
        mean = returns.mean()
        std = returns.std()
        var = stats.norm.ppf(1 - confidence, loc=mean, scale=std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    return float(var)


def calmar_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float:
    """
    Compute Calmar ratio: CAGR / abs(MaxDD).
    
    Args:
        returns: Daily returns series
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Calmar ratio (higher is better)
    """
    metrics = annualized_metrics(returns, risk_free_rate=risk_free_rate)
    cagr = metrics["CAGR"]
    max_dd = metrics["MaxDD"]
    
    if np.isnan(cagr) or np.isnan(max_dd) or max_dd == 0:
        return np.nan
    
    return float(cagr / abs(max_dd))


__all__ = [
    "annualized_metrics",
    "beta_to_benchmark",
    "value_at_risk",
    "calmar_ratio",
    "beta_vs_benchmark",
    "var_95",
]

# Convenience aliases required by V3 UI (minimal surface change)
def beta_vs_benchmark(p: pd.Series, b: pd.Series) -> float:
    """Alias for beta_to_benchmark with sensible overlap default (252)."""
    try:
        return beta_to_benchmark(p, b, min_overlap=126)
    except Exception:
        return float("nan")


def var_95(r: pd.Series) -> float:
    """Historical 95% VaR (negative tail) as a thin wrapper over value_at_risk."""
    try:
        return value_at_risk(r, confidence=0.95, method="historical")
    except Exception:
        return float("nan")


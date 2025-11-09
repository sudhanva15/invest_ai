"""Portfolio simulation and metrics computation utilities."""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from .constraints import apply_weight_constraints


def cumprod_from_returns(r: pd.Series) -> pd.Series:
    """Convert return series to cumulative growth (starting at 1.0)"""
    return (1 + r.fillna(0)).cumprod()


def quick_metrics(cum: pd.Series, years: int) -> Dict[str, float]:
    """Compute key metrics (CAGR, MaxDD, Sharpe) for a window of the cumulative curve"""
    if len(cum) < 2:
        return {"CAGR": np.nan, "MaxDD": np.nan, "Sharpe": np.nan}
        
    # Get window of specified years (prefer last() for DatetimeIndex)
    w = cum.last(f"{years*365}D") if hasattr(cum.index, "freq") else cum.iloc[-252*years:]
    if len(w) < 2:
        return {"CAGR": np.nan, "MaxDD": np.nan, "Sharpe": np.nan}
    
    # CAGR = (end/start)^(1/years) - 1
    cagr = (w.iloc[-1] / w.iloc[0])**(1/years) - 1
    
    # MaxDD = min(price/peak - 1)
    dd = (w / w.cummax() - 1).min()
    
    # Sharpe = sqrt(252) * mean(daily_ret) / std(daily_ret)
    r = w.pct_change().dropna()
    sharpe = (r.mean()*252) / (r.std()*np.sqrt(252)) if r.std() > 0 else np.nan
    
    return {
        "CAGR": round(float(cagr), 4),
        "MaxDD": round(float(dd), 4),
        "Sharpe": round(float(sharpe), 2)
    }


def compute_portfolio_curve(prices_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Compute portfolio equity curve from prices and weights.
    
    Args:
        prices_df: DataFrame with prices (columns = symbols)
        weights: Dict mapping symbol -> weight
        
    Returns:
        Series with cumulative portfolio value (starts at 1.0)
    """
    # Convert weights to aligned vector
    w = pd.Series(weights)
    w = w.reindex(prices_df.columns).fillna(0)
    
    # Compute returns and portfolio curve
    rets = prices_df.pct_change()
    port_ret = rets.fillna(0) @ w
    return cumprod_from_returns(port_ret)
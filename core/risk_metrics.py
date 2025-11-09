import numpy as np, pandas as pd

def annualize_return(returns: pd.Series, periods_per_year=252):
    mu = (1 + returns).prod() ** (periods_per_year/len(returns)) - 1
    return mu

def annualize_vol(returns: pd.Series, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def max_drawdown(returns: pd.Series):
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = (wealth/peak) - 1.0
    return dd.min()

def sharpe(returns: pd.Series, rf=0.0, periods_per_year=252):
    ex = returns - (rf/periods_per_year)
    vol = ex.std() * np.sqrt(periods_per_year)
    if vol == 0: return np.nan
    mu = ex.mean() * periods_per_year
    return mu/vol

def _portfolio_series(returns_df: pd.DataFrame | pd.Series, weights=None) -> pd.Series:
    """
    Convert a DataFrame of asset returns into a single portfolio return series.
    - If `weights` is provided (list/Series/dict), it will be aligned to columns and normalized.
    - If no weights are provided and `returns_df` is a DataFrame, use equal-weights across available columns each date.
    """
    if isinstance(returns_df, pd.Series):
        return returns_df.dropna()

    df = returns_df.copy()
    # Coerce to numeric and drop all-NaN rows
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")

    if df.empty:
        return pd.Series(dtype=float)

    if weights is None:
        # Equal-weight across non-NaN columns for each date
        w = pd.Series(1.0, index=df.columns, dtype=float)
    else:
        if isinstance(weights, dict):
            w = pd.Series(weights, dtype=float)
        else:
            w = pd.Series(weights, index=df.columns[:len(weights)], dtype=float)
        # align to columns
        w = w.reindex(df.columns)
        # replace negatives/NaN with 0 to avoid accidental shorting
        w = w.fillna(0.0).clip(lower=0.0)

    # Normalize weights to sum to 1 over the columns that exist
    s = w.sum()
    if s > 0:
        w = w / s
    else:
        # fall back to equal-weight if weights all zero
        w = pd.Series(1.0, index=df.columns, dtype=float)

    # Compute weighted sum; handle NaNs by normalizing active weight each row
    # (i.e., re-normalize over non-NaN assets for each timestamp)
    w_active = w.where(~df.isna(), other=0.0)
    w_norm = w_active.div(w_active.sum(axis=1).replace(0.0, pd.NA), axis=0)
    port = (df.fillna(0.0) * w_norm.fillna(0.0)).sum(axis=1)

    return port.dropna()

# === [WIRING PATCH] Risk bundle ============================================
def summarize_risk(
    returns_df: pd.DataFrame | pd.Series,
    weights=None,
    rfr: float = 0.015,
    periods_per_year: int = 252
) -> dict:
    """
    Summarize key risk/return metrics for a portfolio or single return series.
    Returns keys:
      - ann_return: annualized geometric return (CAGR)
      - ann_vol: annualized volatility
      - sharpe: Sharpe ratio (rf annualized; converted to per-period internally)
      - max_dd: maximum drawdown
      - n_obs: number of return observations
      - n_assets: number of columns (if DataFrame), else 1
    """
    out = {}
    try:
        series = _portfolio_series(returns_df, weights=weights)
        if series is None or series.empty:
            return {"ann_return": None, "ann_vol": None, "sharpe": None, "max_dd": None, "n_obs": 0, "n_assets": 0}

        out["ann_return"] = annualize_return(series, periods_per_year=periods_per_year)
        out["ann_vol"] = annualize_vol(series, periods_per_year=periods_per_year)
        out["sharpe"] = sharpe(series, rf=rfr, periods_per_year=periods_per_year)
        out["max_dd"] = max_drawdown(series)
        out["n_obs"] = int(series.shape[0])
        out["n_assets"] = int(returns_df.shape[1]) if isinstance(returns_df, pd.DataFrame) else 1
        return out
    except Exception:
        # Defensive: never crash callers; return partials if available
        try:
            out.setdefault("n_assets", int(returns_df.shape[1]) if isinstance(returns_df, pd.DataFrame) else 1)
        except Exception:
            out.setdefault("n_assets", 0)
        return out
# ===========================================================================

__all__ = ["annualize_return", "annualize_vol", "max_drawdown", "sharpe", "summarize_risk"]

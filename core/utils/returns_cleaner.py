from __future__ import annotations
"""Utility functions for robust price->returns cleaning.

This module provides a single primary helper:

clean_prices_to_returns(prices: pd.DataFrame, winsor_p: float = 0.005, min_non_na: int = 252) -> pd.DataFrame

Design Goals:
- Convert a wide price DataFrame (index = datetime-like, columns = symbols) into clean daily returns.
- Remove pathological values (infinities, columns with too little data) prior to optimization or risk modeling.
- Apply symmetric winsorization to reduce the impact of extreme outliers.
- Guarantee the output contains only finite numeric returns with ascending datetime index.

No external dependencies beyond pandas & numpy; pure preprocessing.
"""

from typing import Optional
import pandas as pd
import numpy as np

__all__ = ["clean_prices_to_returns"]

def clean_prices_to_returns(
    prices: pd.DataFrame,
    winsor_p: float = 0.005,
    min_non_na: int = 252,
) -> pd.DataFrame:
    """Convert wide prices to a cleaned daily returns DataFrame.

    Parameters
    ----------
    prices : pd.DataFrame
        Wide price frame with DatetimeIndex (or convertible). Columns represent symbols/assets.
        Non-numeric columns are ignored automatically.
    winsor_p : float, default 0.005
        Symmetric tail proportion for winsorization. Each column is clipped to
        [quantile(winsor_p), quantile(1 - winsor_p)]. Use small values (e.g. 0.001 - 0.01).
    min_non_na : int, default 252
        Minimum number of non-NaN observations required for a column to be retained.
        Helps ensure enough history for annualization assumptions.

    Returns
    -------
    pd.DataFrame
        Cleaned daily returns with only finite numeric values:
        - Columns meeting minimum data threshold
        - Tail-clipped (winsorized) returns
        - No ±inf values
        - Rows with any NaN removed
        - Sorted ascending by index

    Notes
    -----
    The function avoids forward-filling or imputing missing prices. Rows with all NaN returns
    after pct_change are dropped early; columns with insufficient data are removed prior to winsorization.
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame()

    # Ensure datetime index
    out = prices.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
        except Exception:
            pass
    out = out.sort_index()

    # Keep only numeric columns (cast coercively)
    out = out.select_dtypes(include=["number"]).astype("float64")
    if out.empty:
        return pd.DataFrame()

    # Compute raw returns (no implicit fill)
    rets = out.pct_change(fill_method=None)

    # Replace infinities with NaN
    rets = rets.replace([np.inf, -np.inf], np.nan)

    # Drop rows that are entirely NaN
    rets = rets.dropna(how="all")

    if rets.empty:
        return pd.DataFrame()

    # Drop columns lacking sufficient non-NaN observations
    ok_cols = [c for c in rets.columns if rets[c].count() >= int(min_non_na)]
    rets = rets[ok_cols]
    if rets.empty:
        return pd.DataFrame()

    # Winsorize each column (symmetric clipping)
    if winsor_p and 0 < winsor_p < 0.5:
        q_low = rets.quantile(winsor_p)
        q_high = rets.quantile(1 - winsor_p)
        # Clip per-column using aligned quantiles
        for c in rets.columns:
            lo = q_low.get(c, None)
            hi = q_high.get(c, None)
            if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
                rets[c] = rets[c].clip(lower=float(lo), upper=float(hi))

    # Final sanitization: drop any rows with non-finite values
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna(how="any")

    # Ensure ascending index and consistent dtype
    rets = rets.sort_index()
    rets = rets.astype("float64")

    return rets
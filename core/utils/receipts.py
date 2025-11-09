"""Utilities for building data quality receipts for financial time series.

This module provides tools for analyzing price data quality, coverage, and provenance
across multiple data sources. It handles ragged indices, missing data, and produces
normalized quality metrics.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame requirements.
    
    Args:
        df: Price DataFrame to validate
        
    Raises:
        ValueError: If DataFrame doesn't meet requirements
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    if df.empty:
        raise ValueError("DataFrame cannot be empty")

def compute_coverage_metrics(series: pd.Series, full_index: pd.DatetimeIndex) -> Dict[str, Any]:
    """Compute coverage and quality metrics for a price series.
    
    Args:
        series: Price series to analyze
        full_index: Complete index spanning full date range
        
    Returns:
        Dict with coverage metrics (nan_rate, dates, points)
    """
    if series.empty:
        return {
            "first": None,
            "last": None,
            "nan_rate": 1.0,
            "n_points": 0,
            "hist_years": None
        }
        
    # Get date range
    first = series.index.min().strftime("%Y-%m-%d")
    last = series.index.max().strftime("%Y-%m-%d")
    
    # Calculate metrics
    expected_points = len(full_index)
    actual_points = len(series.dropna())
    nan_rate = max(0.0, 1.0 - (actual_points / expected_points))
    years = (pd.to_datetime(last) - pd.to_datetime(first)).days / 365.25
    
    return {
        "first": first,
        "last": last,
        "nan_rate": round(nan_rate, 4),
        "n_points": actual_points,
        "hist_years": round(years, 1)
    }

def compute_return_metrics(series: pd.Series) -> Dict[str, Optional[float]]:
    """Compute return-based metrics if sufficient data exists.
    
    Args:
        series: Price series to analyze
        
    Returns:
        Dict with volatility and Sharpe ratio (None if insufficient data)
    """
    if len(series) < 252:  # Require at least 1yr
        return {"ann_vol": None, "sharpe": None}
        
    rets = series.pct_change().dropna()
    vol = rets.std() * np.sqrt(252)
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    
    return {
        "ann_vol": round(vol, 3),
        "sharpe": round(sharpe, 2)
    }

def build_receipts(
    tickers_or_df: Union[List[str], pd.DataFrame], 
    df_or_provenance: Optional[Union[pd.DataFrame, Dict[str, Dict[str, int]], tuple]] = None
) -> List[Dict[str, Any]]:
    """Build detailed data quality receipts for each ticker.

    Analyzes price data quality, coverage and performance metrics for each ticker,
    handling ragged indices and missing data appropriately. Includes provider
    tracking and backfill statistics.

    Supported usage patterns (backward compatible):
        receipts = build_receipts(tickers, df)                      # legacy form
        receipts = build_receipts(tickers, (df, prov))              # legacy with tuple
        receipts = build_receipts(df)                                # new form (tickers from df.columns)
        receipts = build_receipts(df, provenance_dict)              # new form with explicit prov

    Args:
        tickers_or_df: Either a list of ticker symbols (legacy) OR a DataFrame (new style)
        df_or_provenance: Either a DataFrame (legacy) OR provenance dict/tuple (new style) OR None
            
        When first arg is list: 
            - Second arg must be DataFrame or (DataFrame, provenance_dict) tuple
        When first arg is DataFrame:
            - Second arg is optional provenance (dict or tuple) or None (reads df.attrs)
            
        Provenance can be Dict with keys {provider_map, backfill_pct, coverage}
        or Tuple (df, provenance_dict). If None, reads from df.attrs or df._provider_map.

    Returns:
        List[Dict[str, Any]] — one dict per ticker with keys:
            ticker, provider, backfill_pct, first, last, nan_rate,
            n_points, hist_years, ann_vol, sharpe (None if insufficient data).

    Raises:
        ValueError: If input validation fails.
    """
    # Signature detection: distinguish build_receipts(tickers, df) from build_receipts(df, prov)
    if isinstance(tickers_or_df, list):
        # Legacy form: build_receipts(tickers, df) or build_receipts(tickers, (df, prov))
        tickers = tickers_or_df
        if isinstance(df_or_provenance, tuple) and len(df_or_provenance) == 2:
            df, prov_dict = df_or_provenance
        elif isinstance(df_or_provenance, pd.DataFrame):
            df = df_or_provenance
            prov_dict = None
        else:
            raise ValueError("When first arg is list, second must be DataFrame or tuple(DataFrame, dict)")
    elif isinstance(tickers_or_df, pd.DataFrame):
        # New form: build_receipts(df) or build_receipts(df, prov)
        df = tickers_or_df
        tickers = None  # Will auto-derive from df.columns
        if isinstance(df_or_provenance, tuple) and len(df_or_provenance) == 2:
            _, prov_dict = df_or_provenance  # Unpack tuple
        elif isinstance(df_or_provenance, dict):
            prov_dict = df_or_provenance
        elif df_or_provenance is None:
            prov_dict = None
        else:
            raise ValueError("When first arg is DataFrame, second must be dict, tuple, or None")
    else:
        raise ValueError("First argument must be either list of tickers or DataFrame")
    
    validate_dataframe(df)
    
    # Auto-derive tickers from df.columns if not provided (new form)
    if tickers is None:
        tickers = list(df.columns)
    
    # Get metadata with safe defaults (prefer explicit provenance, fallback to attrs)
    if prov_dict:
        pmap = prov_dict.get("provider_map", {}) or {}
        pfill = prov_dict.get("backfill_pct", {}) or {}
        cov = prov_dict.get("coverage", {}) or {}
    else:
        pmap = getattr(df, "_provider_map", df.attrs.get("provider_map", {})) or {}
        pfill = getattr(df, "_backfill_pct", df.attrs.get("backfill_pct", {})) or {}
        cov = getattr(df, "_coverage", df.attrs.get("coverage", {})) or {}
    
    out = []
    full_index = df.index
    
    for ticker in tickers:
        # Get clean price series
        series = df[ticker].dropna() if ticker in df.columns else pd.Series(dtype=float)
        
        # Get coverage dates from metadata or compute them
        first, last = None, None
        if ticker in cov and cov[ticker]:
            cov_entry = cov[ticker]
            if isinstance(cov_entry, (list, tuple)) and len(cov_entry) == 2:
                first, last = cov_entry
            
        # Compute metrics
        coverage = compute_coverage_metrics(series, pd.DatetimeIndex(full_index))
        metrics = compute_return_metrics(series)
        
        # Build receipt
        receipt = {
            "ticker": ticker,
            "provider": pmap.get(ticker, "stooq(+tiingo?)"),
            "backfill_pct": f"{float(pfill.get(ticker, 0.0)):.2f}",
            **coverage,
            **metrics
        }
        out.append(receipt)
        
    # Return list-of-dicts (callers expecting DataFrame can wrap manually)
    return out
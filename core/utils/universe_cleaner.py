"""Universe pre-cleaning for simulation/validation.

This module provides helpers to expand and filter the effective universe
before running portfolio simulations, allowing assets with partial histories
while maintaining data quality standards.
"""

from __future__ import annotations
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

__all__ = ["preclean_universe_for_simulation"]


def preclean_universe_for_simulation(
    prices: pd.DataFrame,
    k_days: int = 1260,
    min_non_na: int = 126,
    min_overlap_pct: float = 0.70,
    verbose: bool = False
) -> Tuple[pd.DataFrame, dict]:
    """Pre-clean and align universe for simulation while retaining partial histories.
    
    Strategy:
        1. Identify trailing k_days window
        2. For each symbol, check if it has >= min_non_na observations in that window
        3. Require that kept symbols have >= min_overlap_pct common dates
        4. Return aligned price DataFrame and diagnostics
    
    Parameters
    ----------
    prices : pd.DataFrame
        Wide price DataFrame (index=dates, columns=symbols)
    k_days : int, default 1260
        Lookback window in trading days (~5 years)
    min_non_na : int, default 126
        Minimum non-NaN observations per symbol in the window (~6 months)
    min_overlap_pct : float, default 0.70
        Minimum fraction of dates that must be common across kept symbols
    verbose : bool, default False
        Print diagnostic information
    
    Returns
    -------
    cleaned_prices : pd.DataFrame
        Aligned price DataFrame with symbols meeting criteria
    diagnostics : dict
        Contains 'dropped_symbols', 'kept_symbols', 'window', 'common_dates'
    """
    if prices is None or prices.empty:
        return pd.DataFrame(), {
            "dropped_symbols": [],
            "kept_symbols": [],
            "window": (None, None),
            "common_dates": 0
        }
    
    # Sort and ensure datetime index
    df = prices.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()].sort_index()
        except Exception:
            pass
    
    # Define trailing window
    if len(df) > k_days:
        window_start = df.index[-k_days]
        df_window = df.loc[window_start:]
    else:
        window_start = df.index[0] if len(df) > 0 else None
        df_window = df
    
    if df_window.empty:
        return pd.DataFrame(), {
            "dropped_symbols": list(prices.columns),
            "kept_symbols": [],
            "window": (window_start, None),
            "common_dates": 0
        }
    
    window_end = df_window.index[-1]
    
    # Check each symbol for sufficient data
    kept_symbols = []
    dropped_symbols = []
    
    for col in df_window.columns:
        non_na_count = df_window[col].notna().sum()
        if non_na_count >= min_non_na:
            kept_symbols.append(col)
        else:
            dropped_symbols.append((col, f"only_{non_na_count}_obs"))
            if verbose:
                print(f"[preclean] Dropping {col}: {non_na_count} < {min_non_na} observations")
    
    if not kept_symbols:
        return pd.DataFrame(), {
            "dropped_symbols": dropped_symbols,
            "kept_symbols": [],
            "window": (window_start, window_end),
            "common_dates": 0
        }
    
    # Check overlap requirement
    df_kept = df_window[kept_symbols]
    # Count dates where at least one symbol has data
    any_data_mask = df_kept.notna().any(axis=1)
    total_potential_dates = any_data_mask.sum()
    
    # Count dates where ALL kept symbols have data
    all_data_mask = df_kept.notna().all(axis=1)
    common_dates = all_data_mask.sum()
    
    overlap_ratio = common_dates / total_potential_dates if total_potential_dates > 0 else 0.0
    
    if verbose:
        print(f"[preclean] Window: {window_start} to {window_end}")
        print(f"[preclean] Kept {len(kept_symbols)} symbols with sufficient data")
        print(f"[preclean] Common dates: {common_dates}/{total_potential_dates} ({overlap_ratio:.1%})")
    
    # If overlap is too low, try removing symbols with most missing data
    if overlap_ratio < min_overlap_pct and len(kept_symbols) > 3:
        if verbose:
            print(f"[preclean] Overlap {overlap_ratio:.1%} < {min_overlap_pct:.1%}, pruning sparse symbols...")
        
        # Calculate coverage for each symbol
        coverage = {col: (df_kept[col].notna().sum() / len(df_kept)) for col in kept_symbols}
        # Sort by coverage descending
        sorted_syms = sorted(coverage.items(), key=lambda x: x[1], reverse=True)
        
        # Iteratively keep best symbols until overlap requirement met
        for n in range(len(sorted_syms), 2, -1):
            test_syms = [s for s, _ in sorted_syms[:n]]
            test_df = df_window[test_syms]
            test_all = test_df.notna().all(axis=1).sum()
            test_any = test_df.notna().any(axis=1).sum()
            test_overlap = test_all / test_any if test_any > 0 else 0.0
            
            if test_overlap >= min_overlap_pct:
                # Accept this subset
                newly_dropped = [s for s in kept_symbols if s not in test_syms]
                for s in newly_dropped:
                    dropped_symbols.append((s, f"low_overlap_{coverage[s]:.1%}"))
                kept_symbols = test_syms
                df_kept = test_df
                common_dates = test_all
                overlap_ratio = test_overlap
                if verbose:
                    print(f"[preclean] Pruned to {n} symbols with overlap {test_overlap:.1%}")
                break
    
    diagnostics = {
        "dropped_symbols": dropped_symbols,
        "kept_symbols": kept_symbols,
        "window": (window_start, window_end),
        "common_dates": int(common_dates),
        "overlap_ratio": float(overlap_ratio)
    }
    
    return df_kept, diagnostics

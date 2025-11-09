from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict
import os
from pathlib import Path
import pandas as pd

from core.data_sources.router_smart import fetch_union as _fetch_union
from core.data_sources.provider_registry import get_ordered_providers as _get_providers

# Ensure API keys are loaded (prefer .env, then config/credentials.env)
try:
    from core.utils.env_tools import load_env_once as _load_env
    _load_env(".env")
    if not os.getenv("TIINGO_API_KEY") and Path("config/credentials.env").exists():
        _load_env("config/credentials.env")
except Exception:
    pass

def _series_from_ohlcv(df: pd.DataFrame) -> pd.Series:
    if df is None or len(df) == 0:
        return pd.Series(dtype="float64")
    d = df.copy()
    # standardize columns
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d.columns:
        # nothing we can do
        return pd.Series(dtype="float64")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d[d["date"].notna()].sort_values("date").drop_duplicates("date", keep="last")
    # prefer adj_close, else close, else price
    val_col = "adj_close" if "adj_close" in d.columns else ("close" if "close" in d.columns else ("price" if "price" in d.columns else None))
    if val_col is None:
        return pd.Series(dtype="float64")
    return pd.Series(d[val_col].values, index=pd.DatetimeIndex(d["date"].values, name="date"))

def _fetch_one(symbol: str, start: Optional[str], end: Optional[str], attempts_list: list | None = None):
    # use union merge (leftmost provider wins on overlaps, later ones extend history)
    merged, provenance = _fetch_union(symbol=symbol, attempts=attempts_list, start=start, end=end, return_provenance=True)
    return merged, provenance

def get_prices_with_provenance(
    symbols: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str,int]]]:
    """
    Returns a (prices_df, provenance_by_symbol) tuple.
    prices_df: index=date, columns=symbols, values=adj_close (or close/price fallback).
    provenance_by_symbol: {symbol: {provider_name: row_count}}
    """
    symbols = [s.strip().upper() for s in symbols if s and str(s).strip()]
    frames = []
    prov_all: Dict[str, Dict[str,int]] = {}
    provider_map = {}
    backfill_pct = {}
    coverage = {}
    
    for sym in symbols:
        attempts: list = []
        df, prov = _fetch_one(sym, start, end, attempts_list=attempts)
        if df is not None and len(df) > 0:
            # Extract metadata before converting to series
            provider_map.update(df.attrs.get("provider_map", {}))
            backfill_pct.update(df.attrs.get("backfill_pct", {}))
            coverage.update(df.attrs.get("coverage", {}))
            
            s = _series_from_ohlcv(df).rename(sym)
            if len(s) > 0:
                frames.append(s)
                prov_all[sym] = prov
        else:
            prov_all[sym] = {}
            
    if frames:
        out = pd.concat(frames, axis=1).sort_index()
        # Attach collected metadata
        out.attrs["provider_map"] = provider_map
        out.attrs["backfill_pct"] = backfill_pct
        out.attrs["coverage"] = coverage
        return out, prov_all
        
    return pd.DataFrame(), prov_all

def get_prices(
    symbols: Iterable[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper that returns just the price DataFrame (wide, adj_close preferred).
    """
    df, _ = get_prices_with_provenance(symbols, start=start, end=end, force=force)
    return df

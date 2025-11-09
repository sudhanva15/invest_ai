from __future__ import annotations
import os
from typing import Callable, Optional, Tuple
import pandas as pd

# Lightweight provider wrappers so we can hand back simple callables
def _stooq_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    from core.data_sources import stooq
    try:
        return stooq.fetch_daily(symbol, start=start, end=end, force=force)
    except TypeError:
        return stooq.fetch_daily(symbol)

def _tiingo_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    from core.data_sources.backfill_tiingo import fetch_tiingo_history
    return fetch_tiingo_history(symbol, start=start, end=end)

def _yfinance_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    # used as a late fallback only
    import datetime as _dt
    import yfinance as yf
    params = {}
    if start and start not in ("earliest", ""):
        params["start"] = str(start)
    if end:
        params["end"] = str(end)
    try:
        df = yf.download(symbol, auto_adjust=False, progress=False, **params)
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    df = df.reset_index().rename(columns={
        "Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"
    })
    return df

def detect_provider_env() -> dict:
    """Report which keys are present (for the UI debug panel)."""
    return {
        "TIINGO_API_KEY_present": bool(os.environ.get("TIINGO_API_KEY")),
        "FRED_API_KEY_present": bool(os.environ.get("FRED_API_KEY")),
        # Polygon has a few variants; your .env provides ACCESS/SECRET which we surface as present
        "POLYGON_API_KEY_present": bool(os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_API_TOKEN")),
        "env_aliases": {
            "POLYGON_API_KEY": bool(os.environ.get("POLYGON_API_KEY")),
            "POLYGON_API_TOKEN": bool(os.environ.get("POLYGON_API_TOKEN")),
            "POLYGON_ACCESS_KEY_ID": bool(os.environ.get("POLYGON_ACCESS_KEY_ID")),
            "POLYGON_SECRET_ACCESS_KEY": bool(os.environ.get("POLYGON_SECRET_ACCESS_KEY")),
        }
    }

def get_ordered_providers() -> list[Callable]:
    """
    Provider precedence: Stooq primary, Tiingo backfill (union merge). yfinance optional/legacy.
    Each provider returns a pandas.DataFrame with at least 'date' and 'close'/'adj_close'.
    """
    return [_tiingo_fetch, _stooq_fetch, _yfinance_fetch]

def router_fetch_daily(symbol: str, asset_class: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None, force: bool = False) -> Tuple[pd.DataFrame, str]:
    """
    Legacy-compatible router: try Tiingo, then Stooq, then yfinance; return (df, provenance_str).
    Used as a final fallback by router_smart._legacy_router().
    """
    prov = []
    for name, fn in (("tiingo", _tiingo_fetch), ("stooq", _stooq_fetch), ("yfinance", _yfinance_fetch)):
        try:
            df = fn(symbol, start=start, end=end, force=force)
        except Exception:
            df = None
        if df is not None and len(df) > 0:
            prov.append(name)
            df = df.copy()
            if "ticker" not in df.columns:
                df["ticker"] = symbol
            return df, "+".join(prov)
        prov.append(f"{name}:0")
    return pd.DataFrame(), "+".join(prov)

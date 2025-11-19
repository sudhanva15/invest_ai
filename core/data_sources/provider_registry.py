from __future__ import annotations
import os
from typing import Callable, Optional, Tuple
import pandas as pd
from core.env_tools import is_demo_mode
from core.demo_data import load_demo_price_history

from core.data_sources.yf_source import fetch_yfinance_history

DEMO_MODE = is_demo_mode()

# Lightweight provider wrappers so we can hand back simple callables
def _stooq_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    from core.data_sources import stooq
    if DEMO_MODE:
        return load_demo_price_history(symbol, start=start, end=end)
    try:
        return stooq.fetch_daily(symbol, start=start, end=end, force=force)
    except TypeError:
        return stooq.fetch_daily(symbol)

def _tiingo_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    if DEMO_MODE:
        return load_demo_price_history(symbol, start=start, end=end)
    from core.data_sources.backfill_tiingo import fetch_tiingo_history
    return fetch_tiingo_history(symbol, start=start, end=end)

def _yfinance_fetch(symbol: str, start: Optional[str] = None, end: Optional[str] = None, force: bool = False):
    if DEMO_MODE:
        return load_demo_price_history(symbol, start=start, end=end)
    df = fetch_yfinance_history(symbol, start_date=start, end_date=end)
    if df is None or df.empty:
        return None
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
    """Return providers in preferred precedence order.

    Architecture V3: Stooq primary, Tiingo backfill for depth/gaps, yfinance late fallback.
    If a Tiingo rate-limit has been detected we skip adding Tiingo to avoid hammering.
    """
    providers: list[Callable] = []
    # Stooq primary
    providers.append(_stooq_fetch)
    if not DEMO_MODE:
        # Conditional Tiingo backfill
        try:
            from core.data_sources.tiingo import tiingo_rate_limited
            if not tiingo_rate_limited() or os.getenv("TIINGO_SKIP_ON_RATE_LIMIT", "1") != "1":
                providers.append(_tiingo_fetch)
        except Exception:
            providers.append(_tiingo_fetch)
        # yfinance legacy fallback
        providers.append(_yfinance_fetch)
    return providers

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

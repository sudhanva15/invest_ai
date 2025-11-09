from __future__ import annotations
import os, time, json
import pandas as pd
from datetime import datetime
from fredapi import Fred

CACHE_DIR = os.path.join("data","macro")
os.makedirs(CACHE_DIR, exist_ok=True)

# Default macro menu (you can add more):
DEFAULT_SERIES = {
    "CPI (CPIAUCSL)": "CPIAUCSL",
    "Fed Funds Rate (FEDFUNDS)": "FEDFUNDS",
    "10Y Treasury Yield (DGS10)": "DGS10",
    "Unemployment Rate (UNRATE)": "UNRATE",
    "Industrial Production (INDPRO)": "INDPRO",
}

# Exported symbols for menu discoverability and UI/routers
__all__ = ("fetch_series", "load_series", "load_macro", "load_macro_latest", "load_macro_filled", "DEFAULT_SERIES")

def _fred():
    key = os.environ.get("FRED_API_KEY")
    return Fred(api_key=key) if key else Fred()

# Helper to coerce values to numeric safely
def _to_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce values to numeric, keeping NaN for non-convertible entries."""
    return pd.to_numeric(s, errors="coerce")

def _cache_path(series_id: str) -> str:
    return os.path.join(CACHE_DIR, f"{series_id}.csv")

def fetch_series(series_id: str, start: str | None=None, end: str | None=None) -> pd.Series:
    """
    Fetch a single FRED series with csv cache; returns pandas Series (Date index).
    - Uses 24h file cache in data/macro/{series_id}.csv
    - Coerces values to numeric
    - Gracefully falls back to empty Series on hard failure
    """
    cp = _cache_path(series_id)
    try:
        if os.path.exists(cp) and (time.time() - os.path.getmtime(cp) < 24*3600):
            s = pd.read_csv(cp, parse_dates=["date"], index_col="date")["value"]
        else:
            fred = _fred()
            data = fred.get_series(series_id)  # pandas Series with DatetimeIndex
            s = pd.Series(data)
            s.index.name = "date"
            s.name = series_id
            df = s.sort_index().to_frame("value")
            df.to_csv(cp)
            s = df["value"]
        if start: s = s[s.index >= pd.to_datetime(start)]
        if end:   s = s[s.index <= pd.to_datetime(end)]
        return _to_numeric_series(s)
    except Exception:
        # On any provider/cache/network error, return a well-formed empty Series
        return pd.Series(name=series_id, dtype="float64")

# UI/router compatibility alias
def load_series(series_id: str, start: str | None=None, end: str | None=None) -> pd.Series:
    """Alias for router/UI compatibility; delegates to fetch_series."""
    return fetch_series(series_id, start=start, end=end)

def load_macro(series_map: dict[str,str] | None=None, start: str | None=None, end: str | None=None) -> pd.DataFrame:
    """Load multiple FRED series into one DataFrame (daily index via forward-fill)."""
    series_map = series_map or DEFAULT_SERIES
    frames = []
    for label, sid in series_map.items():
        s = fetch_series(sid, start, end).rename(label)
        frames.append(s)
    df = pd.concat(frames, axis=1).sort_index()
    # Many FRED series are monthly; make them daily for overlay by ffill
    df = df.resample("D").ffill()
    return df


def load_macro_latest():
    """Return the most recent macro snapshot (one row) where at least one series is non-null."""
    df = load_macro()
    # Keep only rows where ANY column is non-null
    mask = df.notna().any(axis=1)
    df = df.loc[mask]
    if df.empty:
        return df  # nothing to show; caller can handle
    return df.iloc[[-1]]


def load_macro_filled():
    """Business-day macro table forward-filled from release dates; drops leading/trailing all-NaN blocks."""
    df = load_macro()
    # Trim to rows where at least one value exists (drop all-NaN rows at ends)
    mask = df.notna().any(axis=1)
    df = df.loc[mask]
    if df.empty:
        return df
    # Forward-fill across business days (safe fallback if index isn't fixed freq)
    try:
        df = df.asfreq("B").ffill()
    except Exception:
        df = df.ffill()
    return df

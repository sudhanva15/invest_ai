

from __future__ import annotations
import os
import io
import pandas as pd
import requests

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"

def _get_token() -> str:
    """
    Return the Tiingo API token from environment.
    Enforced nomenclature: TIINGO_API_KEY
    """
    token = os.getenv("TIINGO_API_KEY")
    if not token:
        raise RuntimeError("TIINGO_API_KEY not set in environment")
    return token

def _normalize_tiingo_df(raw_csv_text: str) -> pd.DataFrame:
    """
    Normalize Tiingo CSV into our canonical schema:
    [date, open, high, low, close, adj_close, volume]
    """
    df = pd.read_csv(io.StringIO(raw_csv_text))
    if df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    # Tiingo CSV columns typically:
    # date,close,high,low,open,volume,adjClose,adjHigh,adjLow,adjOpen,adjVolume,divCash,splitFactor
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]).dt.date,
        "open": df.get("open"),
        "high": df.get("high"),
        "low": df.get("low"),
        "close": df.get("close"),
        "adj_close": df.get("adjClose", df.get("close")),
        "volume": df.get("volume"),
    }).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return out

def fetch_daily(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Fetch daily OHLCV for a symbol using Tiingo REST.
    Returns DataFrame with columns: date, open, high, low, close, adj_close, volume
    """
    params = {"format": "csv", "token": _get_token()}
    if start:
        params["startDate"] = start  # YYYY-MM-DD
    if end:
        params["endDate"] = end

    url = TIINGO_BASE.format(ticker=symbol.lower())
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return _normalize_tiingo_df(r.text)

# --- Registry/Router compatibility ------------------------------------------

def load_daily(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Alias used by provider registry and routers.
    """
    return fetch_daily(symbol, start=start, end=end)

def fetch(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """
    Generic fetch alias; mirrors load_daily for compatibility.
    """
    return fetch_daily(symbol, start=start, end=end)

__all__ = ["fetch_daily", "load_daily", "fetch"]

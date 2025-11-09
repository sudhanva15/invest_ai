
import os
import io
import requests
import pandas as pd

TIINGO_URL_TMPL = "https://api.tiingo.com/tiingo/daily/{symbol}/prices"

def _tiingo_key():
    key = os.getenv("TIINGO_API_KEY","").strip()
    if not key:
        raise RuntimeError("TIINGO_API_KEY missing in environment")
    return key

def fetch_tiingo_daily(symbol: str, start: str = "1980-01-01", end: str | None = None) -> pd.DataFrame:
    params = {
        "startDate": start,
        "resampleFreq": "daily",
        "format": "csv",
        "token": _tiingo_key(),
    }
    if end: params["endDate"] = end
    url = TIINGO_URL_TMPL.format(symbol=symbol.lower())
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    if not r.text.strip():
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    df = pd.read_csv(io.StringIO(r.text))
    # Standardize columns
    colmap = {
        "date":"date","open":"open","high":"high","low":"low","close":"close",
        "adjClose":"adj_close","volume":"volume"
    }
    for c in colmap:
        if c not in df.columns and colmap[c] in df.columns:
            pass  # already ok
    # Some tiingo CSVs use 'adjClose', others 'adjClose' exists; just map robustly:
    if "adjClose" in df.columns and "adj_close" not in df.columns:
        df = df.rename(columns={"adjClose":"adj_close"})
    if "adj_close" not in df.columns:
        df["adj_close"] = df.get("close")
    keep = ["date","open","high","low","close","adj_close","volume"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    return df.reset_index(drop=True)

def merge_old_new(old_df: pd.DataFrame, new_df: pd.DataFrame):
    """
    Merge two OHLCV frames on date, preferring 'new_df' when overlap exists.
    """
    if old_df is None or len(old_df)==0: 
        return new_df.copy()
    if new_df is None or len(new_df)==0:
        return old_df.copy()
    cols = ["date","open","high","low","close","adj_close","volume"]
    for df in (old_df, new_df):
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
    both = pd.concat([old_df[cols], new_df[cols]], axis=0, ignore_index=True)
    both = both.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return both.reset_index(drop=True)
def fetch_tiingo_history(symbol: str, start: str|None=None, end: str|None=None):
    """Shim: use tiingo.fetch_daily across the requested window."""
    import os
    if not os.getenv("TIINGO_API_KEY"):
        raise RuntimeError("TIINGO_API_KEY not set")
    from .tiingo import fetch_daily
    return fetch_daily(symbol, start=start, end=end)

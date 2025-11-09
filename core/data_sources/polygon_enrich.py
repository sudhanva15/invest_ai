from __future__ import annotations
import os, json, datetime as dt, time
from typing import Dict
import pandas as pd
import urllib.request

# Simple token-bucket-ish limiter (very conservative)
_LAST_CALL = 0.0
def _throttle(min_interval=1.0):
    global _LAST_CALL
    now = time.time()
    wait = min_interval - (now - _LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL = time.time()

def _api_key() -> str:
    k = os.environ.get("POLYGON_API_KEY", "")
    if not k:
        raise RuntimeError("POLYGON_API_KEY missing")
    return k

def _get(url: str) -> dict:
    _throttle(1.2)  # ~ <1 req/s to be safe on free tier
    with urllib.request.urlopen(url) as r:
        return json.load(r)

def polygon_price_daily(symbol: str, start="2000-01-01", end=None) -> pd.DataFrame:
    key = _api_key()
    end = (end or dt.date.today().isoformat())
    url = (f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/"
           f"{start}/{end}?adjusted=true&sort=asc&limit=50000&apiKey={key}")
    data = _get(url)
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
    df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["adj_close"] = df["close"]  # adjusted close not separate in this endpoint
    return df[["date","open","high","low","close","adj_close","volume"]]

def polygon_dividends(symbol: str) -> pd.DataFrame:
    key = _api_key()
    url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol}&limit=1000&apiKey={key}"
    data = _get(url)
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["ex_dividend_date","cash_amount","currency"])
    df = pd.DataFrame(rows)
    want = ["cash_amount","currency","ex_dividend_date","pay_date","declaration_date","record_date","ticker"]
    for c in want:
        if c not in df.columns: df[c] = None
    return df[want]

def polygon_splits(symbol: str) -> pd.DataFrame:
    key = _api_key()
    url = f"https://api.polygon.io/v3/reference/splits?ticker={symbol}&limit=1000&apiKey={key}"
    data = _get(url)
    rows = data.get("results", [])
    if not rows:
        return pd.DataFrame(columns=["execution_date","split_from","split_to"])
    df = pd.DataFrame(rows)
    want = ["execution_date","split_from","split_to","ticker"]
    for c in want:
        if c not in df.columns: df[c] = None
    return df[want]

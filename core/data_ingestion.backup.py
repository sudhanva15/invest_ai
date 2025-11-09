from __future__ import annotations
import os, time
from pathlib import Path
from typing import List
import pandas as pd
import requests
import yfinance as yf
from .utils import log, load_config, load_env

CFG = load_config()
ENV = load_env()
CACHE = Path(CFG["data"]["cache_dir"])

def _cache_path(ticker: str) -> Path:
    return CACHE / f"{ticker}.csv"

def fetch_prices_yf(ticker: str, period="max", interval="1d") -> pd.DataFrame:
    log.info(f"[yfinance] {ticker} {period} {interval}")
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    if not df.empty:
        df = df.reset_index().rename(columns={"Date":"date"})
    return df

def get_prices(tickers: List[str], force=False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        cp = _cache_path(t)
        if cp.exists() and not force:
            df = pd.read_csv(cp, parse_dates=["date"])
            log.info(f"[cache] {t}: {len(df)} rows")
        else:
            df = fetch_prices_yf(t)
            df.to_csv(cp, index=False)
            log.info(f"[download] {t}: {len(df)} rows -> {cp}")
            time.sleep(0.2)
        df["ticker"] = t
        frames.append(df[["date","ticker","Close"]].rename(columns={"Close":"price"}))
    return pd.concat(frames, ignore_index=True)

def get_fred_series(series_id: str) -> pd.DataFrame:
    key = os.getenv("FRED_API_KEY","")
    url = f'{CFG["apis"]["fred_base"]}/series/observations'
    params = {"series_id": series_id, "api_key": key, "file_type":"json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["observations"])[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series"] = series_id
    return df.dropna()

from __future__ import annotations
import os, time, random
from pathlib import Path
from typing import List
import pandas as pd
import requests
import yfinance as yf
from .utils import log, load_config, load_env

CFG = load_config()
ENV = load_env()
CACHE = Path(CFG["data"]["cache_dir"])

FALLBACK_PERIODS = ["max","20y","10y","5y","1y"]
SLEEP_RANGE = (0.15, 0.35)

def _cache_path(ticker: str) -> Path:
    return CACHE / f"{ticker}.csv"

def _normalize_df(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns:
        df = df.rename_axis("date").reset_index()

    price_col = None
    for c in ("Close","Adj Close","close","adj_close"):
        if c in df.columns:
            price_col = c
            break
    if price_col is None:
        return pd.DataFrame()

    df = df[["date", price_col]].rename(columns={price_col: "price"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date","price"])
    if df.empty:
        return pd.DataFrame()
    df["ticker"] = ticker
    return df[["date","ticker","price"]].sort_values("date")

def fetch_prices_yf(ticker: str, periods=FALLBACK_PERIODS, interval="1d", attempts=3) -> pd.DataFrame:
    """
    Try multiple periods & attempts. Return empty df if nothing works.
    """
    for attempt in range(1, attempts+1):
        for per in periods:
            try:
                log.info(f"[yfinance] {ticker} period={per} attempt={attempt}")
                # try both API styles
                df = yf.download(ticker, period=per, interval=interval, auto_adjust=True, progress=False)
                if df is None or df.empty:
                    # alt path
                    hist = yf.Ticker(ticker).history(period=per, auto_adjust=True)
                    df = hist if not hist.empty else df
                df = _normalize_df(ticker, df)
                if not df.empty:
                    return df
            except Exception as e:
                log.warning(f"[yfinance] {ticker} error on period {per}: {e}")
                time.sleep(0.2)
        # jitter between attempts
        time.sleep(random.uniform(*SLEEP_RANGE))
    log.warning(f"[yfinance] {ticker}: no usable data after retries.")
    return pd.DataFrame()

def get_prices(tickers: List[str], force: bool=False) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        cp = _cache_path(t)
        df = pd.DataFrame()

        if cp.exists() and not force:
            try:
                cached = pd.read_csv(cp, parse_dates=["date"])
                if set(["date","price"]).issubset(cached.columns):
                    df = cached.assign(ticker=t)[["date","ticker","price"]]
                    log.info(f"[cache] {t}: {len(df)} rows")
                else:
                    log.warning(f"[cache] {t}: missing columns; refetching")
            except Exception as e:
                log.warning(f"[cache] {t}: read failed ({e}); refetching")

        if df.empty:
            fresh = fetch_prices_yf(t)
            if fresh.empty:
                log.warning(f"[skip] {t}: no usable data; skipping")
            else:
                try:
                    fresh.to_csv(cp, index=False)
                    log.info(f"[download] {t}: {len(fresh)} rows -> {cp}")
                except Exception as e:
                    log.warning(f"[cache-write] {t}: failed to write cache ({e})")
                df = fresh

        if not df.empty:
            frames.append(df)
        time.sleep(random.uniform(*SLEEP_RANGE))

    if not frames:
        # return empty df; let caller handle gracefully
        log.error("No usable price data downloaded for any ticker.")
        return pd.DataFrame(columns=["date","ticker","price"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker","date"]).dropna(subset=["date","price"])
    return out

def get_fred_series(series_id: str) -> pd.DataFrame:
    key = os.getenv("FRED_API_KEY", "")
    url = f'{CFG["apis"]["fred_base"]}/series/observations'
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js.get("observations", []))[["date","value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["series"] = series_id
    return df.dropna(subset=["date","value"])

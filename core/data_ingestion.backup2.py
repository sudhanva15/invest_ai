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
    """
    Download with yfinance and normalize columns.
    Returns empty DataFrame if nothing usable is returned.
    """
    try:
        log.info(f"[yfinance] {ticker} {period} {interval}")
        df = yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=True, progress=False
        )
        if df is None or df.empty:
            log.warning(f"[yfinance] {ticker}: empty response")
            return pd.DataFrame()
        # Ensure date is a column named 'date'
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        elif "date" not in df.columns:
            # last resort: try to coerce an index to date
            log.warning(f"[yfinance] {ticker}: no 'Date' column; attempting to coerce index")
            df = df.rename_axis("date").reset_index()
        # Ensure a price column named 'price'
        price_col = None
        for c in ("Close", "Adj Close", "close", "adj_close"):
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            log.warning(f"[yfinance] {ticker}: no close price column found")
            return pd.DataFrame()
        df = df[["date", price_col]].rename(columns={price_col: "price"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "price"])
        return df
    except Exception as e:
        log.exception(f"[yfinance] {ticker}: exception {e}")
        return pd.DataFrame()

def get_prices(tickers: List[str], force: bool = False) -> pd.DataFrame:
    """
    Get prices for a list of tickers with caching and safety checks.
    Skips tickers that fail; raises if all fail.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        cp = _cache_path(t)
        df = pd.DataFrame()

        if cp.exists() and not force:
            try:
                cached = pd.read_csv(cp, parse_dates=["date"])
                # sanity check columns
                if set(["date", "price"]).issubset(cached.columns):
                    df = cached[["date", "price"]].copy()
                    log.info(f"[cache] {t}: {len(df)} rows")
                else:
                    log.warning(f"[cache] {t}: missing columns; refetching")
            except Exception as e:
                log.warning(f"[cache] {t}: failed to read ({e}); refetching")

        if df.empty:
            fresh = fetch_prices_yf(t)
            if fresh.empty:
                log.warning(f"[skip] {t}: no usable data; skipping")
                continue
            try:
                fresh.to_csv(cp, index=False)
                log.info(f"[download] {t}: {len(fresh)} rows -> {cp}")
            except Exception as e:
                log.warning(f"[cache-write] {t}: failed to write cache ({e})")
            df = fresh

        df = df.assign(ticker=t)
        frames.append(df[["date", "ticker", "price"]])
        time.sleep(0.15)  # be polite

    if not frames:
        raise ValueError("No usable price data downloaded. Check tickers or network.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["ticker", "date"]).dropna(subset=["date", "price"])
    return out

def get_fred_series(series_id: str) -> pd.DataFrame:
    key = os.getenv("FRED_API_KEY", "")
    url = f'{CFG["apis"]["fred_base"]}/series/observations'
    params = {"series_id": series_id, "api_key": key, "file_type": "json"}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js.get("observations", []))[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["series"] = series_id
    return df.dropna(subset=["date", "value"])

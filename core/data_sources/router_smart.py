from __future__ import annotations
import os
import time
import pandas as pd
from typing import Callable, Optional

from core.data_sources.yf_source import fetch_yfinance_history
from core.utils.env_tools import is_demo_mode

# Stooq symbol mapping (US suffixed tickers)
STOOQ_MAP = {"SPY":"SPY.US","QQQ":"QQQ.US","TLT":"TLT.US","IEF":"IEF.US","GLD":"GLD.US"}

INVEST_AI_DEBUG = os.environ.get("INVEST_AI_DEBUG") == "1"
DEMO_MODE = is_demo_mode()

def _dbg(*a):
    if INVEST_AI_DEBUG:
        print(*a)

# ---------- Safe DataFrame helpers ----------
def _nonempty(df: pd.DataFrame | None) -> bool:
    return df is not None and len(df) > 0

def _clean_fallback(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if not _nonempty(df):
        return df
    df = df.set_axis([str(c).strip().lower() for c in df.columns], axis=1, copy=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    for c in ["open", "high", "low", "close", "adj_close", "volume", "price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "price" in df.columns and ("close" not in df.columns or df["close"].isna().all()):
        df["close"] = df["price"]
    if "adj_close" in df.columns:
        df["adj_close"] = df["adj_close"].fillna(df.get("close"))
    elif "close" in df.columns:
        df["adj_close"] = df["close"]
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close", "price"] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how="all")
    if "date" in df.columns and len(df) > 0:
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)

def ensure_ticker(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
    """Guarantee a scalar ticker column (fill if missing/NaN)."""
    if not _nonempty(df):
        return df
    if "ticker" not in df.columns:
        df["ticker"] = symbol
    else:
        df["ticker"] = df["ticker"].fillna(symbol)
    return df

def _align_cols(left: pd.DataFrame, right: pd.DataFrame):
    cols = sorted(set(left.columns) | set(right.columns))
    return left.reindex(columns=cols), right.reindex(columns=cols), cols

def _prefer(left: pd.DataFrame | None, right: pd.DataFrame | None, key: str = "date") -> pd.DataFrame | None:
    """Merge two OHLCV frames by key, preferring 'left' values when both present."""
    if not _nonempty(left) and not _nonempty(right):
        return None
    if not _nonempty(right):
        return left
    if not _nonempty(left):
        return right
    L = _clean_fallback(left.copy())
    R = _clean_fallback(right.copy())
    if not _nonempty(L):
        return R
    if not _nonempty(R):
        return L
    L, R, _ = _align_cols(L, R)
    out = pd.concat([L, R], ignore_index=True)
    if key in out.columns:
        out = out.sort_values(key).drop_duplicates(subset=[key], keep="first").reset_index(drop=True)
    return out

# ---------- Providers ----------
def _stooq_fetch(symbol, start=None, end=None, force=False):
    _dbg(f"[router] stooq({symbol}, start={start}, end={end})")
    try:
        from core.data_sources import stooq
        # Prefer raw symbol for local cache; fallback to mapped if empty
        try:
            df = stooq.fetch_daily(symbol, start=start, end=end, force=force)
        except TypeError:
            df = stooq.fetch_daily(symbol)
        if (df is None or len(df)==0):
            mapped = STOOQ_MAP.get(str(symbol).upper())
            if mapped:
                try:
                    df = stooq.fetch_daily(mapped, start=start, end=end, force=force)
                except TypeError:
                    df = stooq.fetch_daily(mapped)
        return _clean_fallback(df)
    except Exception as e:
        _dbg("[router] STQ error:", type(e).__name__, e)
        return None

def _fetch_tiingo_history(symbol, start=None, end=None):
    _dbg(f"[router] tiingo({symbol}, start={start}, end={end})")
    if DEMO_MODE:
        return None
    try:
        from core.data_sources.backfill_tiingo import fetch_tiingo_history
        df = fetch_tiingo_history(symbol, start=start, end=end)
        return _clean_fallback(df)
    except Exception as e:
        _dbg("[router] TIINGO error:", type(e).__name__, e)
        return None

# ---------- Union fetcher (Stooq primary, Tiingo backfill) ----------
from core.utils.env_tools import load_config
from pathlib import Path

def _yfinance_fetch(symbol, start=None, end=None):
    if DEMO_MODE:
        return None
    df = fetch_yfinance_history(symbol, start_date=start, end_date=end)
    if df is None or df.empty:
        return None
    return df

def fetch_union(
    symbol: str,
    *,
    attempts: list | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False,
    return_provenance: bool = False,
):
    # Decide whether to include yfinance as a last-resort backfill based on config
    try:
        cfg = load_config(Path("config/config.yaml"))
        use_yf = bool(cfg.get("apis", {}).get("use_yfinance_fallback", cfg.get("data", {}).get("use_yfinance_fallback", False)))
    except Exception:
        use_yf = False
    if DEMO_MODE:
        use_yf = False
    providers: list[tuple[str, Callable]] = [
        ("stooq",  lambda: _stooq_fetch(symbol, start=start, end=end, force=force)),
    ]
    if not DEMO_MODE:
        # Conditionally include Tiingo (skip if rate-limit previously hit and skip flag enabled)
        try:
            from core.data_sources.tiingo import tiingo_rate_limited
            if not (tiingo_rate_limited() and os.getenv("TIINGO_SKIP_ON_RATE_LIMIT", "1") == "1"):
                providers.append(("tiingo", lambda: _fetch_tiingo_history(symbol, start=start, end=end)))
        except Exception:
            providers.append(("tiingo", lambda: _fetch_tiingo_history(symbol, start=start, end=end)))
    if use_yf:
        providers.append(("yfinance", lambda: _yfinance_fetch(symbol, start=start, end=end)))
    merged: pd.DataFrame | None = None
    provenance: dict[str, int] = {}

    for name, call in providers:
        t0 = time.time()
        err_name, rows, ok = None, 0, False
        try:
            df = call()
            df = ensure_ticker(df, symbol)
            ok = _nonempty(df)
            rows = len(df) if ok else 0
        except Exception as e:
            df = None
            err_name = type(e).__name__
        dt_ms = int((time.time() - t0) * 1000)

        if attempts is not None:
            attempts.append({"provider": name, "ms": dt_ms, "ok": bool(ok), "rows": rows, "error": err_name})

        if ok:
            provenance[name] = rows
            merged = df if merged is None else _prefer(merged, df, key="date")
        else:
            _dbg("[router] union: provider failed:", name, err_name or "empty", f"{dt_ms}ms")

    if not _nonempty(merged):
        return (None, provenance) if return_provenance else None

    # Align by common start across available price columns, then drop very sparse columns (>20% NaN)
    aligned = merged.copy().sort_values("date")
    price_cols = [c for c in ["open","high","low","close","adj_close","volume","price"] if c in aligned.columns]
    first_valid_dates = {}
    if price_cols:
        for c in price_cols:
            # first date where column has a non-NaN value
            non_null_rows = aligned.loc[aligned[c].notna(), ["date"]]
            if not non_null_rows.empty:
                first_valid_dates[c] = non_null_rows.iloc[0]["date"]
        if first_valid_dates:
            common_start = max(first_valid_dates.values())
            aligned = aligned[aligned["date"] >= common_start].reset_index(drop=True)
    # Drop columns with excessive NaNs (>20%) post-alignment
    to_drop = [c for c in price_cols if aligned[c].isna().mean() > 0.20]
    if to_drop:
        aligned = aligned.drop(columns=to_drop)

    merged = aligned

    # Attach metadata to DataFrame (after alignment)
    df = merged.copy()
    if df is not None and not df.empty:
        provider_map = {symbol: "+".join(sorted(provenance.keys()))}
        total_rows = sum(provenance.values())
        backfill_pct = (provenance.get("tiingo", 0) / total_rows * 100) if total_rows > 0 else 0
        
        # Store metadata in DataFrame attributes (will be preserved through merges)
        df.attrs["provider_map"] = provider_map
        df.attrs["backfill_pct"] = {symbol: backfill_pct}
        
        # Capture date coverage
        df = df.sort_values("date")
        df.attrs["coverage"] = {symbol: (df["date"].min().strftime("%Y-%m-%d"), 
                                      df["date"].max().strftime("%Y-%m-%d"))}

    return (df, provenance) if return_provenance else df

# ---------- Public entry ----------
def router_fetch_daily_smart(
    symbol: str,
    asset_class: str | None = None,
    start: str | None = None,
    end: str | None = None,
    force: bool = False
):
    """
    Canonical daily fetch for prices:
    - Builds a union of Stooq (primary) with Tiingo backfill for gaps/depth.
    - 'start' defaults to 'earliest' (1900-01-01) for deep backfills.
    """
    if start in (None, ""):
        start = "earliest"
    if start == "earliest":
        start = "1900-01-01"

    attempts: list[dict] = []
    df, prov = fetch_union(symbol, attempts=attempts, start=start, end=end, force=force, return_provenance=True)
    # Optional: print attempts in debug
    if INVEST_AI_DEBUG:
        _dbg(f"[router] {symbol} attempts:", attempts)
        _dbg(f"[router] {symbol} provenance:", prov)

    source_tag = " + ".join([k for k in ["tiingo", "stooq"] if k in prov]) or "none"
    return df, source_tag
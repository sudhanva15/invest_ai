"""
Centralized price history fetcher with multi-provider fallback and caching.

This module provides a single entry point for all price data fetches:
    fetch_price_history(ticker, start, end)

Provider precedence:
    1. Tiingo (primary, if key available)
    2. Stooq (secondary)
    3. yfinance (fallback, config-controlled)
    4. Cache (as last resort)

Cache behavior:
    - Format: parquet (faster I/O, smaller files)
    - Location: data/cache/{ticker}.parquet
    - Metadata: {ticker}_meta.json with fetched_at, provider, start, end
    - TTL: configurable (default 1 day for production, longer for dev)
    - Auto-write on successful fetch from any provider
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from core.env_tools import is_demo_mode, load_config
from core.demo_data import load_demo_price_history

# Configure logging
logger = logging.getLogger(__name__)

# Cache configuration
DEMO_MODE = is_demo_mode()
CACHE_DIR = Path("data/cache") if not DEMO_MODE else None
if CACHE_DIR is not None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_DAYS = int(os.getenv("CACHE_TTL_DAYS", "1"))  # 1 day default

# Symbol mapping for provider-specific quirks
# NOTE: Stooq's fetch_daily() internally uses uppercase for cache lookup in data/raw/{SYMBOL}.csv
# So we pass uppercase ticker to Stooq, not .us suffix
SYMBOL_MAP = {
    "stooq": lambda sym: sym.upper(),
    "tiingo": lambda sym: sym.lower(),
    "yfinance": lambda sym: sym.upper(),
}


class ProviderResult:
    """Container for provider fetch results with metadata."""
    
    def __init__(self, data: Optional[pd.DataFrame], provider: str, error: Optional[str] = None, source: str = "api"):
        self.data = data
        self.provider = provider
        self.error = error
        self.source = source  # e.g., api, cache
        self.success = data is not None and not data.empty


def _get_cache_paths(ticker: str) -> Tuple[Path, Path]:
    """Return (data_path, metadata_path) for ticker cache."""
    if CACHE_DIR is None:
        raise FileNotFoundError("Cache disabled in demo mode")
    data_path = CACHE_DIR / f"{ticker.upper()}.parquet"
    meta_path = CACHE_DIR / f"{ticker.upper()}_meta.json"
    return data_path, meta_path


def _is_cache_fresh(meta_path: Path, required_start: Optional[date], required_end: Optional[date]) -> bool:
    """
    Check if cached data is fresh enough and covers the required date range.
    
    Returns True if:
        - Metadata file exists and is valid JSON
        - fetched_at is within CACHE_TTL_DAYS
        - Cached start <= required_start (or required_start is None)
        - Cached end >= required_end (or required_end is None)
    """
    if not meta_path.exists():
        return False
    
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        
        # Check TTL
        fetched_at = datetime.fromisoformat(meta["fetched_at"])
        age_days = (datetime.now() - fetched_at).days
        if age_days > CACHE_TTL_DAYS:
            logger.debug(f"Cache for {meta_path.stem} is stale ({age_days} days old)")
            return False
        
        # Check date coverage
        cached_start = pd.to_datetime(meta.get("start")).date() if meta.get("start") else None
        cached_end = pd.to_datetime(meta.get("end")).date() if meta.get("end") else None
        
        if required_start and cached_start and cached_start > required_start:
            logger.debug(f"Cache for {meta_path.stem} doesn't cover required start: {cached_start} > {required_start}")
            return False
        
        if required_end and cached_end and cached_end < required_end:
            logger.debug(f"Cache for {meta_path.stem} doesn't cover required end: {cached_end} < {required_end}")
            return False
        
        return True
    
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.warning(f"Invalid cache metadata for {meta_path.stem}: {e}")
        return False


def _read_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Read cached data for ticker if available."""
    data_path, _ = _get_cache_paths(ticker)
    
    if not data_path.exists():
        return None
    
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"✓ Cache hit for {ticker}: {len(df)} rows")
        return df
    except Exception as e:
        logger.warning(f"Failed to read cache for {ticker}: {e}")
        return None


def _write_cache(ticker: str, df: pd.DataFrame, provider: str, source: str = "api"):
    """
    Write DataFrame to cache with metadata.
    
    NEVER cache empty or suspiciously small frames (< 100 rows).
    These indicate incomplete/bad data that should not persist.
    """
    data_path, meta_path = _get_cache_paths(ticker)
    
    # Validation: never cache bad data
    if df is None or df.empty:
        logger.debug(f"Skip cache: {ticker} → empty DataFrame")
        return
    
    if len(df) < 100:
        logger.warning(f"Skip cache: {ticker} → too small ({len(df)} rows < 100)")
        return
    
    try:
        # Write data
        df.to_parquet(data_path, index=True)
        
        # Write metadata
        meta = {
            "ticker": ticker.upper(),
            "provider": provider,
            "fetched_at": datetime.now().isoformat(),
            "start": df.index.min().isoformat() if not df.empty else None,
            "end": df.index.max().isoformat() if not df.empty else None,
            "rows": len(df),
            "source": f"{provider}.{source}",
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"✓ Cached {ticker}: {len(df)} rows from {provider} (source={source})")
        
    except Exception as e:
        logger.warning(f"Failed to write cache for {ticker}: {e}")


def _fetch_stooq(ticker: str, start: Optional[date], end: Optional[date]) -> ProviderResult:
    """Fetch from Stooq with proper error handling."""
    try:
        from core.data_sources import stooq
        
        symbol = SYMBOL_MAP["stooq"](ticker)
        df = stooq.fetch_daily(symbol, start=start.isoformat() if start else None, end=end.isoformat() if end else None)
        
        if df is None or df.empty:
            return ProviderResult(None, "stooq", "empty_response")
        logger.info(f"✓ Stooq: {ticker} → {len(df)} rows (source=cache)")
        return ProviderResult(df, "stooq", None, source="cache")
    
    except Exception as e:
        logger.debug(f"✗ Stooq: {ticker} → {type(e).__name__}: {e}")
        return ProviderResult(None, "stooq", f"{type(e).__name__}: {str(e)[:50]}")


def _fetch_tiingo(ticker: str, start: Optional[date], end: Optional[date]) -> ProviderResult:
    """
    Fetch from Tiingo with proper error handling.
    
    NOTE: For full history, pass start=None to let Tiingo return all available data.
    Forcing a start date (e.g., 2010-01-01) artificially limits the response.
    """
    # Check for API key via layered loader
    try:
        from core.data_sources.tiingo import fetch_daily, is_tiingo_enabled
        if not is_tiingo_enabled(ping=False):
            logger.debug(f"Tiingo disabled for {ticker}: missing API key")
            return ProviderResult(None, "tiingo", "missing_api_key")
    except Exception:
        return ProviderResult(None, "tiingo", "missing_api_key")

    try:
        symbol = SYMBOL_MAP["tiingo"](ticker)
        # Pass None for start/end to get full history when caller doesn't specify
        df = fetch_daily(symbol, start=start.isoformat() if start else None, end=end.isoformat() if end else None)
        
        if df is None or df.empty:
            logger.warning(f"✗ Tiingo: {ticker} → empty_response")
            return ProviderResult(None, "tiingo", "empty_response")
        
        # Additional validation: reject too-small responses (< 100 rows)
        if len(df) < 100:
            logger.warning(f"✗ Tiingo: {ticker} → too_small ({len(df)} rows < 100)")
            return ProviderResult(None, "tiingo", f"too_small ({len(df)} rows)")
        
        logger.info(f"✓ Tiingo: {ticker} → {len(df)} rows (source=api)")
        return ProviderResult(df, "tiingo", None, source="api")
    except Exception as e:
        logger.error(f"✗ Tiingo: {ticker} → {type(e).__name__}: {e}")
        return ProviderResult(None, "tiingo", f"{type(e).__name__}: {str(e)[:50]}")


def _fetch_yfinance(ticker: str, start: Optional[date], end: Optional[date]) -> ProviderResult:
    """Fetch from yfinance with proper error handling."""
    try:
        import yfinance as yf
        symbol = SYMBOL_MAP["yfinance"](ticker)
        # Safe mode
        hist_kwargs = {"auto_adjust": True}
        if start:
            hist_kwargs["start"] = start.isoformat()
        if end:
            hist_kwargs["end"] = end.isoformat()
        try:
            t = yf.Ticker(symbol)
            df = t.history(**hist_kwargs)
        except Exception:
            df = yf.download(symbol, progress=False, auto_adjust=True, threads=False, start=hist_kwargs.get("start"), end=hist_kwargs.get("end"))
        if df is None or df.empty:
            return ProviderResult(None, "yfinance", "empty_response")
        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        logger.info(f"✓ yfinance: {ticker} → {len(df)} rows (source=api)")
        return ProviderResult(df, "yfinance", None, source="api")
    except Exception as e:
        logger.debug(f"✗ yfinance: {ticker} → {type(e).__name__}: {e}")
        return ProviderResult(None, "yfinance", f"{type(e).__name__}: {str(e)[:50]}")


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize DataFrame to standard schema:
        - DatetimeIndex named 'date'
        - Columns: date, open, high, low, close, adj_close, volume
        - adj_close filled from close if missing
        - Sorted by date, deduped
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    
    df = df.copy()
    
    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df[df["date"].notna()]
            df = df.set_index("date")
    
    df.index.name = "date"
    
    # Lowercase columns
    df.columns = [str(c).lower() for c in df.columns]
    
    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col not in df.columns:
            df[col] = pd.NA
    
    # Fill adj_close from close
    if "adj_close" in df.columns and "close" in df.columns:
        df["adj_close"] = df["adj_close"].fillna(df["close"])
    elif "close" in df.columns:
        df["adj_close"] = df["close"]
    
    # Coerce numeric
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with all-NaN prices
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how="all")
    
    # Sort and dedupe
    df = df.sort_index().loc[~df.index.duplicated(keep="last")]
    
    return df

def _is_dataframe_valid(df: pd.DataFrame, min_non_missing_ratio: float = 0.05) -> bool:
    """
    Validate DataFrame shape and continuity.
    - Required columns exist
    - Index is increasing
    - Not >95% missing by expected business days
    """
    if df is None or df.empty:
        return False
    required = {"open", "high", "low", "close", "adj_close", "volume"}
    if not required.issubset(set(df.columns)):
        return False
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing:
        return False
    try:
        st, en = df.index.min(), df.index.max()
        expected = len(pd.bdate_range(st, en))
        got = df.shape[0]
        if expected > 0 and (got / expected) < min_non_missing_ratio:
            return False
    except Exception:
        pass
    return True


def fetch_price_history(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    use_cache: bool = True,
    write_cache: bool = True,
    allow_live: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Fetch price history for a ticker with cache-first, live-best-effort behavior.
    
    Args:
        ticker: Ticker symbol (e.g., "SPY", "QQQ")
        start: Start date (inclusive). None = earliest available.
        end: End date (inclusive). None = most recent available.
        use_cache: Whether to read from cache if available
        write_cache: Whether to write successful fetch to cache
        allow_live: Whether to allow live API calls. If None, reads from config.
    
    Returns:
        DataFrame with DatetimeIndex and columns:
            open, high, low, close, adj_close, volume
        
        Returns empty DataFrame if all providers fail.
    
    Cache-first behavior:
        1. Try cache first if fresh and covers requested date range
        2. If cache miss/stale AND allow_live=True:
           - Try providers: Tiingo → Stooq → yfinance
        3. If all live providers fail, return stale cache if exists (degraded mode)
        4. If no usable data exists, return empty DataFrame
    
    Raises:
        ValueError: If ticker is invalid
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"Invalid ticker: {ticker}")
    
    ticker = ticker.strip().upper()
    
    if DEMO_MODE:
        demo_df = load_demo_price_history(ticker, start.isoformat() if start else None, end.isoformat() if end else None)
        demo_df = demo_df.copy()
        demo_df["date"] = pd.to_datetime(demo_df["date"])
        demo_df = demo_df.set_index("date").sort_index()
        cols = ["open","high","low","close","adj_close","volume"]
        for c in cols:
            if c not in demo_df.columns:
                demo_df[c] = pd.NA
        return demo_df[cols]

    # Load config for allow_live and min_cache_rows
    try:
        cfg = load_config(Path("config/config.yaml"))
        if allow_live is None:
            allow_live = bool(cfg.get("fetch", {}).get("allow_live_fetch", True))
        min_cache_rows = int(cfg.get("fetch", {}).get("min_cache_rows", 100))
    except Exception:
        if allow_live is None:
            allow_live = True
        min_cache_rows = 100
    
    # STEP 1: Try cache first (cache-first)
    if use_cache:
        if CACHE_DIR is not None:
            data_path, meta_path = _get_cache_paths(ticker)
            if _is_cache_fresh(meta_path, start, end):
                df = _read_cache(ticker)
                if df is not None and not df.empty:
                    df = _normalize_dataframe(df)
                    if _is_dataframe_valid(df):
                        # Filter to requested date range
                        if start:
                            df = df[df.index >= pd.Timestamp(start)]
                        if end:
                            df = df[df.index <= pd.Timestamp(end)]
                        logger.debug(f"✓ Cache hit (fresh): {ticker} → {len(df)} rows")
                        return df
    
    # STEP 2: Cache miss/stale → try live providers if allowed
    if not allow_live:
        logger.info(f"✗ {ticker}: cache miss and live fetch disabled")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])
    
    # Try providers in order
    providers = [
        ("tiingo", _fetch_tiingo),
        ("stooq", _fetch_stooq),
    ]
    
    # Add yfinance if enabled in config
    try:
        use_yf = bool(cfg.get("apis", {}).get("use_yfinance_fallback", cfg.get("data", {}).get("use_yfinance_fallback", False)))
        if use_yf:
            providers.append(("yfinance", _fetch_yfinance))
    except Exception:
        pass  # Config not available, skip yfinance
    
    errors = []
    for provider_name, fetch_fn in providers:
        result = fetch_fn(ticker, start, end)
        
        if result.success:
            df = _normalize_dataframe(result.data)
            # Sanity: require at least min_cache_rows to avoid caching stubs
            if df is not None and len(df) < min_cache_rows:
                logger.warning(f"provider={provider_name} symbol={ticker} status=too_small rows={len(df)}")
                errors.append(f"{provider_name}: too_few_rows={len(df)}")
                continue
            # Validate before caching/return
            if _is_dataframe_valid(df):
                # Filter to requested date range
                if start:
                    df = df[df.index >= pd.Timestamp(start)]
                if end:
                    df = df[df.index <= pd.Timestamp(end)]
                if write_cache and not df.empty:
                    _write_cache(ticker, df, result.provider, source=result.source)
                return df
            else:
                logger.warning(f"provider={provider_name} symbol={ticker} status=invalid_frame rows={len(df) if df is not None else 0}")
                errors.append(f"{provider_name}: invalid_frame")
                continue
        
        # Log structured error
        logger.warning(f"provider={provider_name} symbol={ticker} status=error reason={result.error}")
        errors.append(f"{provider_name}: {result.error}")
    
    # STEP 3: All live providers failed → try stale cache as last resort
    if use_cache:
        data_path, meta_path = _get_cache_paths(ticker)
        if CACHE_DIR is not None and data_path.exists():
            df = _read_cache(ticker)
            if df is not None and not df.empty:
                df = _normalize_dataframe(df)
                if _is_dataframe_valid(df):
                    if start:
                        df = df[df.index >= pd.Timestamp(start)]
                    if end:
                        df = df[df.index <= pd.Timestamp(end)]
                    logger.info(f"✓ Using stale cache (degraded mode): {ticker} → {len(df)} rows")
                    return df
    
    # STEP 4: No usable data exists
    logger.warning(f"✗ All providers failed for {ticker}: {'; '.join(errors)}")
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "adj_close", "volume"])


def fetch_multiple(
    tickers: list[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
    use_cache: bool = True,
    write_cache: bool = True,
    allow_live: Optional[bool] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Fetch price history for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date (inclusive). None = earliest available.
        end: End date (inclusive). None = most recent available.
        use_cache: Whether to read from cache if available
        write_cache: Whether to write successful fetch to cache
        allow_live: Whether to allow live API calls. If None, reads from config.
    
    Returns:
        (prices_df, metadata_dict)
        
        prices_df: Wide DataFrame with tickers as columns, DatetimeIndex
        metadata_dict: {ticker: {"provider": str, "rows": int, "error": str|None}}
    """
    frames = []
    metadata = {}
    
    for ticker in tickers:
        df = fetch_price_history(ticker, start=start, end=end, use_cache=use_cache, write_cache=write_cache, allow_live=allow_live)
        
        if not df.empty:
            # Extract adj_close as a series
            series = df["adj_close"].rename(ticker)
            frames.append(series)
            
            # Record metadata
            meta_path = _get_cache_paths(ticker)[1]
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    metadata[ticker] = {
                        "provider": meta.get("provider", "unknown"),
                        "rows": len(df),
                        "error": None,
                    }
            else:
                metadata[ticker] = {
                    "provider": "unknown",
                    "rows": len(df),
                    "error": None,
                }
        else:
            metadata[ticker] = {
                "provider": None,
                "rows": 0,
                "error": "all_providers_failed",
            }
    
    if frames:
        prices_df = pd.concat(frames, axis=1).sort_index()
        return prices_df, metadata
    else:
        return pd.DataFrame(), metadata


__all__ = ["fetch_price_history", "fetch_multiple", "CACHE_DIR", "CACHE_TTL_DAYS"]

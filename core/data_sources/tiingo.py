
from __future__ import annotations
import os
import io
import logging
import pandas as pd
import requests
from pathlib import Path

from core.utils.env_tools import load_env_once, load_config

logger = logging.getLogger(__name__)

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
TIINGO_PING = "https://api.tiingo.com/api/test"

def _get_token_layered() -> str | None:
    """
    Layered retrieval of TIINGO_API_KEY from:
    - os.environ (after loading .env)
    - config/config.yaml → apis.tiingo_api_key
    Returns None if not found.
    """
    try:
        load_env_once()
    except Exception:
        pass
    token = os.getenv("TIINGO_API_KEY")
    if token:
        return token
    try:
        cfg = load_config(Path("config/config.yaml"))
        token = (
            cfg.get("apis", {}).get("tiingo_api_key")
            or cfg.get("credentials", {}).get("tiingo_api_key")
        )
        return token
    except Exception:
        return None

def is_tiingo_enabled(ping: bool = True) -> bool:
    """
    Return True if a Tiingo API key is configured and (optionally) ping succeeds.
    """
    token = _get_token_layered()
    if not token:
        logger.debug("Tiingo disabled: missing TIINGO_API_KEY")
        return False
    if not ping:
        return True
    try:
        r = requests.get(TIINGO_PING, params={"token": token}, timeout=10)
        if r.status_code == 200:
            return True
        logger.debug(f"Tiingo ping failed: HTTP {r.status_code}")
        return False
    except Exception as e:
        logger.debug(f"Tiingo ping error: {type(e).__name__}: {e}")
        return False

def _normalize_tiingo_df(raw_csv_text: str) -> pd.DataFrame:
    """
    Normalize Tiingo CSV into our canonical schema:
    [date, open, high, low, close, adj_close, volume]
    
    Detects error messages (e.g., rate limit errors) that Tiingo returns as plain text.
    """
    # Check for error messages in response
    if "Error:" in raw_csv_text or "error" in raw_csv_text.lower()[:200]:
        logger.warning(f"Tiingo API error: {raw_csv_text[:200]}")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    
    try:
        df = pd.read_csv(io.StringIO(raw_csv_text))
    except Exception as e:
        logger.warning(f"Failed to parse Tiingo CSV: {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    
    if df.empty:
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    
    # Check for required columns
    if "date" not in df.columns:
        logger.warning(f"Tiingo response missing 'date' column. Columns: {list(df.columns)}")
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
    
    Note: If start=None, Tiingo returns full available history (often 10-20+ years).
    This is preferred over forcing a specific start date.
    """
    token = _get_token_layered()
    if not token:
        # Graceful disable when key missing
        logger.debug("Tiingo fetch skipped (no API key)")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])

    params = {"format": "csv", "token": token}
    # Only set date params if explicitly provided; let Tiingo return full history otherwise
    if start:
        params["startDate"] = start  # YYYY-MM-DD
    if end:
        params["endDate"] = end

    url = TIINGO_BASE.format(ticker=symbol.lower())
    
    # Structured logging for diagnostics
    try:
        logger.info(f"Tiingo request: {symbol} | start={start or 'None (full history)'} | end={end or 'latest'}")
        r = requests.get(url, params=params, timeout=30)
        
        # Log HTTP status
        status = r.status_code
        logger.info(f"Tiingo response: {symbol} | HTTP {status}")
        
        r.raise_for_status()
        df = _normalize_tiingo_df(r.text)
        
        # Detailed logging for result
        if df.empty:
            logger.warning(f"Tiingo EMPTY: {symbol} | HTTP {status} | rows=0")
            return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
        
        first_date = df["date"].min()
        last_date = df["date"].max()
        rows = len(df)
        
        logger.info(f"Tiingo SUCCESS: {symbol} | rows={rows} | {first_date} to {last_date}")
        
        # Validate minimum quality: reject suspiciously small responses
        if rows < 50:
            logger.warning(f"Tiingo TOO_SMALL: {symbol} | rows={rows} < 50 (likely incomplete data)")
            return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
        
        return df
        
    except requests.HTTPError as e:
        logger.error(f"Tiingo HTTP_ERROR: {symbol} | status={e.response.status_code if hasattr(e, 'response') else 'unknown'} | {str(e)[:100]}")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    except Exception as e:
        logger.error(f"Tiingo EXCEPTION: {symbol} | {type(e).__name__}: {str(e)[:100]}")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])

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


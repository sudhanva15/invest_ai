
from __future__ import annotations
import os
import io
import logging
import time
import pandas as pd
import requests
from pathlib import Path

from core.env_tools import load_env_once, load_config, is_demo_mode
from core.demo_data import load_demo_price_history

# --- Rate-limit state (module-level) -------------------------------------------------
# If Tiingo starts returning HTTP 429 or textual rate-limit errors, we set a flag
# so the rest of the ingest pipeline can fast-skip further Tiingo calls within
# the same process run (avoids hammering the API and wasting latency).
RATE_LIMIT_HIT: bool = False
RATE_LIMIT_TS: float | None = None

# Retry configuration (Phase 3 enhancement)
MAX_RETRIES: int = 3
RETRY_BASE_DELAY: float = 2.0  # seconds
RETRY_MAX_DELAY: float = 16.0  # seconds

def tiingo_rate_limited() -> bool:
    """Return True if we have observed a Tiingo rate-limit condition this run."""
    return RATE_LIMIT_HIT

def _mark_rate_limit():
    global RATE_LIMIT_HIT, RATE_LIMIT_TS
    RATE_LIMIT_HIT = True
    RATE_LIMIT_TS = time.time()
    logger.warning("Tiingo rate limit detected; subsequent Tiingo fetches will be skipped.")

def reset_tiingo_rate_limit():  # manual reset hook (rarely used; keep for tests)
    global RATE_LIMIT_HIT, RATE_LIMIT_TS
    RATE_LIMIT_HIT = False
    RATE_LIMIT_TS = None

logger = logging.getLogger(__name__)
DEMO_MODE = is_demo_mode()

TIINGO_BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
TIINGO_PING = "https://api.tiingo.com/api/test"

def _demo_frame(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    df = load_demo_price_history(symbol, start=start, end=end)
    keep = ["date","open","high","low","close","adj_close","volume"]
    for col in keep:
        if col not in df.columns:
            df[col] = pd.NA
    return df[keep].reset_index(drop=True)

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
    if DEMO_MODE:
        logger.debug("Tiingo fetch skipped for %s (demo mode)", symbol)
        return _demo_frame(symbol, start=start, end=end)

    # Fast skip if we've already hit rate limit and skip flag is enabled.
    skip_flag_raw = os.getenv("TIINGO_SKIP_ON_RATE_LIMIT", "1")
    skip_on_rate_limit = str(skip_flag_raw).strip().lower() in {"1", "true", "yes"}

    if tiingo_rate_limited() and skip_on_rate_limit:
        logger.debug("Tiingo fetch skipped (rate-limit previously detected)")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])

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
    
    # Structured logging for diagnostics + exponential backoff retry (Phase 3 enhancement)
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Tiingo request: {symbol} | attempt={attempt+1}/{MAX_RETRIES} | start={start or 'None (full history)'} | end={end or 'latest'}")
            r = requests.get(url, params=params, timeout=30)
            # Log HTTP status
            status = r.status_code
            logger.info(f"Tiingo response: {symbol} | HTTP {status} | attempt={attempt+1}")
            body_lower = r.text.lower() if isinstance(r.text, str) else ""
            rate_text_hit = "rate limit" in body_lower

            # Detect explicit HTTP rate limit before raising
            if status == 429 or rate_text_hit:
                _mark_rate_limit()
                logger.warning(
                    "Tiingo rate limit signal for %s (skip_on_rate_limit=%s, status=%s)",
                    symbol,
                    skip_on_rate_limit,
                    status,
                )
                if skip_on_rate_limit:
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
                if attempt < MAX_RETRIES - 1:
                    # Exponential backoff: 2s, 4s, 8s (capped at 16s)
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    logger.warning(
                        f"Tiingo rate limit: {symbol} | retry in {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    continue  # Retry
                else:
                    logger.error(f"Tiingo rate limit: {symbol} | max retries exhausted")
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])

            r.raise_for_status()
            df = _normalize_tiingo_df(r.text)
            
            # If we got here, request succeeded - break retry loop
            break
            
        except requests.HTTPError as e:
            status = e.response.status_code if hasattr(e, "response") and e.response is not None else "unknown"
            if status == 429:
                _mark_rate_limit()
                logger.warning(
                    "Tiingo HTTP 429 for %s (skip_on_rate_limit=%s)",
                    symbol,
                    skip_on_rate_limit,
                )
                if skip_on_rate_limit:
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    logger.warning(
                        f"Tiingo HTTP 429: {symbol} | retry in {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    continue  # Retry
                else:
                    logger.error(f"Tiingo HTTP 429: {symbol} | max retries exhausted")
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
            else:
                # Other HTTP errors - don't retry
                logger.error(f"Tiingo HTTP_ERROR: {symbol} | status={status} | {str(e)[:100]}")
                return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
        except Exception as e:
            # Detect indirect rate-limit patterns
            msg_low = str(e).lower()
            if any(k in msg_low for k in ["rate", "429", "too many", "exceed"]):
                _mark_rate_limit()
                if skip_on_rate_limit:
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    logger.warning(
                        f"Tiingo rate hint: {symbol} | retry in {delay:.1f}s (attempt {attempt+1}/{MAX_RETRIES})"
                    )
                    time.sleep(delay)
                    continue  # Retry
                else:
                    logger.error(f"Tiingo rate limit: {symbol} | max retries exhausted")
                    return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
            else:
                # Non-rate-limit exception - don't retry
                logger.error(f"Tiingo EXCEPTION: {symbol} | {type(e).__name__}: {str(e)[:100]}")
                return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    
    # Post-retry validation (if we got here, we have a response in 'df' and 'r')
    # Heuristic textual detection (Tiingo sometimes returns 200 with an error string)
    head_text = r.text.lower()[:300]
    if any(phrase in head_text for phrase in ["rate limit", "too many", "exceeded", "throttled"]):
        _mark_rate_limit()
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    
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

__all__ = [
    "fetch_daily",
    "load_daily",
    "fetch",
    "tiingo_rate_limited",
    "reset_tiingo_rate_limit",
]


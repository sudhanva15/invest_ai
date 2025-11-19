from __future__ import annotations
import pandas as pd
import logging
from core.utils.env_tools import is_demo_mode

def _normalize_stooq_symbol(sym: str) -> str:
    s = sym.strip().lower()
    # Most US ETFs/stocks work as 'spy.us', 'aapl.us', etc.
    if not s.endswith('.us') and s.isalpha():
        s = f"{s}.us"
    return s

from pathlib import Path
from datetime import date
import urllib.error
import sys

# Determine repo root and absolute cache path
if __name__ == "__main__":
    _ROOT = Path(__file__).resolve().parents[2]
else:
    _ROOT = Path(__file__).resolve().parents[2]

_CACHE_DIR = _ROOT / "data" / "raw"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)
DEMO_MODE = is_demo_mode()

def _stooq_url(symbol: str) -> str:
    s = symbol.strip().lower()
    # Try plain, then ".us" (Stooq uses many suffixes; this covers common US tickers/ETFs)
    return f"https://stooq.com/q/d/l/?s={_normalize_stooq_symbol(symbol)}&i=d"

def _stooq_url_alt(symbol: str) -> str:
    s = symbol.strip().lower()
    return f"https://stooq.com/q/d/l/?s={_normalize_stooq_symbol(symbol)}&i=d"

def load_daily(symbol: str, start: str = "2000-01-01") -> pd.DataFrame:
    """
    Download daily OHLCV from Stooq (free) and cache to data/raw/{SYMBOL}.csv.
    Returns a DataFrame with columns: date, Open, High, Low, Close, Volume
    (upstream case preserved; 'date' is normalized to datetime.date)
    """
    sym_u = symbol.upper()
    cache_path = _CACHE_DIR / f"{sym_u}.csv"

    # Try cache first
    df = None
    if cache_path.exists():
        try:
            df = pd.read_csv(cache_path)
        except Exception:
            df = None

    # Fetch if cache missing/invalid
    if (df is None or df.empty) and not DEMO_MODE:
        for url in (_stooq_url(symbol), _stooq_url_alt(symbol)):
            try:
                df = pd.read_csv(url)
                if df is not None and not df.empty:
                    try:
                        df.to_csv(cache_path, index=False)
                    except Exception:
                        pass
                    break
            except urllib.error.URLError:
                continue
            except Exception:
                continue

    # Normalize or return empty frame
    if df is None or df.empty:
        return pd.DataFrame(columns=["date","Open","High","Low","Close","Volume"])

    # Normalize columns to expected names
    cols = {c.lower(): c for c in df.columns}
    # Stooq headers are usually: Date,Open,High,Low,Close,Volume
    # Make sure we have them:
    if "date" not in cols:
        raise ValueError("Stooq CSV missing 'Date' column")
    # Ensure 'date' is datetime.date
    df = df.rename(columns={"Date":"date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Filter by start
    try:
        start_d = pd.to_datetime(start).date()
        df = df[df["date"] >= start_d]
    except Exception:
        pass

    # Keep consistent ordering; fill any missing expected columns
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:
            df[c] = pd.NA

    return df[["date","Open","High","Low","Close","Volume"]].reset_index(drop=True)


def _to_canonical_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return DataFrame with canonical columns:
    ['date','open','high','low','close','adj_close','volume']
    - date coerced to pandas datetime
    - numeric columns coerced to numeric
    - fills adj_close with close if missing
    - sorted by date, de-duplicated on date
    """
    import pandas as pd
    if df is None or getattr(df, "empty", True):
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume"])
    d = df.copy()
    # Normalize headers: support both Stooq default case and lower-case
    rename_map = {
        "Date":"date", "Open":"open", "High":"high", "Low":"low", "Close":"close", "Volume":"volume",
        "date":"date", "open":"open", "high":"high", "low":"low", "close":"close", "volume":"volume",
        "Adj Close":"adj_close", "adj_close":"adj_close"
    }
    d = d.rename(columns={k:v for k,v in rename_map.items() if k in d.columns})
    # Ensure required columns exist
    for c in ["open","high","low","close","volume","adj_close"]:
        if c not in d.columns:
            d[c] = pd.NA
    # Date coercion
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d[d["date"].notna()]
    # Numeric coercion
    for c in ["open","high","low","close","adj_close","volume"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # Fill adj_close with close when missing
    if "adj_close" in d.columns and "close" in d.columns:
        d["adj_close"] = d["adj_close"].fillna(d["close"])
    # Sort & dedupe by date
    if "date" in d.columns:
        d = d.sort_values("date").drop_duplicates(subset=["date"])
    return d[["date","open","high","low","close","adj_close","volume"]]

# --- cleaning helpers ---------------------------------------------------------
def _clean_ohlcv(df):
    import pandas as pd
    if df is None or df.empty:
        return df
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    # Coerce numeric columns
    for c in ['open','high','low','close','adj_close','volume','price']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # If only 'price' present (some stooq endpoints), map to close/adj_close
    if 'price' in df.columns:
        if 'close' not in df.columns:
            df['close'] = df['price']
        if 'adj_close' not in df.columns:
            df['adj_close'] = df['close']
    # Fill missing adj_close with close if needed
    if 'adj_close' in df.columns and 'close' in df.columns:
        df['adj_close'] = df['adj_close'].fillna(df['close'])
    # Drop rows with no price at all
    price_cols = [c for c in ['close','adj_close','price'] if c in df.columns]
    if price_cols:
        df = df.dropna(subset=price_cols, how='all')
    # Sort and dedupe
    if 'date' in df.columns:
        df = df.sort_values('date').drop_duplicates(subset=['date'])
    return df
# ------------------------------------------------------------------------------

# ---- compatibility wrapper for router_smart.py --------------------------------
# Reads from the local Stooq cache at data/raw/{SYMBOL}.csv and normalizes columns.
def fetch_daily(symbol: str, start: str|None=None, end: str|None=None, force: bool=False):
    import pandas as pd
    from pathlib import Path

    sym = (symbol or "").upper()
    p = Path("data/raw") / f"{sym}.csv"
    if not p.exists():
        # no local cache; return empty DataFrame
        logger.debug(f"Stooq: source=cache miss for {sym}")
        return pd.DataFrame(columns=["date","open","high","low","close","adj_close","volume","ticker"])

    df = pd.read_csv(p)
    logger.debug(f"Stooq: source=cache hit for {sym} ({len(df)} rows)")
    # Normalize column names
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Map common Stooq schema ("date,price") to OHLC-ish
    if "price" in df.columns and "close" not in df.columns:
        df["close"] = df["price"]
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()]
    # Filter by start/end if provided
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    df = df.sort_values("date").reset_index(drop=True)
    if "ticker" not in df.columns:
        df["ticker"] = sym
    # Ensure expected columns exist
    for col in ["open","high","low","close","adj_close","volume"]:
        if col not in df.columns:
            df[col] = pd.NA
    df = _to_canonical_ohlcv(df); return df.assign(ticker=sym)[["date","open","high","low","close","adj_close","volume","ticker"]]
# ------------------------------------------------------------------------------

# --- Public API aliases for provider registry / routers ----------------------
def load_daily_canon(symbol: str, start: str = "2000-01-01"):
    """Load Stooq daily OHLCV and return in canonical schema."""
    return _to_canonical_ohlcv(load_daily(symbol, start=start))

def fetch(symbol: str, start: str | None = None, end: str | None = None):
    """Generic fetch alias returning canonical schema."""
    df = load_daily(symbol, start=start or "2000-01-01")
    if end:
        df = df[df["date"] <= pd.to_datetime(end, errors="coerce")]
    return _to_canonical_ohlcv(df)

__all__ = ["load_daily", "load_daily_canon", "fetch_daily", "fetch", "_clean_ohlcv", "_to_canonical_ohlcv"]

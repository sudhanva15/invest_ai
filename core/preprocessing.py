import pandas as pd
# Robust logger
try:
    from .utils import log  # if your package exposes a structured logger
except Exception:
    import logging
    log = logging.getLogger("invest_ai.core.preprocessing")

def compute_returns(wide_prices: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Compute percent returns from a wide price frame.
    - Ensures DatetimeIndex
    - If freq provided (e.g., 'B','W','M'), resamples with .last() before pct_change
    """
    if wide_prices is None or wide_prices.empty:
        log.info("compute_returns: empty input")
        return pd.DataFrame()

    df = wide_prices.copy()
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            pass
    df = df.sort_index()

    # Optional resample
    if isinstance(df.index, pd.DatetimeIndex) and freq and freq.upper() != "D":
        try:
            if freq.upper() == "B":
                # Use asfreq + ffill to avoid deprecated fill_method behavior
                df = df.resample("B").asfreq().ffill()
            else:
                df = df.resample(freq).last()
        except Exception:
            # If resample fails, proceed without resampling
            pass

    import numpy as np
    # Avoid deprecated default fill_method behavior and sanitize infinities
    rets = df.pct_change(fill_method=None)
    rets = rets.replace([np.inf, -np.inf], np.nan)
    rets = rets.dropna(how="all")
    log.info(f"returns shape: {rets.shape}, freq={freq}")
    return rets


def ensure_tidy_price(df):
    """
    Ensure columns: date (datetime64), ticker (str), price (float).
    Accepts frames with 'price' OR 'adj_close' OR 'close' (fallbacks).
    Drops rows with empty/NaN tickers or missing dates/prices.
    """
    import pandas as pd
    import numpy as np

    if df is None or df.empty:
        return df

    d = df.copy()

    # Figure out which price column we have
    price_col = None
    for c in ["price", "adj_close", "close"]:
        if c in d.columns:
            price_col = c
            break
    # If still None, try OHLC fallback (use 'close' if present, else the first numeric)
    if price_col is None:
        for c in ["Close","Adj Close","Close*","Last"]:
            if c in d.columns:
                d = d.rename(columns={c: "close"})
                price_col = "close"
                break
    if price_col is None:
        # no usable price in this chunk -> return empty
        return d.iloc[0:0].assign(date=pd.to_datetime([]), ticker=pd.Series([], dtype=str), price=pd.Series([], dtype="float64"))

    # Normalize columns
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
    elif "Date" in d.columns:
        d["date"] = pd.to_datetime(d["Date"], errors="coerce")
    else:
        # No date column -> empty
        return d.iloc[0:0].assign(date=pd.to_datetime([]), ticker=pd.Series([], dtype=str), price=pd.Series([], dtype="float64"))

    # Ticker normalization
    if "ticker" not in d.columns:
        if "symbol" in d.columns:
            d["ticker"] = d["symbol"]
        elif "Ticker" in d.columns:
            d["ticker"] = d["Ticker"]
        else:
            # If a single-symbol frame was passed without 'ticker', just set NA -> drop later
            d["ticker"] = pd.NA

    d["ticker"] = d["ticker"].astype(str).str.strip()

    # Set 'price'
    d["price"] = pd.to_numeric(d[price_col], errors="coerce")

    # Clean up
    d = d[["date","ticker","price"]].copy()
    d = d.drop_duplicates(subset=["date","ticker"])
    d = d.sort_values(["date","ticker"])
    d = d.dropna(subset=["date","price"])
    d = d[d["ticker"].notna() & (d["ticker"] != "")]

    # Final dtypes
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["price"] = d["price"].astype("float64")

    return d


def to_wide(prices):
    """
    Pivot tidy -> wide. Accepts frames that may not yet have 'price' (uses ensure_tidy_price()).
    Ignores blank/None tickers automatically.
    """
    import pandas as pd
    if prices is None or len(prices)==0:
        return pd.DataFrame()

    tidy = ensure_tidy_price(prices)
    if tidy.empty:
        return tidy

    # Pivot to wide
    wide = tidy.pivot(index="date", columns="ticker", values="price").sort_index().dropna(how="all")
    return wide

__all__ = ["ensure_tidy_price", "to_wide", "compute_returns"]

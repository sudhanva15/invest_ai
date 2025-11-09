import pandas as pd
from .risk_metrics import annualize_return, annualize_vol, max_drawdown, sharpe

# -------------------------------------------------------------------
# Canonical price handling (prefer adjusted close when available)
def normalize_price_columns(df):
    """
    Lowercase cols; create a canonical 'price' column:
      adj_close -> close -> price
    Returns the same DataFrame (mutating) for convenience.
    """
    import pandas as pd
    if df is None or len(df) == 0:
        return df
    df.columns = [str(c).strip().lower() for c in df.columns]
    # if we only have 'price', keep it; otherwise build 'price'
    if "adj_close" in df.columns:
        df["price"] = df["adj_close"]
    elif "close" in df.columns:
        df["price"] = df["close"]
    elif "price" in df.columns:
        # already present; leave it
        pass
    else:
        # last resort: try 'value'
        if "value" in df.columns:
            df["price"] = df["value"]
        else:
            # create empty so downstream code fails loudly if accessed
            df["price"] = None
    # Ensure date is parsed and sorted if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values("date")
    return df

def pick_price_col(df):
    """
    Returns the chosen column name used for returns math.
    """
    for c in ("adj_close","close","price"):
        if c in df.columns:
            return c
    raise KeyError("No price-like column found (need one of adj_close/close/price).")

def price_series(df):
    """
    Returns the price series to use in backtests (adjusted when available).
    """
    return df[pick_price_col(df)]
# -------------------------------------------------------------------

def summarize_backtest(port_rets: pd.Series, rf=0.015):
    return {
        "CAGR": round(annualize_return(port_rets), 4),
        "Vol": round(annualize_vol(port_rets), 4),
        "Sharpe": round(sharpe(port_rets, rf=rf), 2),
        "MaxDD": round(max_drawdown(port_rets), 4),
        "N": len(port_rets)
    }

def equity_curve(port_rets: pd.Series):
    return (1 + port_rets).cumprod()

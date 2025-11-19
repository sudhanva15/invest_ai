"""Synthetic demo datasets for offline/demo environments.

This module keeps small in-code CSV snapshots so demo mode never touches disk
or real data providers. The price generator is deterministic per symbol so the
UI gets stable outputs between sessions without shipping large binary assets.
"""

from __future__ import annotations

import copy
import io
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List

import pandas as pd

_DEMO_PRICE_DAYS = 900
_DEMO_START = datetime(2018, 1, 2)

DEMO_UNIVERSE_SYMBOLS: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "LQD",
    "HYG",
    "GLD",
    "VNQ",
    "BIL",
    "DBC",
    "VWO",
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BND",
]

# Minimal-but-rich universe metadata used when demo mode bypasses filesystem
_DEMO_UNIVERSE_ASSETS: List[Dict] = [
    {"symbol": "SPY", "name": "S&P 500 ETF", "class": "equity_us", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.30, "risk_bucket": "core", "asset_class": "equity", "region": "US", "core_or_satellite": "core"},
    {"symbol": "QQQ", "name": "NASDAQ 100", "class": "equity_us", "sector": "technology", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "IWM", "name": "Russell 2000", "class": "equity_us", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "EFA", "name": "Developed Markets ex-US", "class": "equity_intl", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.25, "risk_bucket": "core", "asset_class": "equity", "region": "Intl Dev", "core_or_satellite": "core"},
    {"symbol": "VTI", "name": "Total US Market", "class": "equity_us", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.30, "risk_bucket": "core", "asset_class": "equity", "region": "US", "core_or_satellite": "core"},
    {"symbol": "VEA", "name": "Developed ex-US", "class": "equity_intl", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.25, "risk_bucket": "core", "asset_class": "equity", "region": "Intl Dev", "core_or_satellite": "core"},
    {"symbol": "EEM", "name": "Emerging Markets", "class": "equity_intl", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "EM", "core_or_satellite": "satellite"},
    {"symbol": "VWO", "name": "Vanguard Emerging Markets", "class": "equity_intl", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "EM", "core_or_satellite": "satellite"},
    {"symbol": "VNQ", "name": "US Real Estate", "class": "reit", "sector": "real_estate", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "reit", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "GLD", "name": "Gold", "class": "commodities", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "commodity", "region": "Global", "core_or_satellite": "satellite"},
    {"symbol": "DBC", "name": "Diversified Commodities", "class": "commodities", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "commodity", "region": "Global", "core_or_satellite": "satellite"},
    {"symbol": "AGG", "name": "US Aggregate Bond", "class": "bonds_ig", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.50, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "BND", "name": "Total Bond Market", "class": "bonds_ig", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.50, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "LQD", "name": "Investment Grade Credit", "class": "bonds_ig", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.40, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "HYG", "name": "High Yield Credit", "class": "high_yield", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "bond", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "TLT", "name": "Long Treasuries", "class": "bonds_tsy", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.60, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "IEF", "name": "7-10Y Treasuries", "class": "bonds_tsy", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.50, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "SHY", "name": "1-3Y Treasuries", "class": "bonds_tsy", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.40, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "BIL", "name": "T-Bills", "class": "cash", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.50, "risk_bucket": "core", "asset_class": "cash", "region": "US", "core_or_satellite": "core"},
    {"symbol": "MUB", "name": "US Munis", "class": "munis", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.30, "risk_bucket": "core", "asset_class": "bond", "region": "US", "core_or_satellite": "core"},
    {"symbol": "ACWI", "name": "Global Equities", "class": "equity_intl", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.30, "risk_bucket": "core", "asset_class": "equity", "region": "Global", "core_or_satellite": "core"},
    {"symbol": "USMV", "name": "Min Vol USA", "class": "equity_us", "sector": None, "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "XLK", "name": "Tech Sector", "class": "equity_sector", "sector": "information_technology", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.20, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "AAPL", "name": "Apple Inc.", "class": "single_stock", "sector": "information_technology", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "MSFT", "name": "Microsoft Corp.", "class": "single_stock", "sector": "information_technology", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "AMZN", "name": "Amazon.com Inc.", "class": "single_stock", "sector": "consumer_discretionary", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "GOOGL", "name": "Alphabet Inc.", "class": "single_stock", "sector": "communication_services", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "META", "name": "Meta Platforms Inc.", "class": "single_stock", "sector": "communication_services", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
    {"symbol": "TSLA", "name": "Tesla Inc.", "class": "single_stock", "sector": "consumer_discretionary", "provider": "demo", "eligibility": "retail", "max_weight_default": 0.10, "risk_bucket": "satellite", "asset_class": "equity", "region": "US", "core_or_satellite": "satellite"},
]

_DEMO_CAPS: Dict[str, Dict] = {
    "asset_class": {
        "equity": 0.65,
        "bond": 0.70,
        "commodity": 0.20,
        "cash": 0.20,
        "reit": 0.20,
    }
}

_DEMO_CONSTRAINTS: Dict[str, Dict] = {
    "core_min_years": 10.0,
    "sat_min_years": 7.0,
    "max_missing_pct": 5.0,
}


def _symbol_seed(symbol: str) -> int:
    return sum(ord(ch) for ch in symbol.upper())


def _build_demo_csv(symbol: str) -> str:
    seed = _symbol_seed(symbol)
    base_price = 50 + (seed % 200)
    drift = 0.0004 + ((seed % 17) / 10000)
    volatility = 0.01 + ((seed % 7) / 1000)
    rows = ["date,open,high,low,close,adj_close,volume"]
    price = base_price
    current = _DEMO_START
    for _ in range(_DEMO_PRICE_DAYS):
        noise = ((seed * (current.timetuple().tm_yday + 1)) % 13) - 6
        close = max(5.0, price * (1 + drift) + noise)
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_px = (high + low) / 2
        volume = 800_000 + ((seed + current.day) % 200_000)
        rows.append(
            f"{current.date()},{open_px:.2f},{high:.2f},{low:.2f},{close:.2f},{close:.2f},{int(volume)}"
        )
        price = close
        current += timedelta(days=1)
    return "\n".join(rows)


@lru_cache(maxsize=None)
def _demo_csv(symbol: str) -> str:
    return _build_demo_csv(symbol)


def load_demo_price_history(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Return a canonical OHLCV frame for the requested symbol from in-code CSV."""
    csv_text = _demo_csv(symbol.upper())
    df = pd.read_csv(io.StringIO(csv_text), parse_dates=["date"])
    df["ticker"] = symbol.upper()
    if start:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["date"] <= pd.to_datetime(end)]
    return df.reset_index(drop=True)


def get_demo_universe_frame() -> pd.DataFrame:
    """Return a lightweight universe dataframe used in demo mode."""
    return pd.DataFrame(_DEMO_UNIVERSE_ASSETS).copy()


def get_demo_universe_symbols() -> List[str]:
    return DEMO_UNIVERSE_SYMBOLS.copy()


def get_demo_caps() -> Dict:
    return copy.deepcopy(_DEMO_CAPS)


def get_demo_constraints() -> Dict:
    return copy.deepcopy(_DEMO_CONSTRAINTS)


def load_demo_macro_series(series_id: str) -> pd.DataFrame:
    """Generate a simple macro time series for demo mode (monthly frequency)."""
    start = _DEMO_START.replace(day=1)
    dates = pd.date_range(start=start, periods=150, freq="MS")
    base = 2.0 + (sum(ord(c) for c in series_id.upper()) % 5)
    slope = 0.05 + ((len(series_id) % 7) * 0.01)
    values = [base + idx * slope for idx, _ in enumerate(dates)]
    return pd.DataFrame({"date": dates, "value": values, "series_id": series_id})


__all__ = [
    "load_demo_price_history",
    "get_demo_universe_frame",
    "get_demo_universe_symbols",
    "get_demo_caps",
    "get_demo_constraints",
    "load_demo_macro_series",
    "DEMO_UNIVERSE_SYMBOLS",
]

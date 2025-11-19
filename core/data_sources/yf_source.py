from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

try:  # Optional dependency
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore

from core.env_tools import load_config, is_demo_mode
from core.demo_data import load_demo_price_history

_log = logging.getLogger(__name__)


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


DEMO_MODE = is_demo_mode()

_CFG = {}
if not DEMO_MODE:
    try:  # load_config may raise if config missing during tests
        _CFG = load_config()
    except Exception:  # pragma: no cover
        _CFG = {}

_DEFAULT_SKIP = True
_ENV_OVERRIDE = os.getenv("YF_SKIP_REMOTE")
if DEMO_MODE:
    YF_SKIP_REMOTE = True
elif _ENV_OVERRIDE is not None:
    YF_SKIP_REMOTE = _to_bool(_ENV_OVERRIDE, _DEFAULT_SKIP)
else:
    cfg_flag = None
    if isinstance(_CFG, dict):
        try:
            cfg_flag = (
                _CFG.get("data_sources", {})
                .get("yfinance", {})
                .get("skip_remote")
            )
        except Exception:  # pragma: no cover
            cfg_flag = None
    YF_SKIP_REMOTE = _to_bool(cfg_flag, _DEFAULT_SKIP)


def fetch_yfinance_history(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical prices via yfinance with defensive guards."""
    if DEMO_MODE:
        return load_demo_price_history(symbol, start_date, end_date)

    if YF_SKIP_REMOTE:
        _log.debug("[yfinance] Remote fetch skipped for %s (YF_SKIP_REMOTE=1)", symbol)
        return pd.DataFrame()

    if yf is None:  # pragma: no cover - dependency missing
        _log.warning("[yfinance] Package unavailable; returning empty frame for %s", symbol)
        return pd.DataFrame()

    params = {"progress": False}
    if start_date:
        params["start"] = str(start_date)
    if end_date:
        params["end"] = str(end_date)

    try:
        raw = yf.download(symbol, **params)
    except Exception as exc:  # pragma: no cover - capture API failures
        _log.warning("[yfinance] download failed for %s: %s", symbol, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    df = (
        raw.reset_index()
        .rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
    )
    df["ticker"] = symbol
    return df


__all__ = ["fetch_yfinance_history", "YF_SKIP_REMOTE"]

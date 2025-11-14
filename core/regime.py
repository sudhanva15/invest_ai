"""Simple macro regime scoring and tilts.

Signals from FRED series: DGS10 (10Y), T10Y2Y (curve), CPIAUCSL, UNRATE.
Produces small asset-class tilts that never violate constraints.
"""
from __future__ import annotations
import pandas as pd
from typing import Dict

try:
    from core.data_sources.fred import load_series
except Exception:  # fallback import path
    from .data_sources.fred import load_series  # type: ignore


def _last_growth(s: pd.Series, lookback: int = 6) -> float:
    s = s.dropna()
    if len(s) < lookback + 1:
        return 0.0
    return float((s.iloc[-1] - s.iloc[-1 - lookback]) / abs(s.iloc[-1 - lookback] or 1.0))


def macro_scores() -> Dict[str, float]:
    dgs10 = load_series("DGS10")
    curve = load_series("T10Y2Y")
    cpi = load_series("CPIAUCSL")
    unemp = load_series("UNRATE")

    # Normalize directions
    steep_curve = float(curve.tail(60).mean())  # positive = steep
    curve_trend = _last_growth(curve, 3)
    cpi_trend = _last_growth(cpi, 6)
    unemp_trend = _last_growth(unemp, 6)
    rates_trend = _last_growth(dgs10, 3)

    risk_on = max(0.0, steep_curve / 2.0) + max(0.0, -unemp_trend) + max(0.0, -cpi_trend)
    tightening = max(0.0, rates_trend) + max(0.0, -curve_trend)
    recessionary = max(0.0, -curve.tail(60).mean()) + max(0.0, unemp_trend)

    # Scale to ~[0,1]
    def clamp(x):
        return float(max(0.0, min(1.0, x)))

    return {
        "risk_on_score": clamp(risk_on),
        "tightening_score": clamp(tightening),
        "recessionary_score": clamp(recessionary),
    }


def current_regime_onehot() -> Dict[str, float]:
    s = macro_scores()
    # One-hot on argmax
    k = max(s, key=s.get)
    return {"risk_on": 1.0 if k == "risk_on_score" else 0.0,
            "tightening": 1.0 if k == "tightening_score" else 0.0,
            "recessionary": 1.0 if k == "recessionary_score" else 0.0}


def regime_tilt(asset_class: str) -> float:
    """Return small tilt by asset class.
    risk_on: equities +0.05, bonds -0.05
    recessionary: bonds +0.05, equities -0.05, gold +0.02
    tightening: cash +0.02, bonds -0.02
    """
    oh = current_regime_onehot()
    if asset_class.startswith("equity"):
        return 0.05 * oh.get("risk_on", 0.0) - 0.05 * oh.get("recessionary", 0.0)
    if asset_class.startswith("bonds"):
        return -0.05 * oh.get("risk_on", 0.0) + 0.05 * oh.get("recessionary", 0.0) - 0.02 * oh.get("tightening", 0.0)
    if asset_class == "commodities" or asset_class == "gold":
        return 0.02 * oh.get("recessionary", 0.0)
    if asset_class == "cash":
        return 0.02 * oh.get("tightening", 0.0)
    return 0.0

__all__ = ["macro_scores", "current_regime_onehot", "regime_tilt"]

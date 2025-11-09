from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd

def _safe_ann_vol(daily_ret: pd.Series) -> float:
    x = pd.to_numeric(daily_ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(x.std(ddof=0) * (252**0.5)) if len(x) else 0.0

def _safe_cagr(curve: pd.Series) -> float:
    if curve is None or len(curve)==0: return 0.0
    N = len(curve)
    end = float(curve.iloc[-1])
    if end <= 0: return 0.0
    return float(end**(252.0/N) - 1.0)

def _max_drawdown(curve: pd.Series) -> float:
    if curve is None or len(curve)==0: return 0.0
    peak = curve.cummax()
    dd = (curve/peak - 1.0).min()
    return float(dd)

def bootstrap_interval(daily_ret: pd.Series, stat: str="cagr", n: int=1000, alpha: float=0.05) -> tuple[float,float]:
    """Blockless bootstrap on daily returns (simple but fast).
    Returns (lo, hi) for the requested stat over 1-year horizon equivalent."""
    x = pd.to_numeric(daily_ret, errors="coerce").replace([np.inf,-np.inf], np.nan).dropna().values
    if x.size < 30:
        return (np.nan, np.nan)
    N = x.size
    stats = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        samp = x[rng.integers(0, N, N)]
        curve = (1.0 + pd.Series(samp)).cumprod()
        if stat == "cagr":
            stats.append(_safe_cagr(curve))
        elif stat == "ann_vol":
            stats.append(_safe_ann_vol(pd.Series(samp)))
        else:
            stats.append(_safe_cagr(curve))
    lo, hi = np.nanpercentile(stats, [alpha/2*100, (1-alpha/2)*100])
    return (float(lo), float(hi))

def effective_n_assets(weights: dict[str,float]) -> float:
    """Diversification proxy: 1 / sum(w^2). Higher = more diversified."""
    w = np.array([max(0.0, float(v)) for v in weights.values()])
    s = w.sum()
    if s <= 0: return 0.0
    w = w / s
    denom = float((w**2).sum())
    return float(1.0/denom) if denom>1e-12 else 0.0

def data_coverage(pxt: pd.DataFrame) -> float:
    """% of non-missing entries in price/return matrix."""
    if pxt is None or pxt.size==0: return 0.0
    total = pxt.size
    nonmiss = pxt.replace([np.inf,-np.inf], np.nan).notna().sum().sum()
    return float(nonmiss/total)

def provenance_hash(obj: dict) -> str:
    """Stable hash of inputs (symbols, dates, config), good for reproducibility logs."""
    b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()

def credibility_score(*, n_obs:int, cov:float, effN:float, sharpe_is:float, sharpe_oos:float|None=None) -> dict:
    """Compute a 0..100 score with a transparent breakdown."""
    s_size = min(1.0, n_obs/2500.0)
    s_cov  = max(0.0, min(1.0, cov))
    s_div  = max(0.0, min(1.0, (effN-1.0)/9.0))
    s_shp  = max(0.0, min(1.0, 0.2 + 0.3*sharpe_is + 0.5*max(0.0, sharpe_is-1.0)))
    if sharpe_oos is None:
        s_oos = 0.7
    else:
        gap = sharpe_is - sharpe_oos
        s_oos = max(0.0, 1.0 - max(0.0, gap)/1.0)
    w = dict(size=0.25, cov=0.20, div=0.20, shp=0.20, oos=0.15)
    raw = (s_size*w["size"] + s_cov*w["cov"] + s_div*w["div"] + s_shp*w["shp"] + s_oos*w["oos"])
    score = round(100.0*raw, 1)
    parts = {
        "sample_size": round(100*s_size,1),
        "coverage": round(100*s_cov,1),
        "diversification": round(100*s_div,1),
        "in_sample_sharpe": round(100*s_shp,1),
        "oos_penalty": round(100*s_oos,1),
    }
    return {"score": score, "parts": parts, "weights": w}

# === [WIRING PATCH] Compatibility helpers for UI & simplified calls ========
from typing import Sequence, Mapping
import math
import statistics

def credibility_score_simple(receipts: Mapping[str, float] | None = None) -> float:
    """
    Simple 0..1 credibility proxy used by UI smoke tests.
    Computes mean of clamped receipt values. Non-breaking alongside detailed score.
    """
    if not receipts:
        return 0.6
    vals = [max(0.0, min(1.0, float(v))) for v in receipts.values()]
    return float(sum(vals) / len(vals)) if vals else 0.6

def data_coverage_simple(series_lengths: Mapping[str, int] | None = None) -> float:
    """
    Normalize coverage from per-asset series lengths.
    Returns average(length_i / max_length) across assets, in 0..1.
    """
    if not series_lengths:
        return 0.0
    m = max(series_lengths.values()) if series_lengths else 0
    if m <= 0:
        return 0.0
    return float(sum((max(0, int(v)) / m) for v in series_lengths.values()) / len(series_lengths))

def effective_n_assets_seq(weights: Sequence[float] | None = None) -> float:
    """
    Herfindahl-based effective count for a sequence of weights.
    Mirrors effective_n_assets(dict) but accepts sequences.
    """
    if not weights:
        return 0.0
    w = [float(max(0.0, x)) for x in weights]
    s = sum(w)
    if s <= 0:
        return 0.0
    w = [x / s for x in w]
    hhi = sum(x * x for x in w)
    return 0.0 if hhi <= 1e-12 else float(1.0 / hhi)

def bootstrap_interval_simple(values: Sequence[float] | None = None, alpha: float = 0.05) -> tuple[float, float]:
    """
    Normal-approx interval for a generic sequence (used for quick UI intervals).
    This complements the bootstrap_interval(daily_ret, stat=..., ...) function above.
    """
    values = list(values or [])
    if not values:
        return (0.0, 0.0)
    m = statistics.fmean(values)
    sd = statistics.pstdev(values) or 0.0
    z = 1.96 if abs(alpha - 0.05) < 1e-6 else 1.96
    n = len(values)
    half = z * sd / math.sqrt(n) if n > 0 else 0.0
    return (float(m - half), float(m + half))
# ===========================================================================

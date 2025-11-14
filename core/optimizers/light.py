"""Lightweight optimizers with no heavy deps.

All functions return raw weight vectors (pd.Series) aligned to index.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def equal_weight(index) -> pd.Series:
    n = len(index)
    if n == 0:
        return pd.Series(dtype=float)
    w = np.full(n, 1.0 / n)
    return pd.Series(w, index=index)


def risk_parity(cov: pd.DataFrame, tol: float = 1e-8, max_iter: int = 500) -> pd.Series:
    """Simple equal-risk-contribution via fixed-point iteration (no CVX).
    Reference: Maillard et al. (2010) heuristic.
    """
    S = cov.values.astype(float)
    n = S.shape[0]
    w = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        port_var = float(w @ S @ w)
        if port_var <= 0:
            break
        # Marginal contribution to risk
        mrc = (S @ w)
        # Risk contributions
        rc = w * mrc
        # Target equal contributions
        target = port_var / n
        # Update rule
        grad = rc - target
        step = 0.5  # conservative
        w_new = np.clip(w - step * grad, 0.0, 1.0)
        s = w_new.sum()
        if s <= 0:
            w_new = np.full(n, 1.0 / n)
        else:
            w_new = w_new / s
        if np.linalg.norm(w_new - w, 1) < tol:
            w = w_new
            break
        w = w_new
    return pd.Series(w, index=cov.index)


def max_diversification(cov: pd.DataFrame) -> pd.Series:
    """Heuristic maximize diversification ratio: sum(w*vol)/sqrt(w' S w)
    Simple proxy: allocate proportional to inverse-variance, then smooth.
    """
    vol = np.sqrt(np.diag(cov.values))
    vol = np.where(vol <= 0, 1.0, vol)
    inv_vol = 1.0 / vol
    w = inv_vol / inv_vol.sum()
    # Smooth with equal weight for stability
    ew = np.full(len(w), 1.0 / len(w))
    w = 0.5 * w + 0.5 * ew
    return pd.Series(w, index=cov.index)

__all__ = ["equal_weight", "risk_parity", "max_diversification"]

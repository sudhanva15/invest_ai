"""Lightweight statistical helpers (V3).

Currently provides bootstrap-based ANOVA style winner selection for
candidate equity curves without assuming normality.

Functions are intentionally minimal (no external deps beyond numpy/pandas).
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Tuple


def anova_bootstrap(candidate_curves: Dict[str, pd.Series], n_boot: int = 2000, seed: int = 42) -> Dict[str, object]:
    """Bootstrap comparison of mean returns across candidate equity curves.

    Parameters
    ----------
    candidate_curves : dict[name -> pd.Series]
        Equity curve per candidate (indexed by datetime, numeric values). Curves may differ slightly in index; we align.
    n_boot : int
        Number of bootstrap resamples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        winner: str (name of highest mean candidate)
        p_value: float | None (approximate p-value of winner mean being greater; None if data too short)
        means: dict[name -> float]
        order: list[str] sorted by descending mean

    Notes
    -----
    - If fewer than 60 overlapping observations, we return p_value=None and just rank by Sharpe-like heuristic (mean/std).
    - Bootstrap procedure: resample (with replacement) aligned daily return differences between winner and others.
      p-value approximated as fraction of bootstrap samples where diff_mean <= 0.
    - We work on daily returns not raw level; convert curve to returns first.
    """
    if not candidate_curves:
        return {"winner": None, "p_value": None, "means": {}, "order": []}

    # Convert to aligned returns DataFrame
    rets = {}
    for name, curve in candidate_curves.items():
        try:
            if curve is None or len(curve) < 5:
                continue
            r = pd.Series(curve).pct_change().dropna()
            rets[name] = r
        except Exception:
            continue
    if not rets:
        return {"winner": None, "p_value": None, "means": {}, "order": []}

    df = pd.DataFrame(rets).dropna(how="any")
    if df.empty:
        return {"winner": None, "p_value": None, "means": {}, "order": []}

    # Basic stats
    means = df.mean().to_dict()
    # Sharpe-like ordering fallback if very short
    sharpe_like = (df.mean() / (df.std().replace(0, np.nan))).fillna(-1e9)
    order = list(sharpe_like.sort_values(ascending=False).index)
    winner = order[0]

    # If insufficient length, skip p-value
    if len(df) < 60:
        return {"winner": winner, "p_value": None, "means": means, "order": order}

    rng = np.random.default_rng(seed)
    base = df[winner]
    p_counts = 0
    other_means = []
    # Build distribution of mean differences winner - others pooled
    diffs = []
    for _ in range(n_boot):
        # Sample indices with replacement
        idx = rng.integers(0, len(df), size=len(df))
        sampled = df.iloc[idx]
        w_sample = sampled[winner]
        for other in df.columns:
            if other == winner:
                continue
            o_sample = sampled[other]
            diffs.append(w_sample.mean() - o_sample.mean())
    diffs_arr = np.array(diffs)
    # p-value: probability difference <= 0
    p_value = float((diffs_arr <= 0).mean()) if len(diffs_arr) else None

    return {"winner": winner, "p_value": p_value, "means": means, "order": order}

__all__ = ["anova_bootstrap"]

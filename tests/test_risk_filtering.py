from __future__ import annotations
from typing import List, Dict, Any

from core.recommendation_engine import select_candidates_for_risk_score, pick_portfolio_from_slider


def _make_candidate(name: str, mu: float, sigma: float) -> Dict[str, Any]:
    return {
        "name": name,
        "weights": {"SPY": 1.0},
        "metrics": {"CAGR": mu, "Vol": sigma, "Sharpe": mu / sigma if sigma > 0 else 0.0, "MaxDD": -0.1},
        "optimizer": "hrp",
        "sat_cap": 0.2,
    }


essential = [
    _make_candidate("low", 0.05, 0.06),
    _make_candidate("mid1", 0.07, 0.11),
    _make_candidate("mid2", 0.08, 0.125),
    _make_candidate("mid3", 0.09, 0.14),
    _make_candidate("high", 0.12, 0.18),
]


def test_select_candidates_for_risk_score_band():
    # risk_score 50 => target_sigma midway between 0.05 and 0.20 => 0.125
    # band 0.02 => accept [0.105, 0.145] → expect mid1 (0.11), mid2 (0.125), mid3 (0.14)
    filtered = select_candidates_for_risk_score(essential, risk_score=50.0, sigma_min=0.05, sigma_max=0.20, band=0.02)
    names = [c["name"] for c in filtered]
    assert names == ["mid1", "mid2", "mid3"]


def test_select_candidates_relax_when_too_few():
    # With a very tight band, first pass might yield <3; we relax once (x1.5). It may still be <3 depending on spacing.
    filtered = select_candidates_for_risk_score(essential, risk_score=50.0, sigma_min=0.05, sigma_max=0.20, band=0.005)
    assert len(filtered) >= 1
    assert filtered[0]["name"] == "mid2"


def test_pick_portfolio_from_slider_extremes():
    # Ensure sorted order by mu ascending is respected by caller; simulate sorted input
    sorted_cands = sorted(essential, key=lambda c: c["metrics"]["CAGR"])  # low .. high
    # slider 0.0 → choose low mu
    c0 = pick_portfolio_from_slider(sorted_cands, 0.0)
    assert c0 and c0["name"] == "low"
    # slider 1.0 → choose high mu
    c1 = pick_portfolio_from_slider(sorted_cands, 1.0)
    assert c1 and c1["name"] == "high"


def test_pick_portfolio_from_slider_midpoint():
    sorted_cands = sorted(essential, key=lambda c: c["metrics"]["CAGR"])  # low .. high
    # midpoint slider should target mean(mu)
    c = pick_portfolio_from_slider(sorted_cands, 0.5)
    assert c is not None
    # It should be one of the central items (mid2 or mid3 depending on spacing)
    assert c["name"] in {"mid2", "mid3"}

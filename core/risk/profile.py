"""Risk questionnaire to constraints mapping (V3 minimal).

Exposes risk_profile_to_constraints(answers) -> dict with keys:
- core_min (0..1)
- satellite_max (0..1)
- single_max (0..1)

This is intentionally simple and deterministic; tune as needed.
"""
from __future__ import annotations
from typing import Dict


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def risk_profile_to_constraints(answers: Dict) -> Dict[str, float]:
    """Map a small questionnaire to constraint overrides.

    Expected answers keys (all optional):
      - risk_attitude: one of {"very_low","low","moderate","high","very_high"}
      - drawdown_tolerance: one of {"low","medium","high"}
      - horizon_years: int
      - income_stability: one of {"low","medium","high"}

    Returns a dict suitable to override objective bounds:
      {"core_min": float, "satellite_max": float, "single_max": float}
    """
    ra = (answers or {}).get("risk_attitude", "moderate")
    dd = (answers or {}).get("drawdown_tolerance", "medium")
    hz = int((answers or {}).get("horizon_years", 10) or 10)
    inc = (answers or {}).get("income_stability", "medium")

    # Base defaults
    core_min = 0.65
    sat_max = 0.35
    single_max = 0.07

    # Adjust by risk attitude
    if ra in {"very_low", "low"}:
        core_min += 0.10
        sat_max -= 0.10
        single_max -= 0.02
    elif ra in {"high", "very_high"}:
        core_min -= 0.10
        sat_max += 0.10
        single_max += 0.02

    # Drawdown tolerance tweak
    if dd == "low":
        core_min += 0.05
        sat_max -= 0.05
    elif dd == "high":
        core_min -= 0.05
        sat_max += 0.05

    # Horizon effect
    if hz < 5:
        core_min += 0.05
        sat_max -= 0.05
    elif hz > 15:
        core_min -= 0.05
        sat_max += 0.05

    # Income stability
    if inc == "low":
        core_min += 0.05
        sat_max -= 0.05

    core_min = clamp(core_min, 0.40, 0.90)
    sat_max = clamp(sat_max, 0.10, 0.60)
    single_max = clamp(single_max, 0.03, 0.12)

    return {"core_min": float(core_min), "satellite_max": float(sat_max), "single_max": float(single_max)}

__all__ = ["risk_profile_to_constraints"]

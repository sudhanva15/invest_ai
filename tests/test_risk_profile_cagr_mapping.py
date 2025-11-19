"""
Tests for TRUE_RISK → CAGR band mapping (Phase 3).

Validates that:
1. CAGR band increases with risk score
2. Approximate numeric ranges match calibration anchors
3. Edge cases (0, 100) handled correctly
"""

import pytest
from core.risk_profile import map_true_risk_to_cagr_band, compute_risk_profile


def test_cagr_band_increases_with_risk():
    """CAGR band should increase monotonically with TRUE_RISK."""
    # Test increasing risk scores
    scores = [20, 50, 80]
    bands = [map_true_risk_to_cagr_band(s) for s in scores]
    
    # Extract targets
    targets = [band[1] for band in bands]
    
    # Should be strictly increasing
    assert targets[0] < targets[1] < targets[2], \
        f"CAGR targets should increase: {targets}"
    
    # Minimums should also increase
    mins = [band[0] for band in bands]
    assert mins[0] < mins[1] < mins[2], \
        f"CAGR minimums should increase: {mins}"


def test_cagr_band_approximate_ranges():
    """Verify CAGR bands match expected calibration anchors."""
    # Conservative (risk ≈ 20): ~5-6% CAGR
    cagr_min_20, cagr_target_20 = map_true_risk_to_cagr_band(20)
    assert 0.04 <= cagr_min_20 <= 0.06, f"Conservative min out of range: {cagr_min_20:.2%}"
    assert 0.05 <= cagr_target_20 <= 0.07, f"Conservative target out of range: {cagr_target_20:.2%}"
    
    # Moderate (risk ≈ 50): ~8-9% CAGR
    cagr_min_50, cagr_target_50 = map_true_risk_to_cagr_band(50)
    assert 0.07 <= cagr_min_50 <= 0.09, f"Moderate min out of range: {cagr_min_50:.2%}"
    assert 0.08 <= cagr_target_50 <= 0.10, f"Moderate target out of range: {cagr_target_50:.2%}"
    
    # Aggressive (risk ≈ 80): ~10-11% CAGR
    cagr_min_80, cagr_target_80 = map_true_risk_to_cagr_band(80)
    assert 0.09 <= cagr_min_80 <= 0.11, f"Aggressive min out of range: {cagr_min_80:.2%}"
    assert 0.10 <= cagr_target_80 <= 0.12, f"Aggressive target out of range: {cagr_target_80:.2%}"


def test_cagr_band_edge_cases():
    """Test boundary conditions (0, 100, negative, >100)."""
    # Risk = 0 (ultra-conservative)
    cagr_min_0, cagr_target_0 = map_true_risk_to_cagr_band(0)
    assert cagr_min_0 >= 0.0, "CAGR min should be non-negative"
    assert cagr_target_0 > cagr_min_0, "CAGR target should exceed min"
    
    # Risk = 100 (ultra-aggressive)
    cagr_min_100, cagr_target_100 = map_true_risk_to_cagr_band(100)
    assert cagr_min_100 <= 0.15, "CAGR min should be reasonable even at max risk"
    assert cagr_target_100 <= 0.18, "CAGR target should be realistic (<18%)"
    
    # Negative risk (should clamp to 0)
    cagr_min_neg, cagr_target_neg = map_true_risk_to_cagr_band(-10)
    assert cagr_min_neg == cagr_min_0, "Negative risk should clamp to 0"
    assert cagr_target_neg == cagr_target_0
    
    # Risk > 100 (should clamp to 100)
    cagr_min_over, cagr_target_over = map_true_risk_to_cagr_band(150)
    assert cagr_min_over == cagr_min_100, "Risk >100 should clamp to 100"
    assert cagr_target_over == cagr_target_100


def test_cagr_band_continuity():
    """Verify smooth interpolation between anchors."""
    # Test intermediate values between anchors
    # Between 20 and 50
    cagr_min_35, cagr_target_35 = map_true_risk_to_cagr_band(35)
    cagr_min_20, cagr_target_20 = map_true_risk_to_cagr_band(20)
    cagr_min_50, cagr_target_50 = map_true_risk_to_cagr_band(50)
    
    # Should be between the anchors
    assert cagr_min_20 < cagr_min_35 < cagr_min_50
    assert cagr_target_20 < cagr_target_35 < cagr_target_50
    
    # Between 50 and 80
    cagr_min_65, cagr_target_65 = map_true_risk_to_cagr_band(65)
    cagr_min_80, cagr_target_80 = map_true_risk_to_cagr_band(80)
    
    assert cagr_min_50 < cagr_min_65 < cagr_min_80
    assert cagr_target_50 < cagr_target_65 < cagr_target_80


def test_risk_profile_includes_cagr_fields():
    """Verify RiskProfileResult populates CAGR fields automatically."""
    questionnaire = {
        "q1_time_horizon": 70,
        "q2_loss_tolerance": 60,
        "q3_reaction_20_drop": 50,
        "q4_income_stability": 55,
        "q5_dependence_on_money": 65,
        "q6_investing_experience": 60,
        "q7_safety_net": 55,
        "q8_goal_type": 60,
    }
    income_profile = {
        "annual_income": 120000,
        "income_stability": "Stable",
        "emergency_fund_months": 6.0,
        "investable_amount": 20000,
        "monthly_expenses": 4000,
        "outstanding_debt": 10000,
        "objective": "Balanced",
        "horizon_years": 10,
    }
    
    profile = compute_risk_profile(questionnaire, income_profile, slider_score=60.0)
    
    # CAGR fields should exist and be reasonable
    assert hasattr(profile, "cagr_min"), "Profile should have cagr_min"
    assert hasattr(profile, "cagr_target"), "Profile should have cagr_target"
    
    assert 0.0 <= profile.cagr_min <= 0.20, f"cagr_min out of range: {profile.cagr_min:.2%}"
    assert 0.0 <= profile.cagr_target <= 0.20, f"cagr_target out of range: {profile.cagr_target:.2%}"
    assert profile.cagr_min < profile.cagr_target, "cagr_min should be < cagr_target"
    
    # For moderate risk (~60), should be in moderate CAGR range (7-10%)
    assert 0.06 <= profile.cagr_min <= 0.10, \
        f"Moderate risk should have moderate CAGR min: {profile.cagr_min:.2%}"
    assert 0.07 <= profile.cagr_target <= 0.11, \
        f"Moderate risk should have moderate CAGR target: {profile.cagr_target:.2%}"


def test_risk_profile_backward_compat():
    """Verify existing code paths still work without cagr_min/cagr_target."""
    from core.risk_profile import RiskProfileResult
    
    # Old-style construction (without CAGR fields)
    profile_old = RiskProfileResult(
        questionnaire_score=60,
        facts_score=55,
        combined_score=57.5,
        slider_score=60,
        true_risk=58.25,
        vol_min=0.15,
        vol_target=0.17,
        vol_max=0.19,
        label="Moderate",
        horizon_years=10,
        objective="Balanced"
    )
    
    # Should auto-populate CAGR fields from true_risk
    assert hasattr(profile_old, "cagr_min")
    assert hasattr(profile_old, "cagr_target")
    assert profile_old.cagr_min > 0.0
    assert profile_old.cagr_target > profile_old.cagr_min


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

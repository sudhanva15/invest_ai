import math
from typing import Dict, Any

from core.risk_profile import RiskProfileResult, compute_risk_profile


def test_compute_risk_profile_basic():
    # Synthetic questionnaire: 8 questions with integer answers mapped 0-100
    questionnaire: Dict[str, Any] = {
        "q1_time_horizon": 70,
        "q2_loss_tolerance": 60,
        "q3_reaction_20_drop": 50,
        "q4_income_stability": 55,
        "q5_dependence_on_money": 65,
        "q6_investing_experience": 60,
        "q7_safety_net": 55,
        "q8_goal_type": 60,
    }

    # Synthetic income profile
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

    result = compute_risk_profile(questionnaire, income_profile, slider_score=65.0)

    # Type and range checks
    assert isinstance(result, RiskProfileResult)
    assert 0.0 <= result.true_risk <= 100.0

    # Vol band ordering
    assert 0.0 <= result.vol_min < result.vol_target < result.vol_max

    # Label non-empty
    assert isinstance(result.label, str) and len(result.label) > 0

    # Backward-compat properties
    assert math.isclose(result.sigma_target, result.vol_target)
    assert math.isclose(result.band_min_vol, result.vol_min)
    assert math.isclose(result.band_max_vol, result.vol_max)

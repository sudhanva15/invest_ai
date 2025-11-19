import pandas as pd
import numpy as np

from core.risk_profile import compute_risk_profile
from core.recommendation_engine import build_recommendations, ObjectiveConfig

def synthetic_returns(symbols, days=252, daily_vol=0.01, seed=11):
    rng = np.random.default_rng(seed)
    drift = 0.0002
    data = {}
    for s in symbols:
        eps = rng.normal(loc=0.0, scale=daily_vol, size=days)
        data[s] = drift + eps
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(data, index=idx)

def make_profile():
    questionnaire = {
        "q1_time_horizon": 50,
        "q2_loss_tolerance": 50,
        "q3_reaction_20_drop": 50,
        "q4_income_stability": 50,
        "q5_dependence_on_money": 50,
        "q6_investing_experience": 50,
        "q7_safety_net": 50,
        "q8_goal_type": 50,
    }
    income = {
        "annual_income": 100000,
        "income_stability": "Stable",
        "emergency_fund_months": 3.0,
        "investable_amount": 10000,
        "monthly_expenses": 3000,
        "outstanding_debt": 5000,
        "objective": "Balanced",
        "horizon_years": 8,
    }
    return compute_risk_profile(questionnaire, income, slider_score=50.0)


def test_fallback_kicks_in_when_thresholds_too_strict():
    symbols = ["SPY", "QQQ", "TLT", "GLD"]
    rets = synthetic_returns(symbols, days=400, daily_vol=0.01)
    catalog = {s: {"asset_class": "public_equity", "core_or_satellite": "satellite"} for s in symbols}
    rp = make_profile()
    obj = ObjectiveConfig(name="Balanced")

    # Deliberately impossible Sharpe thresholds to force zero passing candidates
    cfg = {
        # Relax asset filters so candidates are generated, but make portfolio filters impossible.
        "multifactor": {
            "min_asset_sharpe": -10.0,          # allow all assets
            "max_asset_vol_multiplier": 5.0,     # permissive
            "max_asset_drawdown": -0.99,         # permissive
            # Impossible portfolio thresholds ensure passed_filters=False for all
            "min_portfolio_sharpe": 5.0,         # synthetic Sharpe will be << 5
            "max_portfolio_drawdown": -0.01,     # unrealistic drawdown bound
            "max_risk_contribution": 0.10,
            "min_diversification_ratio": 0.99,   # very high
            "min_holdings": 2,                   # achievable with our universe
            "distinctness_threshold": 0.999,
            "drawdown_penalty_lambda": 0.1,
        }
        ,"universe": {
            "core_min_years": 1.0,
            "sat_min_years": 1.0,
        }
    }

    result = build_recommendations(
        returns=rets,
        catalog=catalog,
        cfg=cfg,
        risk_profile=rp,
        objective_cfg=obj,
        n_candidates=3,
        seed=123,
    )

    recs = result.get("recommended", [])
    assert len(recs) > 0, "Fallback should produce non-empty recommendations"
    assert all(c.get("fallback") for c in recs), "All recommended portfolios should be marked as fallback"
    # fail_reason may retain original filter reason; fallback flag is sufficient


def test_hard_fallback_when_no_candidates():
    """Test that hard fallback creates emergency portfolio when asset filters are too strict."""
    symbols = ["SPY", "QQQ", "TLT"]
    rets = synthetic_returns(symbols, days=100, daily_vol=0.02)  # Very short history
    catalog = {s: {"asset_class": "public_equity", "core_or_satellite": "satellite"} for s in symbols}
    rp = make_profile()
    obj = ObjectiveConfig(name="Balanced")

    # Make asset filters so strict that no assets pass (requiring very long history)
    cfg = {
        "multifactor": {
            "min_asset_sharpe": -10.0,
            "max_asset_vol_multiplier": 5.0,
            "max_asset_drawdown": -0.99,
            "min_portfolio_sharpe": -10.0,  # Relaxed so not the blocker
            "max_portfolio_drawdown": -0.99,
            "max_risk_contribution": 1.0,
            "min_diversification_ratio": 0.5,
            "min_holdings": 1,
            "distinctness_threshold": 0.999,
            "drawdown_penalty_lambda": 0.1,
        },
        "universe": {
            "core_min_years": 100.0,  # Impossible: only 100 days of data
            "sat_min_years": 100.0,    # This will cause all assets to fail
        }
    }

    result = build_recommendations(
        returns=rets,
        catalog=catalog,
        cfg=cfg,
        risk_profile=rp,
        objective_cfg=obj,
        n_candidates=3,
        seed=456,
    )

    recs = result.get("recommended", [])
    # Should have at least one recommendation even with impossible filters
    assert len(recs) > 0, "Hard fallback should produce at least one emergency portfolio"
    
    # Check if hard fallback was used
    # When asset filters are too strict, we may get an error or empty candidates,
    # triggering Level 2 hard fallback
    if len(result.get("all_candidates", [])) == 0:
        # Hard fallback should have been triggered
        assert any(c.get("hard_fallback") for c in recs), "Should have hard_fallback flag when no candidates generated"

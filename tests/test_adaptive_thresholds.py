"""
Tests for adaptive portfolio filtering thresholds (Phase 3 Task 2).

Validates that:
1. derive_portfolio_thresholds() produces sensible risk-scaled thresholds
2. Conservative profiles get easier CAGR/Sharpe, stricter drawdown
3. Aggressive profiles demand higher CAGR/Sharpe, tolerate larger drawdown
4. portfolio_passes_filters() respects dynamic_thresholds when provided
"""

import pytest
import pandas as pd
import numpy as np
from core.multifactor import derive_portfolio_thresholds, portfolio_passes_filters
from core.risk_profile import RiskProfileResult


@pytest.fixture
def mock_config():
    """Mock config dict matching config.yaml structure."""
    return {
        "multifactor": {
            "min_portfolio_sharpe": 0.3,
            "max_portfolio_drawdown": -0.50,
            "vol_soft_lower_factor": 0.6,
            "max_risk_contribution": 0.40,
            "min_diversification_ratio": 1.2,
            "min_holdings": 3,
        }
    }


@pytest.fixture
def conservative_profile():
    """Conservative profile: risk=20, vol_target=13%, CAGR_target=6%."""
    return RiskProfileResult(
        questionnaire_score=20,
        facts_score=18,
        combined_score=19,
        slider_score=20,
        true_risk=20.0,
        vol_min=0.12,
        vol_target=0.13,
        vol_max=0.14,
        band_min_vol=0.11,
        band_max_vol=0.15,
        label="Conservative",
        horizon_years=20,
        objective="Preserve Capital",
        cagr_min=0.05,
        cagr_target=0.06,
    )


@pytest.fixture
def moderate_profile():
    """Moderate profile: risk=50, vol_target=17%, CAGR_target=9%."""
    return RiskProfileResult(
        questionnaire_score=50,
        facts_score=48,
        combined_score=49,
        slider_score=50,
        true_risk=50.0,
        vol_min=0.16,
        vol_target=0.17,
        vol_max=0.18,
        band_min_vol=0.15,
        band_max_vol=0.19,
        label="Moderate",
        horizon_years=10,
        objective="Balanced",
        cagr_min=0.08,
        cagr_target=0.09,
    )


@pytest.fixture
def aggressive_profile():
    """Aggressive profile: risk=80, vol_target=21%, CAGR_target=11%."""
    return RiskProfileResult(
        questionnaire_score=80,
        facts_score=78,
        combined_score=79,
        slider_score=80,
        true_risk=80.0,
        vol_min=0.20,
        vol_target=0.21,
        vol_max=0.22,
        band_min_vol=0.19,
        band_max_vol=0.23,
        label="Aggressive",
        horizon_years=5,
        objective="Aggressive Growth",
        cagr_min=0.10,
        cagr_target=0.11,
    )


def test_derive_thresholds_conservative(conservative_profile, mock_config):
    """Conservative profile should get lower CAGR/Sharpe, stricter drawdown."""
    thresholds = derive_portfolio_thresholds(conservative_profile, mock_config)
    
    # CAGR should stay within conservative band (relaxed Stage 2 may lower slightly)
    assert 0.045 <= thresholds["min_cagr"] <= 0.05, "Conservative CAGR min should remain near 5%"
    
    # Sharpe should be below baseline (0.3)
    assert 0.15 <= thresholds["min_sharpe"] <= 0.30, \
        f"Conservative Sharpe too high: {thresholds['min_sharpe']:.2f}"
    
    # Drawdown should remain tighter than aggressive profiles (less negative than -0.50)
    assert -0.45 <= thresholds["max_drawdown"] <= -0.25, \
        f"Conservative drawdown too lenient: {thresholds['max_drawdown']:.1%}"
    
    # Vol bounds should match profile
    assert thresholds["vol_lower"] < thresholds["vol_upper"]
    assert thresholds["vol_upper"] == conservative_profile.band_max_vol


def test_derive_thresholds_moderate(moderate_profile, mock_config):
    """Moderate profile should get middle-ground thresholds."""
    thresholds = derive_portfolio_thresholds(moderate_profile, mock_config)
    
    # CAGR should stay near profile minimum (~8%) with limited relaxation
    assert 0.073 <= thresholds["min_cagr"] <= 0.08, "Moderate CAGR min should stay near 8%"
    
    # Sharpe should be near baseline (0.3)
    assert 0.28 <= thresholds["min_sharpe"] <= 0.40, \
        f"Moderate Sharpe out of range: {thresholds['min_sharpe']:.2f}"
    
    # Drawdown should be near baseline (-0.50)
    assert -0.60 <= thresholds["max_drawdown"] <= -0.40, \
        f"Moderate drawdown out of range: {thresholds['max_drawdown']:.1%}"


def test_derive_thresholds_aggressive(aggressive_profile, mock_config):
    """Aggressive profile should demand higher CAGR/Sharpe, tolerate larger drawdown."""
    thresholds = derive_portfolio_thresholds(aggressive_profile, mock_config)
    
    # CAGR should remain close to profile minimum (~10%) while allowing light fallback slack
    assert 0.086 <= thresholds["min_cagr"] <= 0.10, "Aggressive CAGR min should stay near 10%"
    
    # Sharpe should stay at or above baseline (0.3) even with relaxation
    assert 0.30 <= thresholds["min_sharpe"] <= 0.50, \
        f"Aggressive Sharpe too low: {thresholds['min_sharpe']:.2f}"
    
    # Drawdown should be more lenient (more negative) than baseline (-0.50)
    assert -0.80 <= thresholds["max_drawdown"] <= -0.55, \
        f"Aggressive drawdown too strict: {thresholds['max_drawdown']:.1%}"


def test_thresholds_increase_with_risk(conservative_profile, moderate_profile, aggressive_profile, mock_config):
    """Verify monotonic increase: Conservative < Moderate < Aggressive."""
    cons = derive_portfolio_thresholds(conservative_profile, mock_config)
    mod = derive_portfolio_thresholds(moderate_profile, mock_config)
    agg = derive_portfolio_thresholds(aggressive_profile, mock_config)
    
    # CAGR should increase
    assert cons["min_cagr"] < mod["min_cagr"] < agg["min_cagr"], \
        "CAGR thresholds should increase with risk"
    
    # Sharpe should increase
    assert cons["min_sharpe"] < mod["min_sharpe"] < agg["min_sharpe"], \
        "Sharpe thresholds should increase with risk"
    
    # Drawdown tolerance should increase (more negative)
    assert cons["max_drawdown"] > mod["max_drawdown"] > agg["max_drawdown"], \
        "Drawdown tolerance should increase (more negative) with risk"


def test_portfolio_passes_with_dynamic_thresholds(moderate_profile, mock_config):
    """Verify portfolio_passes_filters respects dynamic_thresholds."""
    # Create portfolio stats that pass baseline but fail strict CAGR
    portfolio_stats = {
        "valid": True,
        "cagr": 0.07,  # 7% CAGR
        "sharpe": 0.35,
        "volatility": 0.17,
        "max_drawdown": -0.45,
        "diversification_ratio": 1.5,
        "num_holdings": 5,
    }
    # Risk contributions should sum to ~portfolio volatility (0.17)
    # Each asset contributes ~3.4% (0.034), well below 40% threshold (0.068)
    risk_contrib = pd.Series([0.034, 0.036, 0.032, 0.033, 0.035], index=["A", "B", "C", "D", "E"])
    
    # With strict thresholds (min_cagr=0.08), should fail
    strict = derive_portfolio_thresholds(moderate_profile, mock_config)
    passes, reason = portfolio_passes_filters(
        portfolio_stats, risk_contrib, mock_config, moderate_profile, dynamic_thresholds=strict
    )
    assert not passes, "Should fail with strict CAGR=8%"
    assert "low cagr" in reason.lower(), f"Expected CAGR failure, got: {reason}"
    
    # With relaxed thresholds (min_cagr=0.06), should pass
    relaxed = {
        "min_cagr": 0.06,  # Lowered from 0.08
        "min_sharpe": 0.25,  # Lowered from 0.35
        "max_drawdown": -0.60,  # Relaxed from -0.50
        "vol_lower": 0.12,
        "vol_upper": 0.22,
    }
    passes_relaxed, reason_relaxed = portfolio_passes_filters(
        portfolio_stats, risk_contrib, mock_config, moderate_profile, dynamic_thresholds=relaxed
    )
    assert passes_relaxed, f"Should pass with relaxed thresholds, but failed: {reason_relaxed}"


def test_portfolio_fails_without_dynamic_thresholds(moderate_profile, mock_config):
    """Verify static config thresholds still work when dynamic_thresholds=None."""
    # Portfolio with Sharpe below baseline (0.3)
    portfolio_stats = {
        "valid": True,
        "cagr": 0.09,
        "sharpe": 0.25,  # Below min_portfolio_sharpe=0.3
        "volatility": 0.17,
        "max_drawdown": -0.45,
        "diversification_ratio": 1.5,
        "num_holdings": 5,
    }
    # Risk contributions should be reasonable
    risk_contrib = pd.Series([0.034, 0.036, 0.032, 0.033, 0.035], index=["A", "B", "C", "D", "E"])
    
    passes, reason = portfolio_passes_filters(
        portfolio_stats, risk_contrib, mock_config, moderate_profile, dynamic_thresholds=None
    )
    assert not passes, "Should fail with static config thresholds"
    assert "low sharpe" in reason.lower(), f"Expected Sharpe failure, got: {reason}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

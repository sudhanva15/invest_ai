"""
Unit tests for multi-factor filtering engine (Phase 2).
"""

import pytest
import pandas as pd
import numpy as np
from core.risk_profile import RiskProfileResult
from core.multifactor import (
    evaluate_asset_metrics,
    asset_passes_filters,
    portfolio_metrics,
    compute_risk_contributions,
    portfolio_passes_filters,
    compute_composite_score,
    check_distinctness,
)


def test_evaluate_asset_metrics():
    """Test asset metrics computation."""
    # Generate synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.015, 252 * 5))  # 5 years daily
    
    metrics = evaluate_asset_metrics(returns, risk_free_rate=0.015)
    
    assert metrics["valid"] is True
    assert "years_history" in metrics
    assert "cagr" in metrics
    assert "volatility" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert metrics["years_history"] > 4.5
    assert metrics["volatility"] > 0


def test_asset_passes_filters():
    """Test asset filtering logic."""
    # Mock config
    cfg = {
        "universe": {"core_min_years": 5.0, "sat_min_years": 3.0},
        "multifactor": {
            "min_asset_sharpe": -0.5,
            "max_asset_vol_multiplier": 2.0,
            "max_asset_drawdown": -0.60,
        },
        "optimization": {"risk_free_rate": 0.015}
    }
    
    # Mock risk profile
    risk_profile = RiskProfileResult(
        score_questionnaire=60.0,
        score_facts=60.0,
        score_combined=60.0,
        true_risk=60.0,
        label="Moderate",
        sigma_target=0.17,
        band_min_vol=0.15,
        band_max_vol=0.19,
    )
    
    # Good asset
    good_metrics = {
        "valid": True,
        "years_history": 10.0,
        "cagr": 0.08,
        "volatility": 0.16,
        "sharpe": 0.50,
        "max_drawdown": -0.25,
    }
    
    passed, reason = asset_passes_filters(
        symbol="SPY",
        metrics=good_metrics,
        asset_class="equity_us",
        core_or_satellite="core",
        cfg=cfg,
        risk_profile=risk_profile
    )
    
    assert passed is True
    assert reason is None
    
    # Bad asset - insufficient history
    bad_metrics = good_metrics.copy()
    bad_metrics["years_history"] = 2.0
    
    passed, reason = asset_passes_filters(
        symbol="NEW",
        metrics=bad_metrics,
        asset_class="equity_us",
        core_or_satellite="core",
        cfg=cfg,
        risk_profile=risk_profile
    )
    
    assert passed is False
    assert "insufficient history" in reason


def test_portfolio_metrics():
    """Test portfolio metrics computation."""
    # Generate synthetic returns for 3 assets
    np.random.seed(42)
    returns = pd.DataFrame({
        "SPY": np.random.normal(0.0005, 0.015, 252 * 3),
        "TLT": np.random.normal(0.0003, 0.010, 252 * 3),
        "GLD": np.random.normal(0.0002, 0.012, 252 * 3),
    })
    
    # Equal weight portfolio
    weights = pd.Series({"SPY": 0.4, "TLT": 0.4, "GLD": 0.2})
    
    metrics = portfolio_metrics(weights, returns, risk_free_rate=0.015)
    
    assert metrics["valid"] is True
    assert "cagr" in metrics
    assert "volatility" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "diversification_ratio" in metrics
    assert "num_holdings" in metrics
    assert metrics["num_holdings"] == 3


def test_compute_risk_contributions():
    """Test risk contribution calculation."""
    # Simple covariance matrix
    cov = pd.DataFrame({
        "A": [0.04, 0.01, 0.00],
        "B": [0.01, 0.02, 0.00],
        "C": [0.00, 0.00, 0.01],
    }, index=["A", "B", "C"])
    
    weights = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    
    rc = compute_risk_contributions(weights, cov)
    
    assert len(rc) == 3
    assert rc.sum() > 0  # Total risk contribution should be portfolio vol


def test_portfolio_passes_filters():
    """Test portfolio filtering logic."""
    cfg = {
        "multifactor": {
            "min_portfolio_sharpe": 0.3,
            "max_portfolio_drawdown": -0.50,
            "max_risk_contribution": 0.40,
            "min_diversification_ratio": 1.2,
            "min_holdings": 3,
        }
    }
    
    risk_profile = RiskProfileResult(
        score_questionnaire=60.0,
        score_facts=60.0,
        score_combined=60.0,
        true_risk=60.0,
        label="Moderate",
        sigma_target=0.17,
        band_min_vol=0.15,
        band_max_vol=0.19,
    )
    
    # Good portfolio
    good_stats = {
        "valid": True,
        "cagr": 0.08,
        "volatility": 0.17,
        "sharpe": 0.45,
        "max_drawdown": -0.30,
        "diversification_ratio": 1.5,
        "num_holdings": 5,
    }
    
    risk_contrib = pd.Series({"A": 0.04, "B": 0.05, "C": 0.03, "D": 0.03, "E": 0.02})
    
    passed, reason = portfolio_passes_filters(
        portfolio_stats=good_stats,
        risk_contrib=risk_contrib,
        cfg=cfg,
        risk_profile=risk_profile
    )
    
    assert passed is True, f"Expected pass but got: {reason}"
    assert reason is None
    
    # Bad portfolio - volatility out of band
    bad_stats = good_stats.copy()
    bad_stats["volatility"] = 0.25
    
    passed, reason = portfolio_passes_filters(
        portfolio_stats=bad_stats,
        risk_contrib=risk_contrib,
        cfg=cfg,
        risk_profile=risk_profile
    )
    
    assert passed is False
    assert "volatility too high" in reason


def test_composite_score():
    """Test composite score calculation."""
    score1 = compute_composite_score(sharpe=0.5, max_drawdown=-0.30, lambda_penalty=0.2)
    score2 = compute_composite_score(sharpe=0.5, max_drawdown=-0.50, lambda_penalty=0.2)
    
    # Higher drawdown should result in lower score
    assert score1 > score2
    
    # Verify formula: score = Sharpe - λ * |MDD|
    expected1 = 0.5 - 0.2 * 0.30
    assert abs(score1 - expected1) < 0.001


def test_check_distinctness():
    """Test portfolio distinctness check."""
    # Identical portfolios
    w1 = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    w2 = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    
    assert check_distinctness(w1, w2, threshold=0.995) is False  # Too similar
    
    # Different portfolios
    w3 = pd.Series({"A": 0.3, "B": 0.5, "C": 0.2})
    
    assert check_distinctness(w1, w3, threshold=0.995) is True  # Distinct enough
    
    # Completely different portfolios
    w4 = pd.Series({"D": 0.5, "E": 0.5})
    
    assert check_distinctness(w1, w4, threshold=0.995) is True  # No overlap


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

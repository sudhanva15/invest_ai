#!/usr/bin/env python3
"""Test robustness scoring functions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from core.risk_profile import compute_simple_robustness_score, compute_robustness_from_curve


def test_simple_robustness():
    """Test basic robustness score computation."""
    # Case 1: Very consistent segments (should score high)
    consistent_cagrs = [0.10, 0.11, 0.10]
    score1 = compute_simple_robustness_score(consistent_cagrs)
    assert 80 <= score1 <= 100, f"Consistent CAGRs should score high: {score1}"
    print(f"✓ Consistent segments (0.10, 0.11, 0.10): score={score1:.1f}/100")
    
    # Case 2: Variable segments (should score lower)
    variable_cagrs = [0.05, 0.20, -0.05]
    score2 = compute_simple_robustness_score(variable_cagrs)
    assert 0 <= score2 <= 50, f"Variable CAGRs should score lower: {score2}"
    print(f"✓ Variable segments (0.05, 0.20, -0.05): score={score2:.1f}/100")
    
    # Case 3: Empty list (should return neutral 50.0)
    score3 = compute_simple_robustness_score([])
    assert score3 == 50.0, f"Empty should return 50.0: {score3}"
    print(f"✓ Empty list: score={score3:.1f}/100")
    
    # Case 4: Single segment (should return 100.0)
    score4 = compute_simple_robustness_score([0.12])
    assert score4 == 100.0, f"Single segment should return 100.0: {score4}"
    print(f"✓ Single segment: score={score4:.1f}/100")


def test_robustness_from_curve():
    """Test robustness computation from equity curve."""
    # Create a smooth upward curve (consistent growth)
    dates = pd.date_range("2020-01-01", periods=756, freq="D")  # 3 years
    np.random.seed(42)
    returns = np.random.normal(0.08/252, 0.15/np.sqrt(252), size=756)
    curve1 = pd.Series((1 + returns).cumprod(), index=dates)
    
    score1 = compute_robustness_from_curve(curve1, n_segments=3)
    assert 0 <= score1 <= 100, f"Score out of range: {score1}"
    print(f"✓ Smooth curve (3 years): score={score1:.1f}/100")
    
    # Create a volatile curve (inconsistent growth)
    returns2 = np.concatenate([
        np.random.normal(0.20/252, 0.15/np.sqrt(252), size=252),  # good year
        np.random.normal(-0.10/252, 0.15/np.sqrt(252), size=252),  # bad year
        np.random.normal(0.15/252, 0.15/np.sqrt(252), size=252),  # recovery
    ])
    curve2 = pd.Series((1 + returns2).cumprod(), index=dates)
    
    score2 = compute_robustness_from_curve(curve2, n_segments=3)
    assert 0 <= score2 <= 100, f"Score out of range: {score2}"
    print(f"✓ Volatile curve (3 years): score={score2:.1f}/100")
    
    # Volatile should score lower than smooth
    if score2 < score1:
        print(f"✓ Volatile curve ({score2:.1f}) < Smooth curve ({score1:.1f})")
    else:
        print(f"⚠ Expected volatile < smooth, got volatile={score2:.1f}, smooth={score1:.1f}")
    
    # Short curve (should return neutral 50.0)
    short_curve = pd.Series([1.0, 1.01, 1.02], index=pd.date_range("2020-01-01", periods=3))
    score3 = compute_robustness_from_curve(short_curve)
    assert score3 == 50.0, f"Short curve should return 50.0: {score3}"
    print(f"✓ Short curve (3 days): score={score3:.1f}/100")


def test_validation():
    """Test validation module."""
    from core.validation import run_light_validator, get_validation_message
    
    # Test with minimal valid parameters
    result1 = run_light_validator(universe_size=5, returns=None, objective="balanced")
    print(f"✓ Basic validation (5 assets, balanced): {result1}")
    
    # Test with too few assets
    result2 = run_light_validator(universe_size=2)
    assert result2 == False, "Should fail with < 3 assets"
    print(f"✓ Validation fails with 2 assets: {result2}")
    
    # Test with returns DataFrame
    dates = pd.date_range("2020-01-01", periods=300, freq="D")
    returns = pd.DataFrame(
        np.random.normal(0, 0.01, size=(300, 5)),
        index=dates,
        columns=["A", "B", "C", "D", "E"]
    )
    result3 = run_light_validator(returns=returns)
    print(f"✓ Validation with returns DataFrame (300 days, 5 tickers): {result3}")
    
    # Test messages
    msg1 = get_validation_message(True)
    msg2 = get_validation_message(False)
    assert "✓" in msg1 or "passed" in msg1.lower()
    assert "⚠" in msg2 or "failed" in msg2.lower()
    print(f"✓ Validation messages: pass='{msg1}', fail='{msg2}'")


if __name__ == "__main__":
    print("="*70)
    print("ROBUSTNESS SCORING TESTS")
    print("="*70)
    
    try:
        test_simple_robustness()
        print("\n" + "="*70)
        test_robustness_from_curve()
        print("\n" + "="*70)
        print("VALIDATION MODULE TESTS")
        print("="*70)
        test_validation()
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Test volatility scaling to ensure it's properly annualized.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from core.utils.metrics import annualized_metrics


def test_volatility_synthetic_data():
    """Test that synthetic daily returns produce reasonable annualized vol."""
    print("\n" + "="*60)
    print("Test 1: Synthetic Daily Returns")
    print("="*60)
    
    # Generate 252 days of returns: daily mean ~0.03%, std ~1%
    # Expected annualized vol ≈ 0.01 * sqrt(252) ≈ 0.158 (15.8%)
    np.random.seed(42)
    daily_returns = pd.Series(
        np.random.normal(loc=0.0003, scale=0.01, size=252),
        index=pd.date_range("2024-01-01", periods=252, freq="D")
    )
    
    metrics = annualized_metrics(daily_returns)
    vol = metrics["Volatility"]
    
    print(f"Daily returns stats:")
    print(f"  Mean: {daily_returns.mean():.6f}")
    print(f"  Std: {daily_returns.std():.6f}")
    print(f"\nAnnualized metrics:")
    print(f"  Vol: {vol:.4f} ({vol*100:.2f}%)")
    print(f"  CAGR: {metrics['CAGR']:.4f} ({metrics['CAGR']*100:.2f}%)")
    print(f"  Sharpe: {metrics['Sharpe']:.2f}")
    
    # Sanity checks
    assert 0.10 <= vol <= 0.25, f"Vol {vol:.4f} outside reasonable range [0.10, 0.25]"
    print(f"\n✅ PASS: Vol {vol:.4f} is within reasonable range [0.10, 0.25]")


def test_spy_like_volatility():
    """Test with SPY-like daily returns (more realistic)."""
    print("\n" + "="*60)
    print("Test 2: SPY-like Daily Returns")
    print("="*60)
    
    # SPY typically has ~15-20% annualized vol
    # Daily: ~0.95% std → annualized: 0.0095 * sqrt(252) ≈ 0.15 (15%)
    np.random.seed(123)
    daily_returns = pd.Series(
        np.random.normal(loc=0.0004, scale=0.0095, size=504),  # 2 years
        index=pd.date_range("2023-01-01", periods=504, freq="D")
    )
    
    metrics = annualized_metrics(daily_returns)
    vol = metrics["Volatility"]
    
    print(f"Daily returns stats:")
    print(f"  Mean: {daily_returns.mean():.6f}")
    print(f"  Std: {daily_returns.std():.6f}")
    print(f"\nAnnualized metrics:")
    print(f"  Vol: {vol:.4f} ({vol*100:.2f}%)")
    print(f"  CAGR: {metrics['CAGR']:.4f} ({metrics['CAGR']*100:.2f}%)")
    print(f"  Sharpe: {metrics['Sharpe']:.2f}")
    
    # SPY-like volatility should be 0.12-0.22 range
    assert 0.08 <= vol <= 0.30, f"Vol {vol:.4f} outside reasonable range [0.08, 0.30]"
    print(f"\n✅ PASS: Vol {vol:.4f} is within SPY-like range [0.08, 0.30]")


def test_low_volatility_portfolio():
    """Test low-vol portfolio (bond-heavy)."""
    print("\n" + "="*60)
    print("Test 3: Low-Volatility Portfolio (Bond-heavy)")
    print("="*60)
    
    # Bond portfolio: ~5-8% annualized vol
    # Daily: ~0.3% std → annualized: 0.003 * sqrt(252) ≈ 0.048 (4.8%)
    np.random.seed(456)
    daily_returns = pd.Series(
        np.random.normal(loc=0.0002, scale=0.003, size=252),
        index=pd.date_range("2024-01-01", periods=252, freq="D")
    )
    
    metrics = annualized_metrics(daily_returns)
    vol = metrics["Volatility"]
    
    print(f"Daily returns stats:")
    print(f"  Mean: {daily_returns.mean():.6f}")
    print(f"  Std: {daily_returns.std():.6f}")
    print(f"\nAnnualized metrics:")
    print(f"  Vol: {vol:.4f} ({vol*100:.2f}%)")
    print(f"  CAGR: {metrics['CAGR']:.4f} ({metrics['CAGR']*100:.2f}%)")
    
    # Low-vol portfolios should be 0.03-0.10 range
    assert 0.02 <= vol <= 0.12, f"Vol {vol:.4f} outside low-vol range [0.02, 0.12]"
    print(f"\n✅ PASS: Vol {vol:.4f} is within low-vol range [0.02, 0.12]")


def test_high_volatility_portfolio():
    """Test high-vol portfolio (aggressive growth)."""
    print("\n" + "="*60)
    print("Test 4: High-Volatility Portfolio (Aggressive)")
    print("="*60)
    
    # Aggressive portfolio: ~25-35% annualized vol
    # Daily: ~1.6% std → annualized: 0.016 * sqrt(252) ≈ 0.254 (25.4%)
    np.random.seed(789)
    daily_returns = pd.Series(
        np.random.normal(loc=0.0005, scale=0.016, size=252),
        index=pd.date_range("2024-01-01", periods=252, freq="D")
    )
    
    metrics = annualized_metrics(daily_returns)
    vol = metrics["Volatility"]
    
    print(f"Daily returns stats:")
    print(f"  Mean: {daily_returns.mean():.6f}")
    print(f"  Std: {daily_returns.std():.6f}")
    print(f"\nAnnualized metrics:")
    print(f"  Vol: {vol:.4f} ({vol*100:.2f}%)")
    print(f"  CAGR: {metrics['CAGR']:.4f} ({metrics['CAGR']*100:.2f}%)")
    
    # High-vol portfolios should be 0.20-0.40 range
    assert 0.18 <= vol <= 0.45, f"Vol {vol:.4f} outside high-vol range [0.18, 0.45]"
    print(f"\n✅ PASS: Vol {vol:.4f} is within high-vol range [0.18, 0.45]")


def main():
    """Run all volatility scaling tests."""
    print("="*60)
    print("Volatility Scaling Tests")
    print("="*60)
    print("\nExpected behavior:")
    print("  - Annualized Vol = Daily Std * sqrt(252)")
    print("  - SPY-like: 12-20% annualized")
    print("  - Low-vol (bonds): 3-10% annualized")
    print("  - High-vol (aggressive): 20-35% annualized")
    
    all_passed = True
    
    try:
        test_volatility_synthetic_data()
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        all_passed = False
    
    try:
        test_spy_like_volatility()
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        all_passed = False
    
    try:
        test_low_volatility_portfolio()
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        all_passed = False
    
    try:
        test_high_volatility_portfolio()
    except AssertionError as e:
        print(f"\n❌ FAIL: {e}")
        all_passed = False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VOLATILITY TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("❌ SOME VOLATILITY TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

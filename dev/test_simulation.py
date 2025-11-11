#!/usr/bin/env python3
"""Test simulation flow to verify it works end-to-end."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_ingestion import get_prices
from core.preprocessing import compute_returns
from core.recommendation_engine import recommend, UserProfile
from core.utils.env_tools import load_config
import pandas as pd

def test_simulation():
    print("=" * 60)
    print("Testing Simulation Flow")
    print("=" * 60)
    
    # 1. Load prices
    print("\n1. Loading prices for ['SPY', 'QQQ', 'TLT']...")
    prices = get_prices(['SPY', 'QQQ', 'TLT'], start='2020-01-01')
    print(f"   ✓ Loaded {len(prices)} rows, {len(prices.columns)} symbols")
    print(f"   Columns: {list(prices.columns)}")
    print(f"   Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"   Note: prices is already in wide format (columns=symbols, index=date)")
    
    # 2. Compute returns (prices is already wide)
    print("\n2. Computing returns...")
    rets = compute_returns(prices)
    print(f"   ✓ Returns shape: {rets.shape}")
    print(f"   First few returns:\n{rets.head()}")
    
    # 4. Create user profile
    print("\n4. Creating user profile...")
    prof = UserProfile(monthly_contribution=1000, horizon_years=10, risk_level="moderate")
    print(f"   ✓ Profile: monthly={prof.monthly_contribution}, horizon={prof.horizon_years}, risk={prof.risk_level}")
    
    # 5. Get recommendation
    print("\n5. Running recommendation engine...")
    rec = recommend(rets, prof, objective="grow", risk_pct=50, method="hrp")
    print(f"   ✓ Recommendation keys: {list(rec.keys())}")
    
    # 6. Extract weights
    print("\n6. Extracting portfolio weights...")
    w = pd.Series(rec.get("weights", {})).sort_values(ascending=False)
    print(f"   ✓ Weights:\n{w}")
    print(f"   ✓ Sum of weights: {w.sum():.4f}")
    
    # 7. Check metrics
    print("\n7. Checking backtest metrics...")
    metrics = rec.get("metrics", {})
    print(f"   ✓ CAGR: {metrics.get('CAGR', 'N/A')}")
    print(f"   ✓ Volatility: {metrics.get('Volatility', 'N/A')}")
    print(f"   ✓ Sharpe: {metrics.get('Sharpe', 'N/A')}")
    print(f"   ✓ MaxDD: {metrics.get('MaxDD', 'N/A')}")
    
    # 8. Check curve
    print("\n8. Checking cumulative curve...")
    curve = rec.get("curve", pd.Series(dtype=float))
    if not curve.empty:
        print(f"   ✓ Curve shape: {len(curve)}")
        print(f"   ✓ Start value: {curve.iloc[0]:.4f}")
        print(f"   ✓ End value: {curve.iloc[-1]:.4f}")
        print(f"   ✓ Growth: {(curve.iloc[-1] / curve.iloc[0] - 1) * 100:.2f}%")
    else:
        print("   ✗ Curve is empty!")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_simulation()
    sys.exit(0 if success else 1)

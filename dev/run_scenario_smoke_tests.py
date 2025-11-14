#!/usr/bin/env python3
"""
Smoke tests for risk-based portfolio selection scenarios.
Tests Conservative, Moderate, Aggressive scenarios plus edge cases.
Exit code 0 if all pass, non-zero otherwise.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from typing import List, Dict, Any

def test_scenario(
    scenario_name: str,
    risk_score: float,
    candidates: List[Dict[str, Any]],
    expected_min_candidates: int = 0,
) -> bool:
    """
    Test a single scenario.
    
    Args:
        scenario_name: Name for logging
        risk_score: Risk score (0-100)
        candidates: List of candidate portfolios
        expected_min_candidates: Minimum expected filtered candidates
    
    Returns:
        True if passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing: {scenario_name}")
    print(f"Risk score: {risk_score:.1f}")
    print(f"Input candidates: {len(candidates)}")
    
    try:
        from core.recommendation_engine import select_candidates_for_risk_score, pick_portfolio_from_slider
        
        # Filter candidates
        filtered = select_candidates_for_risk_score(candidates, risk_score)
        print(f"Filtered candidates: {len(filtered)}")
        
        if len(filtered) < expected_min_candidates:
            print(f"❌ FAIL: Expected at least {expected_min_candidates} candidates, got {len(filtered)}")
            return False
        
        if not filtered:
            print(f"✓ PASS: No candidates (expected for edge cases)")
            return True
        
        # Show filtered summary
        for fc in filtered[:3]:  # Show first 3
            name = fc.get("name", "Unknown")
            mu = fc.get("metrics", {}).get("CAGR", 0) * 100
            sig = fc.get("metrics", {}).get("Vol", 0) * 100
            print(f"  - {name}: CAGR={mu:.1f}%, Vol={sig:.1f}%")
        
        # Pick from slider positions
        for slider_val in [0.0, 0.5, 1.0]:
            picked = pick_portfolio_from_slider(filtered, slider_val)
            if not picked:
                print(f"❌ FAIL: pick_portfolio_from_slider returned None for slider={slider_val}")
                return False
            
            name = picked.get("name", "Unknown")
            mu = picked.get("metrics", {}).get("CAGR", 0) * 100
            weights = picked.get("weights", {})
            print(f"  Slider {slider_val:.1f} → {name} (CAGR={mu:.1f}%, {len(weights)} assets)")
        
        print(f"✓ PASS: {scenario_name}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Exception in {scenario_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_mock_candidates(n: int = 20) -> List[Dict[str, Any]]:
    """Create mock candidate portfolios with varying risk/return profiles."""
    np.random.seed(42)
    candidates = []
    
    for i in range(n):
        # Generate realistic CAGR (5% to 15%) and Vol (8% to 25%)
        cagr = 0.05 + (i / n) * 0.10 + np.random.uniform(-0.01, 0.01)
        vol = 0.08 + (i / n) * 0.17 + np.random.uniform(-0.02, 0.02)
        sharpe = cagr / vol if vol > 0 else 0
        
        # Generate mock weights
        n_assets = np.random.randint(3, 8)
        raw_weights = np.random.dirichlet(np.ones(n_assets))
        weights = {f"Asset_{j}": float(w) for j, w in enumerate(raw_weights)}
        
        candidates.append({
            "name": f"Portfolio_{i+1}",
            "weights": weights,
            "metrics": {
                "CAGR": float(cagr),
                "Vol": float(vol),
                "Volatility": float(vol),  # Alias
                "Sharpe": float(sharpe),
                "MaxDD": float(-0.10 - i * 0.01),
            },
            "notes": f"Mock portfolio {i+1}",
            "optimizer": "hrp",
        })
    
    return candidates


def main():
    """Run all smoke tests."""
    print("="*60)
    print("Risk-Based Portfolio Selection - Smoke Tests")
    print("="*60)
    
    # Create mock candidates
    candidates = create_mock_candidates(25)
    
    all_passed = True
    
    # Test 1: Conservative (risk_score ~ 30)
    all_passed &= test_scenario(
        "Conservative (risk_score=30)",
        risk_score=30.0,
        candidates=candidates,
        expected_min_candidates=1,
    )
    
    # Test 2: Moderate (risk_score ~ 50)
    all_passed &= test_scenario(
        "Moderate (risk_score=50)",
        risk_score=50.0,
        candidates=candidates,
        expected_min_candidates=1,
    )
    
    # Test 3: Aggressive (risk_score ~ 80)
    all_passed &= test_scenario(
        "Aggressive (risk_score=80)",
        risk_score=80.0,
        candidates=candidates,
        expected_min_candidates=1,
    )
    
    # Edge case 1: Very low risk score
    all_passed &= test_scenario(
        "Edge: Very low risk (risk_score=10)",
        risk_score=10.0,
        candidates=candidates,
        expected_min_candidates=0,  # May filter out everything
    )
    
    # Edge case 2: Very high risk score
    all_passed &= test_scenario(
        "Edge: Very high risk (risk_score=95)",
        risk_score=95.0,
        candidates=candidates,
        expected_min_candidates=0,  # May filter out everything
    )
    
    # Edge case 3: Identical volatilities (narrow spread)
    narrow_candidates = []
    for i in range(5):
        narrow_candidates.append({
            "name": f"Narrow_{i+1}",
            "weights": {"Asset_A": 0.6, "Asset_B": 0.4},
            "metrics": {
                "CAGR": 0.08 + i * 0.005,
                "Vol": 0.12,  # All same vol
                "Volatility": 0.12,
                "Sharpe": (0.08 + i * 0.005) / 0.12,
            },
        })
    
    all_passed &= test_scenario(
        "Edge: Identical volatilities",
        risk_score=50.0,
        candidates=narrow_candidates,
        expected_min_candidates=0,
    )
    
    # Edge case 4: Small universe (only 3 candidates)
    small_candidates = candidates[:3]
    all_passed &= test_scenario(
        "Edge: Small universe (3 candidates)",
        risk_score=50.0,
        candidates=small_candidates,
        expected_min_candidates=0,
    )
    
    # Edge case 5: Large spread (very different risks)
    large_spread = [
        {
            "name": "Low_Risk",
            "weights": {"Asset_A": 1.0},
            "metrics": {"CAGR": 0.05, "Vol": 0.05, "Volatility": 0.05, "Sharpe": 1.0},
        },
        {
            "name": "High_Risk",
            "weights": {"Asset_B": 1.0},
            "metrics": {"CAGR": 0.15, "Vol": 0.30, "Volatility": 0.30, "Sharpe": 0.5},
        },
    ]
    all_passed &= test_scenario(
        "Edge: Large spread (low vs high risk)",
        risk_score=50.0,
        candidates=large_spread,
        expected_min_candidates=0,
    )
    
    # Edge case 6: Empty candidates
    all_passed &= test_scenario(
        "Edge: Empty candidates list",
        risk_score=50.0,
        candidates=[],
        expected_min_candidates=0,
    )
    
    # Final summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

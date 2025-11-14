#!/usr/bin/env python3
"""
Debug script to inspect risk-based candidate filtering behavior.

Usage:
    .venv/bin/python dev/debug_risk_filtering.py
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.recommendation_engine import (
    select_candidates_for_risk_score,
    pick_portfolio_from_slider,
    generate_candidates,
    DEFAULT_OBJECTIVES,
)
from core.data_ingestion import get_prices
from core.portfolio_engine import clean_prices_to_returns


def create_synthetic_candidates():
    """Create synthetic candidates for testing if real data unavailable."""
    return [
        {
            "name": "Conservative",
            "weights": {"BIL": 0.6, "BND": 0.4},
            "metrics": {"CAGR": 0.03, "Vol": 0.05, "Sharpe": 0.6, "MaxDD": -0.02},
            "optimizer": "min_var",
            "sat_cap": 0.2,
            "notes": "Low risk",
            "shortlist": False,
        },
        {
            "name": "Income",
            "weights": {"BND": 0.5, "TLT": 0.3, "SPY": 0.2},
            "metrics": {"CAGR": 0.055, "Vol": 0.08, "Sharpe": 0.69, "MaxDD": -0.08},
            "optimizer": "risk_parity",
            "sat_cap": 0.2,
            "notes": "Moderate risk",
            "shortlist": False,
        },
        {
            "name": "Balanced",
            "weights": {"SPY": 0.4, "BND": 0.3, "TLT": 0.2, "GLD": 0.1},
            "metrics": {"CAGR": 0.075, "Vol": 0.11, "Sharpe": 0.68, "MaxDD": -0.12},
            "optimizer": "hrp",
            "sat_cap": 0.25,
            "notes": "Balanced",
            "shortlist": False,
        },
        {
            "name": "Growth",
            "weights": {"SPY": 0.5, "QQQ": 0.3, "VTI": 0.2},
            "metrics": {"CAGR": 0.10, "Vol": 0.15, "Sharpe": 0.67, "MaxDD": -0.18},
            "optimizer": "max_sharpe",
            "sat_cap": 0.3,
            "notes": "Growth focus",
            "shortlist": False,
        },
        {
            "name": "Aggressive",
            "weights": {"QQQ": 0.6, "AAPL": 0.2, "MSFT": 0.2},
            "metrics": {"CAGR": 0.12, "Vol": 0.20, "Sharpe": 0.60, "MaxDD": -0.25},
            "optimizer": "max_sharpe",
            "sat_cap": 0.35,
            "notes": "High risk",
            "shortlist": False,
        },
        {
            "name": "Ultra Aggressive",
            "weights": {"TSLA": 0.4, "NVDA": 0.3, "AAPL": 0.3},
            "metrics": {"CAGR": 0.15, "Vol": 0.28, "Sharpe": 0.54, "MaxDD": -0.35},
            "optimizer": "equal_weight",
            "sat_cap": 0.35,
            "notes": "Very high risk",
            "shortlist": False,
        },
    ]


def load_real_candidates():
    """Attempt to load real candidates from the balanced objective."""
    try:
        print("Attempting to load real candidates from balanced objective...")
        # Try to generate candidates for balanced objective
        tickers = ["SPY", "TLT", "GLD", "QQQ", "VTI"]
        prices = get_prices(tickers, start="2020-01-01")
        
        if prices.empty:
            print("  -> No price data available, using synthetic candidates.")
            return None
        
        returns = clean_prices_to_returns(prices, winsor_p=0.005)
        if returns.empty:
            print("  -> Empty returns, using synthetic candidates.")
            return None
        
        obj_cfg = DEFAULT_OBJECTIVES.get("balanced")
        if not obj_cfg:
            print("  -> No balanced objective config, using synthetic candidates.")
            return None
        
        candidates = generate_candidates(
            returns=returns,
            objective_cfg=obj_cfg,
            n_candidates=8,
            seed=42
        )
        
        if len(candidates) >= 5:
            print(f"  -> Successfully loaded {len(candidates)} real candidates.")
            return candidates
        else:
            print(f"  -> Only {len(candidates)} candidates generated, using synthetic.")
            return None
    
    except Exception as e:
        print(f"  -> Error loading real candidates: {e}")
        print("  -> Falling back to synthetic candidates.")
        return None


def main():
    print("="*70)
    print("Risk-Based Candidate Filtering Debug Script")
    print("="*70)
    print()
    
    # Try to load real candidates, fall back to synthetic
    candidates = load_real_candidates()
    if candidates is None:
        candidates = create_synthetic_candidates()
        print("Using synthetic candidates for demonstration.\n")
    
    print(f"Total candidates: {len(candidates)}")
    print("\nCandidate Summary:")
    for i, c in enumerate(candidates, 1):
        mu = c["metrics"].get("CAGR", 0.0)
        sigma = c["metrics"].get("Vol") or c["metrics"].get("Volatility", 0.0)
        print(f"  {i}. {c['name']:<20} mu={mu:>6.2%}  sigma={sigma:>6.2%}")
    
    print("\n" + "="*70)
    print("Testing Risk Score Filtering")
    print("="*70)
    
    # Test different risk scores
    risk_scores = [20, 50, 80]
    
    for risk_score in risk_scores:
        print(f"\n{'─'*70}")
        print(f"Risk Score: {risk_score}")
        print(f"{'─'*70}")
        
        # Compute target sigma
        sigma_min = 0.05
        sigma_max = 0.20
        band = 0.02
        target_sigma = sigma_min + (sigma_max - sigma_min) * (risk_score / 100.0)
        sigma_low = target_sigma - band
        sigma_high = target_sigma + band
        
        print(f"  Target σ: {target_sigma:.4f}")
        print(f"  Range:    [{sigma_low:.4f}, {sigma_high:.4f}]")
        
        # Filter candidates
        filtered = select_candidates_for_risk_score(
            candidates,
            risk_score=risk_score,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            band=band
        )
        
        print(f"\n  Filtered candidates: {len(filtered)}")
        for i, c in enumerate(filtered, 1):
            mu = c["metrics"].get("CAGR", 0.0)
            sigma = c["metrics"].get("Vol") or c["metrics"].get("Volatility", 0.0)
            print(f"    {i}. {c['name']:<20} mu={mu:>6.2%}  sigma={sigma:>6.2%}")
        
        if not filtered:
            print("    (No candidates in this band)")
            continue
        
        # Test slider positions
        print("\n  Slider Picks:")
        for slider_val in [0.0, 0.5, 1.0]:
            chosen = pick_portfolio_from_slider(filtered, slider_val)
            if chosen:
                mu = chosen["metrics"].get("CAGR", 0.0)
                sigma = chosen["metrics"].get("Vol") or chosen["metrics"].get("Volatility", 0.0)
                print(f"    slider={slider_val:.1f} → {chosen['name']:<20} mu={mu:>6.2%}  sigma={sigma:>6.2%}")
    
    print("\n" + "="*70)
    print("Summary: Sigma Distribution of All Candidates")
    print("="*70)
    sigmas = sorted([c["metrics"].get("Vol") or c["metrics"].get("Volatility", 0.0) for c in candidates])
    print(f"  Min σ:  {min(sigmas):.4f}")
    print(f"  Max σ:  {max(sigmas):.4f}")
    print(f"  Mean σ: {sum(sigmas)/len(sigmas):.4f}")
    if len(sigmas) >= 2:
        import statistics
        print(f"  Median σ: {statistics.median(sigmas):.4f}")
    print()


if __name__ == "__main__":
    main()

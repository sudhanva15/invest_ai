#!/usr/bin/env python3
"""Analyze candidate volatility distribution to calibrate risk score mapping.

This script generates candidates across multiple objectives and analyzes
the distribution of their annualized volatility (sigma). Use the results
to set realistic sigma_min and sigma_max bounds for risk-based filtering.

Usage:
    python3 dev/debug_sigma_distribution.py
    python3 dev/debug_sigma_distribution.py --objectives balanced,growth,income
    python3 dev/debug_sigma_distribution.py --n-candidates 20
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.data_ingestion import get_prices
from core.portfolio_engine import clean_prices_to_returns
from core.recommendation_engine import DEFAULT_OBJECTIVES, generate_candidates, ObjectiveConfig
from core.utils.metrics import annualized_metrics


def collect_sigma_distribution(
    objectives: List[str],
    n_candidates: int = 12,
    tickers: List[str] | None = None,
) -> Dict[str, Any]:
    """Generate candidates and collect their sigma (volatility) values.
    
    Args:
        objectives: List of objective names to test.
        n_candidates: Number of candidates per objective.
        tickers: Optional list of tickers. If None, uses default universe.
    
    Returns:
        Dict with sigma statistics and raw values.
    """
    print(f"Generating {n_candidates} candidates for {len(objectives)} objectives...")
    
    # Get default universe if not provided
    if tickers is None:
        try:
            import json
            catalog_path = ROOT / "config" / "assets_catalog.json"
            catalog = json.loads(catalog_path.read_text())
            tickers = [
                k for k, v in catalog.items()
                if v.get("eligible_now") and v.get("tier", 99) <= 2
            ]
            print(f"Using {len(tickers)} eligible tickers from catalog")
        except Exception as e:
            print(f"Warning: Could not load catalog, using fallback universe: {e}", file=sys.stderr)
            tickers = ["SPY", "TLT", "GLD", "VTI", "BND", "QQQ", "IWM", "EFA", "EEM", "VNQ"]
    
    # Get prices and clean to returns
    print(f"Fetching prices for {len(tickers)} tickers...")
    prices = get_prices(tickers, start="1900-01-01")
    
    print("Cleaning prices to returns...")
    returns, diags = clean_prices_to_returns(
        prices,
        winsor_p=0.005,
        min_non_na=126,
        k_days=1260,
        strict=False,
        return_diagnostics=True,
    )
    
    if diags.get("dropped_symbols"):
        print(f"Dropped {len(diags['dropped_symbols'])} symbols during cleaning")
    
    print(f"Final returns shape: {returns.shape}")
    
    # Collect sigma values from all candidates
    all_sigmas = []
    candidate_details = []
    
    for obj_name in objectives:
        print(f"\nGenerating candidates for objective: {obj_name}")
        
        obj_cfg = DEFAULT_OBJECTIVES.get(obj_name)
        if obj_cfg is None:
            print(f"  Warning: Objective {obj_name} not found, using default")
            obj_cfg = ObjectiveConfig(name=obj_name)
        elif isinstance(obj_cfg, dict):
            obj_cfg = ObjectiveConfig(**obj_cfg)
        
        try:
            # Import catalog for constraint enforcement
            try:
                import json
                catalog = json.loads((ROOT / "config" / "assets_catalog.json").read_text())
            except Exception:
                catalog = {}
            
            candidates = generate_candidates(
                returns=returns,
                objective_cfg=obj_cfg,
                catalog=catalog,
                n_candidates=n_candidates,
            )
            
            print(f"  Generated {len(candidates)} candidates")
            
            # Extract sigma from each candidate
            for cand in candidates:
                metrics = cand.get("metrics", {})
                sigma = metrics.get("Vol") or metrics.get("Volatility")
                
                if sigma is not None and pd.notna(sigma):
                    all_sigmas.append(sigma)
                    candidate_details.append({
                        "objective": obj_name,
                        "name": cand.get("name", "unknown"),
                        "sigma": sigma,
                        "cagr": metrics.get("CAGR"),
                        "sharpe": metrics.get("Sharpe"),
                    })
        
        except Exception as e:
            print(f"  Error generating candidates for {obj_name}: {e}", file=sys.stderr)
    
    if not all_sigmas:
        print("\nERROR: No valid sigma values collected!", file=sys.stderr)
        return {}
    
    # Compute statistics
    sigmas_array = np.array(all_sigmas)
    
    stats = {
        "count": len(all_sigmas),
        "min": float(np.min(sigmas_array)),
        "max": float(np.max(sigmas_array)),
        "mean": float(np.mean(sigmas_array)),
        "median": float(np.median(sigmas_array)),
        "std": float(np.std(sigmas_array)),
        "p05": float(np.percentile(sigmas_array, 5)),
        "p25": float(np.percentile(sigmas_array, 25)),
        "p75": float(np.percentile(sigmas_array, 75)),
        "p95": float(np.percentile(sigmas_array, 95)),
        "raw_values": all_sigmas,
        "candidate_details": candidate_details,
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Analyze candidate sigma distribution")
    parser.add_argument(
        "--objectives",
        type=str,
        default="balanced,growth,income",
        help="Comma-separated objectives (default: balanced,growth,income)",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=12,
        help="Number of candidates per objective (default: 12)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (default: use catalog)",
    )
    
    args = parser.parse_args()
    
    objectives = [o.strip() for o in args.objectives.split(",")]
    tickers = [t.strip() for t in args.tickers.split(",")] if args.tickers else None
    
    stats = collect_sigma_distribution(objectives, args.n_candidates, tickers)
    
    if not stats:
        sys.exit(1)
    
    # Print results
    print("\n" + "="*70)
    print("SIGMA (VOLATILITY) DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nTotal candidates analyzed: {stats['count']}")
    print(f"\nSigma statistics (annualized volatility):")
    print(f"  Min:     {stats['min']:.4f} ({stats['min']*100:.2f}%)")
    print(f"  5th %:   {stats['p05']:.4f} ({stats['p05']*100:.2f}%)")
    print(f"  25th %:  {stats['p25']:.4f} ({stats['p25']*100:.2f}%)")
    print(f"  Median:  {stats['median']:.4f} ({stats['median']*100:.2f}%)")
    print(f"  Mean:    {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
    print(f"  75th %:  {stats['p75']:.4f} ({stats['p75']*100:.2f}%)")
    print(f"  95th %:  {stats['p95']:.4f} ({stats['p95']*100:.2f}%)")
    print(f"  Max:     {stats['max']:.4f} ({stats['max']*100:.2f}%)")
    print(f"  Std Dev: {stats['std']:.4f}")
    
    print("\n" + "="*70)
    print("RECOMMENDED SIGMA BOUNDS FOR RISK FILTERING")
    print("="*70)
    print(f"\nBased on observed distribution, recommend:")
    print(f"  sigma_min = {stats['p05']:.4f}  # 5th percentile (conservative lower bound)")
    print(f"  sigma_max = {stats['p95']:.4f}  # 95th percentile (captures most candidates)")
    print(f"\nAlternatively, for a wider range:")
    print(f"  sigma_min = {stats['min']:.4f}  # Absolute minimum")
    print(f"  sigma_max = {stats['max']:.4f}  # Absolute maximum")
    
    print("\n" + "="*70)
    print("SAMPLE CANDIDATES BY SIGMA")
    print("="*70)
    
    # Sort candidates by sigma and show a few examples
    sorted_cands = sorted(stats['candidate_details'], key=lambda c: c['sigma'])
    
    print("\nLowest sigma (conservative):")
    for cand in sorted_cands[:3]:
        print(f"  {cand['name']:40s} sigma={cand['sigma']:.4f} cagr={cand['cagr']:.4f} sharpe={cand['sharpe']:.3f}")
    
    print("\nHighest sigma (aggressive):")
    for cand in sorted_cands[-3:]:
        print(f"  {cand['name']:40s} sigma={cand['sigma']:.4f} cagr={cand['cagr']:.4f} sharpe={cand['sharpe']:.3f}")
    
    print("\n" + "="*70)
    print("USAGE")
    print("="*70)
    print("\nUpdate core/recommendation_engine.py or core/risk_profile.py:")
    print(f"""
# Calibrated from observed candidate distribution ({stats['count']} samples)
# See dev/debug_sigma_distribution.py for analysis
SIGMA_MIN = {stats['p05']:.4f}  # 5th percentile
SIGMA_MAX = {stats['p95']:.4f}  # 95th percentile
""")


if __name__ == "__main__":
    main()

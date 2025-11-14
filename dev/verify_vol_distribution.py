#!/usr/bin/env python3
"""
Quick check: print Vol distribution for a sample of generated candidates.
Confirms that Vol values lie in the expected range [~0.08, ~0.30].
"""
import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import json
from pathlib import Path
from core.data_ingestion import get_prices
from core.preprocessing import compute_returns
from core.recommendation_engine import generate_candidates, DEFAULT_OBJECTIVES

def main():
    print("=" * 60)
    print("Volatility Distribution Check")
    print("=" * 60)
    print()

    # Get data for a small universe
    symbols = ["SPY", "QQQ", "BND", "TLT", "GLD", "VNQ"]
    print(f"Loading price data for: {', '.join(symbols)}")
    
    try:
        prices = get_prices(symbols, start="2015-01-01", end="2023-12-31")
        returns = compute_returns(prices)
        print(f"✓ Loaded {len(prices)} days of data")
        print()
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        return False

    # Load catalog
    catalog_path = Path(__file__).parent.parent / "config" / "assets_catalog.json"
    with open(catalog_path) as f:
        catalog = json.load(f)
    
    # Generate candidates
    print("Generating candidates...")
    obj_cfg = DEFAULT_OBJECTIVES["growth"]
    
    try:
        candidates = generate_candidates(
            returns=returns,
            objective_cfg=obj_cfg,
            catalog=catalog,
            n_candidates=20
        )
        print(f"✓ Generated {len(candidates)} candidates")
        print()
    except Exception as e:
        print(f"✗ Failed to generate candidates: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Extract Volatility values
    vols = [c["metrics"]["Volatility"] for c in candidates]
    
    # Statistics
    print("Vol Distribution:")
    print(f"  Min:    {min(vols):.4f} ({min(vols)*100:.2f}%)")
    print(f"  25th:   {pd.Series(vols).quantile(0.25):.4f} ({pd.Series(vols).quantile(0.25)*100:.2f}%)")
    print(f"  Median: {pd.Series(vols).median():.4f} ({pd.Series(vols).median()*100:.2f}%)")
    print(f"  75th:   {pd.Series(vols).quantile(0.75):.4f} ({pd.Series(vols).quantile(0.75)*100:.2f}%)")
    print(f"  Max:    {max(vols):.4f} ({max(vols)*100:.2f}%)")
    print()
    
    # Check against expected bounds
    SIGMA_MIN = 0.1271  # from config
    SIGMA_MAX = 0.2202  # from config
    EXPECTED_MIN = 0.08  # reasonable lower bound
    EXPECTED_MAX = 0.30  # reasonable upper bound
    
    print("Compatibility Checks:")
    print(f"  Config sigma_min: {SIGMA_MIN:.4f} ({SIGMA_MIN*100:.2f}%)")
    print(f"  Config sigma_max: {SIGMA_MAX:.4f} ({SIGMA_MAX*100:.2f}%)")
    print(f"  Expected range: [{EXPECTED_MIN:.4f}, {EXPECTED_MAX:.4f}]")
    print()
    
    # Validation
    in_range = all(EXPECTED_MIN <= v <= EXPECTED_MAX for v in vols)
    overlaps_config = any(SIGMA_MIN <= v <= SIGMA_MAX for v in vols)
    
    if in_range:
        print("✓ All Vol values in expected range [0.08, 0.30]")
    else:
        outliers = [v for v in vols if v < EXPECTED_MIN or v > EXPECTED_MAX]
        print(f"✗ Found {len(outliers)} outliers: {outliers}")
    
    if overlaps_config:
        overlap_count = sum(1 for v in vols if SIGMA_MIN <= v <= SIGMA_MAX)
        print(f"✓ {overlap_count}/{len(vols)} candidates overlap config bounds")
    else:
        print("⚠ No candidates overlap config sigma_min/max bounds")
    
    print()
    print("=" * 60)
    print("✅ Vol distribution check complete")
    print("=" * 60)
    
    return in_range

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

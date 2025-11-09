"""Snapshot current portfolio weights for stability tracking."""

import sys
import argparse
from pathlib import Path
import json

# Add repo root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from core.data_ingestion import get_prices
from core.preprocessing import to_wide, compute_returns
from core.recommendation_engine import recommend, UserProfile

def snapshot_weights(symbols=None, verbose=False):
    """Generate a sorted, rounded weight snapshot."""
    if symbols is None:
        symbols = ["SPY", "QQQ", "TLT", "IEF", "GLD"]
        
    # Get prices and compute returns
    prices = get_prices(symbols)
    if prices is None or prices.empty:
        if verbose:
            print("⚠ No price data returned", file=sys.stderr)
        raise ValueError("No price data returned")
    
    # Accept both tidy and wide inputs
    if set(["date","ticker","price"]).issubset(set(map(str.lower, map(str, prices.columns)))):
        wide = to_wide(prices)
    else:
        wide = prices
    rets = compute_returns(wide)
    
    # Trim to common start (max first non-NaN) and drop columns with >20% NaNs
    if rets is not None and not rets.empty:
        rets = rets.sort_index()
        first_valid = {}
        for c in rets.columns:
            s = rets[c].dropna()
            if not s.empty:
                first_valid[c] = s.index[0]
        if first_valid:
            common_start = max(first_valid.values())
            rets = rets[rets.index >= common_start]
        # Drop very sparse columns
        sparse = [c for c in rets.columns if rets[c].isna().mean() > 0.20]
        if sparse and verbose:
            print(f"⚠ Dropping sparse columns (>20% NaN): {sparse}", file=sys.stderr)
        if sparse:
            rets = rets.drop(columns=sparse)
        # Drop any remaining rows with NaNs
        rets = rets.dropna(how="any")
        if verbose:
            print(f"ℹ Returns shape after cleaning: {rets.shape}", file=sys.stderr)
    
    # Run HRP with minimal options
    profile = UserProfile(monthly_contribution=1000, horizon_years=10, risk_level="moderate")
    rec = recommend(
        returns=rets,
        profile=profile,
        objective="grow",
        risk_pct=50,
        method="hrp"
    )
    
    # Round to 4 decimals and sort by weight
    weights = rec.get("weights", {}) or {}
    if not weights:
        if verbose:
            print("⚠ Optimizer returned empty weights; using equal-weight fallback", file=sys.stderr)
        cols = list(rets.columns) if rets is not None and not rets.empty else symbols
        n = len(cols) or len(symbols)
        weights = {cols[i]: 1.0 / n for i in range(n)}
    # Sanitize (no negatives, renormalize)
    # Expand to include all requested symbols/columns for stable diffing
    universe = list(rets.columns) if rets is not None and not rets.empty else symbols
    ser = pd.Series(weights)
    ser = ser.reindex(universe, fill_value=0.0).fillna(0.0).clip(lower=0.0)
    tot = ser.sum()
    if tot > 0:
        ser = ser / tot
    else:
        # Emergency equal-weight fallback if sum is zero
        if verbose:
            print("⚠ Weight sum is zero; applying equal-weight fallback", file=sys.stderr)
        n = len(universe)
        ser = pd.Series({universe[i]: 1.0/n for i in range(n)})
    # Sort by key for stable output (alphabetical)
    snap = {k: round(float(ser.get(k, 0.0)), 4) for k in sorted(universe)}
    if verbose:
        print(f"✓ Final weights (sum={sum(snap.values()):.6f}): {list(snap.keys())}", file=sys.stderr)
    return snap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snapshot portfolio weights for stability tracking")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Write snapshot to tests/fixtures/weights_baseline.json")
    parser.add_argument("--verbose", action="store_true",
                        help="Print diagnostics to stderr")
    args = parser.parse_args()

    try:
        # Generate current snapshot
        weights = snapshot_weights(verbose=args.verbose)
        
        # Print to stdout ONLY (sorted keys for stability)
        print(json.dumps(weights, sort_keys=True))
        
        # Optionally update baseline
        if args.update_baseline:
            baseline_path = ROOT / "tests" / "fixtures" / "weights_baseline.json"
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            with baseline_path.open("w") as f:
                json.dump(weights, f, indent=2, sort_keys=True)
            print(f"✓ Updated baseline at: {baseline_path}", file=sys.stderr)
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
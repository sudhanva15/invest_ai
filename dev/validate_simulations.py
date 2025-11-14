#!/usr/bin/env python3
"""V3 Portfolio Simulation Validator

Non-Streamlit validator that performs comprehensive checks on portfolio simulation
components to ensure V3 spec compliance. Validates data quality, returns cleaning,
macro freshness, candidate generation, metrics plausibility, and receipts integrity.

Usage:
    python3 dev/validate_simulations.py --objective balanced --n-candidates 6
    python3 dev/validate_simulations.py --objective growth --n-candidates 8 --json
    python3 dev/validate_simulations.py --tickers SPY,TLT,GLD --start 2020-01-01 --verbose

Exit codes:
    0: All checks passed
    1: One or more checks failed
"""

from __future__ import annotations
import sys
import os
import argparse
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Import V3 components
try:
    from core.data_ingestion import get_prices_with_provenance
    from core.portfolio_engine import clean_prices_to_returns
    from core.recommendation_engine import DEFAULT_OBJECTIVES, generate_candidates
    from core.utils.metrics import annualized_metrics, align_returns_matrix, assert_metrics_consistency
    from core.utils.receipts import build_receipts
    from core.data_sources.fred import load_series
except ImportError as e:
    print(f"ERROR: Failed to import V3 components: {e}", file=sys.stderr)
    sys.exit(1)

# Print build version at startup
try:
    from core.utils.version import get_build_version
    print(f"[build] {get_build_version()}", file=sys.stderr)
except Exception:
    pass


class ValidationResult:
    """Container for validation check results."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = True
        self.messages: List[str] = []
        self.warnings: List[str] = []
    
    def fail(self, message: str):
        """Mark check as failed with error message."""
        self.passed = False
        self.messages.append(f"✗ {message}")
    
    def warn(self, message: str):
        """Add warning (doesn't fail check)."""
        self.warnings.append(f"⚠ {message}")
    
    def info(self, message: str):
        """Add info message."""
        self.messages.append(f"  {message}")
    
    def succeed(self, message: str):
        """Mark explicit success."""
        self.messages.append(f"✓ {message}")


def check_data(
    tickers: List[str], 
    prices: pd.DataFrame, 
    provenance: Dict[str, Any],
    verbose: bool = False
) -> ValidationResult:
    """Check data quality: presence, NaN ratio, span, provenance.
    
    Criteria:
        - All tickers present in prices DataFrame
        - NaN ratio < 5% per ticker
        - Data span > 3 years
        - Provenance information available
    """
    result = ValidationResult("Data Quality")
    
    # Check all tickers present
    missing = set(tickers) - set(prices.columns)
    if missing:
        result.fail(f"Missing tickers: {missing}")
    else:
        result.succeed(f"All {len(tickers)} tickers present")
    
    # Check NaN ratio per ticker
    nan_fails = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        nan_ratio = prices[ticker].isna().sum() / len(prices)
        if nan_ratio > 0.05:
            nan_fails.append(f"{ticker}={nan_ratio:.1%}")
        if verbose:
            result.info(f"{ticker}: NaN ratio={nan_ratio:.1%}")
    
    if nan_fails:
        result.fail(f"Excessive NaN ratios: {', '.join(nan_fails)}")
    else:
        result.succeed("NaN ratios < 5% for all tickers")
    
    # Check data span
    if not prices.empty:
        span_years = (prices.index[-1] - prices.index[0]).days / 365.25
        if span_years < 3.0:
            result.fail(f"Data span too short: {span_years:.1f}y < 3y")
        else:
            result.succeed(f"Data span: {span_years:.1f} years")
    else:
        result.fail("Empty price DataFrame")
    
    # Check provenance
    if not provenance:
        result.warn("No provenance information available")
    else:
        result.succeed(f"Provenance available for {len(provenance)} tickers")
        if verbose:
            for ticker, provider in provenance.items():
                result.info(f"{ticker}: {provider}")
    
    return result


def check_returns(
    returns: pd.DataFrame,
    verbose: bool = False
) -> ValidationResult:
    """Check returns quality: no NaN/inf, plausible daily magnitudes.
    
    Criteria:
        - No NaN values in cleaned returns
        - No infinite values
        - Mean absolute daily return < 10% (sanity check)
    """
    result = ValidationResult("Returns Quality")
    
    if returns.empty:
        result.fail("Empty returns DataFrame")
        return result
    
    # Check for NaN
    nan_count = returns.isna().sum().sum()
    if nan_count > 0:
        result.fail(f"Found {nan_count} NaN values in returns")
    else:
        result.succeed("No NaN values in returns")
    
    # Check for infinities
    inf_count = np.isinf(returns).sum().sum()
    if inf_count > 0:
        result.fail(f"Found {inf_count} infinite values in returns")
    else:
        result.succeed("No infinite values in returns")
    
    # Check mean absolute daily returns (sanity)
    mean_abs = returns.abs().mean().mean()
    if mean_abs > 0.10:  # 10% average daily move is pathological
        result.fail(f"Mean absolute daily return too high: {mean_abs:.1%}")
    else:
        result.succeed(f"Mean absolute daily return: {mean_abs:.2%}")
    
    if verbose:
        result.info(f"Returns shape: {returns.shape}")
        result.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")
    
    return result


def check_macro(verbose: bool = False) -> ValidationResult:
    """Check macro data freshness and availability.
    
    Criteria:
        - DGS10, T10Y2Y, CPIAUCSL available
        - Cadence-aware freshness:
          * Daily-ish series (DGS10, T10Y2Y): <= 60 days stale
          * Monthly-ish series (CPIAUCSL): <= 90 days stale
    """
    result = ValidationResult("Macro Data")
    
    macro_checks = [
        ("DGS10", 60, "daily-ish"),      # 10Y Treasury Yield
        ("T10Y2Y", 60, "daily-ish"),     # 10Y-2Y Spread
        ("CPIAUCSL", 90, "monthly-ish"), # Core CPI
    ]
    
    now = datetime.now()
    
    for series_id, max_age_days, cadence in macro_checks:
        try:
            series = load_series(series_id)
            if series.empty:
                result.warn(f"{series_id}: No data available")
                continue
            
            last_date = series.index[-1]
            age_days = (now - last_date).days
            
            if age_days > max_age_days:
                result.fail(
                    f"{series_id}: Stale data ({age_days}d old, max {max_age_days}d for {cadence})"
                )
            else:
                result.succeed(
                    f"{series_id}: Fresh ({age_days}d old, {cadence})"
                )
            
            if verbose:
                result.info(f"{series_id}: Last date={last_date.strftime('%Y-%m-%d')}, n={len(series)}")
        
        except Exception as e:
            result.fail(f"{series_id}: Load failed - {e}")
    
    return result


def check_candidates(
    candidates: List[Dict[str, Any]],
    n_expected: int,
    objective_name: str,
    verbose: bool = False,
    returns: Optional[pd.DataFrame] = None,
    catalog: Optional[Dict[str, Any]] = None
) -> ValidationResult:
    """Check candidate generation and constraints.
    
    Criteria:
        - Generated at least n_expected candidates
        - All weights non-negative
        - All weights sum to ~1.0 (±1%)
        - Satellite caps enforced (if applicable)
    """
    result = ValidationResult("Candidate Generation")
    
    if len(candidates) < n_expected:
        result.fail(
            f"Insufficient candidates: {len(candidates)} < {n_expected} expected"
        )
    else:
        result.succeed(f"Generated {len(candidates)} candidates (expected {n_expected})")
    
    # Distinctness helper
    def _cos(a: pd.Series, b: pd.Series) -> float:
        import numpy as _np
        aa = a.values; bb = b.values
        den = float((_np.linalg.norm(aa) * _np.linalg.norm(bb)) or 0.0)
        if den == 0:
            return 0.0
        return float(_np.dot(aa, bb) / den)

    # Check each candidate
    for i, cand in enumerate(candidates):
        name = cand.get("name", f"Candidate_{i}")
        weights = cand.get("weights", {})
        
        if not weights:
            result.fail(f"{name}: No weights found")
            continue
        
        # Check non-negative
        if any(w < 0 for w in weights.values()):
            result.fail(f"{name}: Negative weights found")
        
        # Check sum to 1
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01):
            result.fail(f"{name}: Weights sum to {total:.3f} (expected ~1.0)")
        
        # Caps check (if catalog provided)
        if catalog is not None and "assets" in catalog:
            # Build case-insensitive symbol to class mapping
            sym2cls = {str(a.get("symbol","")).upper(): a.get("class","") for a in catalog["assets"]}
            caps = (catalog.get("caps", {}) or {}).get("asset_class", {})
            class_sums: Dict[str, float] = {}
            for s, w in weights.items():
                cls = sym2cls.get(str(s).upper(), "")
                if not cls:
                    # Try heuristic fallback for common tickers
                    s_up = str(s).upper()
                    if s_up in {"DBC", "GSG", "GLD", "IAU"}:
                        cls = "commodities"
                    elif s_up in {"SPY", "VTI", "QQQ", "DIA", "IWM", "VEA", "VXUS", "VWO", "EFA"}:
                        cls = "equity_us"
                    elif s_up in {"TLT", "IEF"}:
                        cls = "bonds_tsy"
                    elif s_up in {"BND", "LQD"}:
                        cls = "bonds_ig"
                    elif s_up in {"MUB"}:
                        cls = "munis"
                    elif s_up in {"BIL", "SHY"}:
                        cls = "cash"
                # Only accumulate if we have a valid class
                if cls and cls != "unknown":
                    class_sums[cls] = class_sums.get(cls, 0.0) + float(w)
            
            # Debug: print class_sums for first failing case
            if verbose and i < 12:
                print(f"  {name} class_sums: {class_sums}")
                print(f"  {name} weights: {weights}")
            
            for cls, total in class_sums.items():
                if not cls or cls == "unknown":
                    continue
                cap = float(caps.get(cls, 1.0))
                # Allow small numerical tolerance for floating point and downstream renormalization
                if total > cap + 1e-3:
                    result.fail(f"{name}: class cap violated for {cls} = {total:.2f} > {cap:.2f}")

        # Consistency gate: metrics from same window
        if returns is not None:
            wser = pd.Series(weights).reindex(returns.columns).fillna(0.0)
            aligned = align_returns_matrix(returns, list(wser.index))
            port = (aligned * wser.reindex(aligned.columns).fillna(0.0)).sum(axis=1)
            ok = assert_metrics_consistency((1+port).cumprod(), port)
            if not ok:
                result.fail(f"{name}: metrics/curve inconsistency detected")

        if verbose and i < 3:  # Show first 3
            result.info(f"{name}: sum={total:.3f}, n={len(weights)}")
    
    if not candidates:
        result.fail("No candidates generated")
    else:
        # Distinctness by cosine similarity of weights
        if returns is not None:
            mats = []
            for c in candidates:
                w = pd.Series(c.get("weights", {})).reindex(returns.columns).fillna(0.0)
                mats.append(w)
            uniq = 0
            picked: List[pd.Series] = []
            for w in mats:
                if not picked:
                    picked.append(w); uniq += 1; continue
                sims = [_cos(w, p) for p in picked]
                if max(sims) < 0.995:
                    picked.append(w); uniq += 1
            if uniq < max(2, min(5, n_expected)):
                result.fail(f"Distinctness low: {uniq} unique by weights (threshold 0.995)")
            else:
                result.succeed(f"Distinct candidates: {uniq}")
        result.succeed(f"All weight constraints satisfied")
    
    return result


def check_metrics(
    candidates: List[Dict[str, Any]],
    verbose: bool = False
) -> ValidationResult:
    """Check metrics plausibility across horizons.
    
    Criteria:
        - CAGR in [-0.8, 0.8] (reasonable annual return range)
        - Sharpe in [-1.5, 4.0] (reasonable Sharpe range)
        - MaxDD in [-0.98, -0.01] (reasonable drawdown range)
        - Metrics present for 1Y and 5Y horizons (if data allows)
    """
    result = ValidationResult("Metrics Plausibility")
    
    if not candidates:
        result.fail("No candidates to check metrics")
        return result
    
    # Check at least one candidate has metrics
    has_metrics = any("metrics" in c for c in candidates)
    if not has_metrics:
        result.fail("No metrics found in candidates")
        return result
    
    result.succeed("Metrics present in candidates")
    
    # Plausibility ranges
    ranges = {
        "CAGR": (-0.8, 0.8),
        "Sharpe": (-1.5, 4.0),
        "MaxDD": (-0.98, -0.01),
    }
    
    violations = []
    
    for i, cand in enumerate(candidates):
        metrics = cand.get("metrics", {})
        if not metrics:
            continue
        
        name = cand.get("name", f"Candidate_{i}")
        
        # Check primary metrics
        for metric, (min_val, max_val) in ranges.items():
            val = metrics.get(metric)
            if val is None:
                continue
            
            if not (min_val <= val <= max_val):
                violations.append(
                    f"{name}: {metric}={val:.2f} outside [{min_val}, {max_val}]"
                )
        
        if verbose and i < 3:
            result.info(
                f"{name}: CAGR={metrics.get('CAGR', 'N/A'):.2%}, "
                f"Sharpe={metrics.get('Sharpe', 'N/A'):.2f}, "
                f"MaxDD={metrics.get('MaxDD', 'N/A'):.1%}"
            )
    
    if violations:
        for v in violations[:5]:  # Show first 5
            result.fail(v)
    else:
        result.succeed("All metrics within plausible ranges")
    
    return result


def check_receipts(
    receipts: List[Dict[str, Any]],
    tickers: List[str],
    verbose: bool = False
) -> ValidationResult:
    """Check receipt completeness and structure.
    
    Criteria:
        - One receipt per ticker
        - Required keys present: ticker, provider, backfill_pct, first, last, nan_rate
        - Values are reasonable (dates valid, percentages in [0,1])
    """
    result = ValidationResult("Receipt Integrity")
    
    if not receipts:
        result.fail("No receipts generated")
        return result
    
    # Check count matches tickers
    if len(receipts) != len(tickers):
        result.warn(
            f"Receipt count mismatch: {len(receipts)} receipts for {len(tickers)} tickers"
        )
    else:
        result.succeed(f"One receipt per ticker ({len(receipts)})")
    
    # Check structure
    required_keys = {"ticker", "provider", "backfill_pct", "first", "last", "nan_rate"}
    missing_keys = []
    
    for i, receipt in enumerate(receipts):
        missing = required_keys - set(receipt.keys())
        if missing:
            ticker = receipt.get("ticker", f"Receipt_{i}")
            missing_keys.append(f"{ticker}: {missing}")
    
    if missing_keys:
        for mk in missing_keys[:3]:  # Show first 3
            result.fail(f"Missing keys - {mk}")
    else:
        result.succeed("All receipts have required keys")
    
    # Check value plausibility
    for receipt in receipts:
        ticker = receipt.get("ticker", "?")
        
        # Check nan_rate
        nan_rate = receipt.get("nan_rate")
        if nan_rate is not None:
            try:
                nan_float = float(nan_rate)
                if not (0.0 <= nan_float <= 1.0):
                    result.warn(f"{ticker}: nan_rate={nan_float} outside [0,1]")
            except (ValueError, TypeError):
                result.warn(f"{ticker}: Invalid nan_rate={nan_rate}")
        
        if verbose:
            result.info(
                f"{ticker}: provider={receipt.get('provider', '?')}, "
                f"nan_rate={receipt.get('nan_rate', '?')}"
            )
    
    return result


def gen_and_validate(
    tickers: List[str],
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    objective_name: str,
    n_candidates: int,
    seed: int,
    verbose: bool = False
) -> tuple[List[Dict[str, Any]], ValidationResult]:
    """Generate candidates and validate their structure.
    
    Returns:
        (candidates, validation_result)
    """
    result = ValidationResult("Candidate Generation & Validation")
    
    # Get objective config
    obj_cfg = DEFAULT_OBJECTIVES.get(objective_name)
    if obj_cfg is None:
        result.fail(f"Unknown objective: {objective_name}")
        return [], result
    
    result.succeed(f"Objective config loaded: {objective_name}")
    
    # Generate candidates
    try:
        candidates = generate_candidates(
            returns=returns,
            objective_cfg=obj_cfg,
            n_candidates=n_candidates,
            seed=seed
        )
        if not candidates:
            result.fail("generate_candidates returned empty list")
            return [], result
        # Check for NaN in bounds
        bounds = getattr(obj_cfg, "bounds", None)
        if bounds is not None:
            import numpy as np
            if any([v is None or (isinstance(v, float) and np.isnan(v)) for v in bounds.values()]):
                result.fail("Invalid constraint bounds: NaN or None detected")
        result.succeed(f"Generated {len(candidates)} candidates")
    except Exception as e:
        result.fail(f"generate_candidates failed: {e}")
        return [], result
    return candidates, result


def main():
    parser = argparse.ArgumentParser(
        description="V3 Portfolio Simulation Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --objective balanced --n-candidates 6
  %(prog)s --objective growth --n-candidates 8 --json
  %(prog)s --tickers SPY,TLT,GLD --start 2020-01-01 --verbose

Exit codes:
  0: All checks passed
  1: One or more checks failed
        """
    )
    
    parser.add_argument(
        "--objective",
        default="balanced",
        choices=list(DEFAULT_OBJECTIVES.keys()),
        help="Portfolio objective (default: balanced)"
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated ticker list (default: use objective defaults)"
    )
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date for price data (default: 2010-01-01)"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=6,
        help="Number of candidates to generate (default: 6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output one-line JSON summary"
    )
    parser.add_argument(
        "--k-days",
        type=int,
        default=1260,
        help="Lookback window in trading days for returns cleaner (default: 1260 ~5y)"
    )
    parser.add_argument(
        "--min-non-na",
        type=int,
        default=126,
        help="Minimum non-NaN observations per asset to keep (default: 126 ~6m)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed diagnostic information"
    )
    
    args = parser.parse_args()
    
    # Determine tickers
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        # Fallback seed tickers; objective-specific broad defaults could be added later
        tickers = ["SPY", "TLT", "GLD", "QQQ", "VTI"]
    
    if not args.json:
        print(f"\n{'='*70}")
        print(f"V3 Portfolio Simulation Validator")
        print(f"{'='*70}")
        print(f"Objective: {args.objective}")
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Start: {args.start}")
        print(f"Candidates: {args.n_candidates}")
        print(f"{'='*70}\n")
    
    results: List[ValidationResult] = []
    
    # 1. Load and check data
    try:
        if args.verbose:
            print("Loading price data...", file=sys.stderr)
        
        prices, provenance = get_prices_with_provenance(tickers, start=args.start)
        
        if prices.empty:
            print("ERROR: No price data loaded", file=sys.stderr)
            sys.exit(1)
        
        # Pre-clean universe to expand effective symbols while maintaining quality
        try:
            from core.utils.universe_cleaner import preclean_universe_for_simulation
            prices_clean, preclean_diags = preclean_universe_for_simulation(
                prices,
                k_days=int(args.k_days),
                min_non_na=int(args.min_non_na),
                min_overlap_pct=0.70,
                verbose=args.verbose
            )
            if not prices_clean.empty and len(prices_clean.columns) > len(prices.columns) * 0.5:
                if args.verbose:
                    print(f"[preclean] Kept {len(prices_clean.columns)}/{len(prices.columns)} symbols", file=sys.stderr)
                prices = prices_clean
                # Update tickers list to match cleaned universe
                tickers = list(prices.columns)
        except ImportError:
            if args.verbose:
                print("[preclean] universe_cleaner not available, skipping", file=sys.stderr)
        except Exception as e:
            if args.verbose:
                print(f"[preclean] Failed: {e}, continuing with original data", file=sys.stderr)
        
        data_result = check_data(tickers, prices, provenance, args.verbose)
        results.append(data_result)
        
    except Exception as e:
        print(f"ERROR: Failed to load price data: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 2. Clean and check returns
    try:
        if args.verbose:
            print("Cleaning returns...", file=sys.stderr)
        
        # Use robust cleaner path (strict=False) to avoid over-dropping symbols
        try:
            _tmp = clean_prices_to_returns(prices, winsor_p=0.005, min_non_na=int(args.min_non_na), k_days=int(args.k_days), strict=False, return_diagnostics=True)
            if isinstance(_tmp, tuple):
                returns, cleaner_diags = _tmp
            else:
                returns = _tmp; cleaner_diags = {"dropped_symbols": [], "kept_symbols": [], "window": (None, None)}
        except TypeError:
            returns = clean_prices_to_returns(prices, winsor_p=0.005, min_non_na=int(args.min_non_na))
            cleaner_diags = {"dropped_symbols": [], "kept_symbols": [], "window": (None, None)}
        except Exception:
            returns = pd.DataFrame(); cleaner_diags = {"dropped_symbols": [], "kept_symbols": [], "window": (None, None)}
        
        if not isinstance(returns, pd.DataFrame) or returns.empty:
            print("ERROR: Returns cleaning produced empty DataFrame", file=sys.stderr)
            sys.exit(1)
        
        returns_result = check_returns(returns, args.verbose)
        results.append(returns_result)
        
    except Exception as e:
        print(f"ERROR: Failed to clean returns: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 3. Check macro data
    try:
        if args.verbose:
            print("Checking macro data...", file=sys.stderr)
        
        macro_result = check_macro(args.verbose)
        results.append(macro_result)
        
    except Exception as e:
        print(f"WARNING: Macro check failed: {e}", file=sys.stderr)
        # Don't exit - macro is supplementary
    
    # 4. Generate and validate candidates
    try:
        if args.verbose:
            print("Generating candidates...", file=sys.stderr)
        candidates, gen_result = gen_and_validate(
            tickers, prices, returns, args.objective,
            args.n_candidates, args.seed, args.verbose
        )
        results.append(gen_result)
        
        # Validate candidate structure
        if candidates:
            # Load catalog for cap checks
            try:
                import json as _json
                catalog = _json.loads((ROOT/"config"/"assets_catalog.json").read_text())
            except Exception:
                catalog = None
            cand_result = check_candidates(
                candidates, args.n_candidates, args.objective, args.verbose, returns=returns, catalog=catalog
            )
            # Enforce distinct >=5; include cleaner diagnostics when failing
            if not cand_result.passed and isinstance(cleaner_diags, dict):
                uniq_errs = [e for e in cand_result.messages if "Distinctness low" in e]
                if uniq_errs and cleaner_diags.get("dropped_symbols"):
                    ds = ", ".join([f"{s}({r})" for s,r in cleaner_diags["dropped_symbols"][:10]])
                    cand_result.warn(f"Cleaner dropped: {ds}")
            results.append(cand_result)
            
            # Check metrics
            metrics_result = check_metrics(candidates, args.verbose)
            results.append(metrics_result)
        
    except Exception as e:
        print(f"ERROR: Candidate generation failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 5. Build and validate receipts
    try:
        if args.verbose:
            print("Building receipts...", file=sys.stderr)
        
        receipts = build_receipts(prices, provenance)
        
        receipts_result = check_receipts(receipts, tickers, args.verbose)
        results.append(receipts_result)
        
    except Exception as e:
        print(f"WARNING: Receipt generation failed: {e}", file=sys.stderr)
        # Don't exit - receipts are supplementary
    
    # Summary
    all_passed = all(r.passed for r in results)
    
    if args.json:
        # One-line JSON output
        summary = {
            "timestamp": datetime.now().isoformat(),
            "objective": args.objective,
            "tickers": tickers,
            "n_candidates": args.n_candidates,
            "passed": all_passed,
            "checks": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "errors": [m for m in r.messages if m.startswith("✗")],
                    "warnings": r.warnings
                }
                for r in results
            ]
        }
        print(json.dumps(summary))
    else:
        # Human-readable output
        print(f"\n{'='*70}")
        print("VALIDATION SUMMARY")
        print(f"{'='*70}\n")
        
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{status} - {result.name}")
            
            for msg in result.messages:
                print(f"  {msg}")
            
            for warn in result.warnings:
                print(f"  {warn}")
            
            print()
        
        print(f"{'='*70}")
        if all_passed:
            print("✓ ALL CHECKS PASSED")
        else:
            failed_checks = [r.name for r in results if not r.passed]
            print(f"✗ FAILED CHECKS: {', '.join(failed_checks)}")
        print(f"{'='*70}\n")
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()

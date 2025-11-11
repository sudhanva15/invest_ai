#!/usr/bin/env python3
"""
Portfolio Scenario Runner - CLI tool for candidate generation and stress testing.

USAGE EXAMPLES:

Basic usage - generate candidates for growth objective:
    python3 dev/run_scenarios.py --objective growth --n-candidates 10 --horizon 5y

Custom optimizers and satellite caps:
    python3 dev/run_scenarios.py --objective income --optimizers hrp,max_sharpe --satellite-caps 0.20,0.25,0.35

Apply stress test shocks:
    python3 dev/run_scenarios.py --objective balanced --shock equity-10% --shock rates+100bp

Custom ticker universe:
    python3 dev/run_scenarios.py --objective growth --tickers SPY,QQQ,IWM,VGT,XLK --start 2018-01-01

Force macro regime:
    python3 dev/run_scenarios.py --objective balanced --macro-override '{"regime":"Recessionary"}'

SHOCK MODEL (Simple & Deterministic):
- "equity-X%": Apply X% shock to all equity tickers (SPY, QQQ, VTI, etc.) at t=0
- "rates+Xbp": Shift bond returns based on duration proxy (100bp = -5% for TLT, -2% for IEF)
- "gold+X%": Apply X% shock to gold (GLD, IAU) at t=0

MACRO OVERRIDE:
- Provide JSON dict to force regime label: {"regime": "Tightening"}
- Overrides computed regime from macro indicators
- Valid regimes: Risk-on, Tightening, Disinflation, Recessionary

OUTPUT:
- <stem>.json: Full results with candidates, metrics, regime info
- <stem>_weights.csv: Portfolio weights matrix (candidates × tickers)
- <stem>_metrics.csv: Performance metrics matrix
- Console: Top 5 candidates table

SCORING HEURISTIC:
- score = Sharpe - 0.2 * |MaxDD|
- Higher is better (rewards high Sharpe, penalizes large drawdowns)
"""

import sys
import os
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Print build version at startup
try:
    from core.utils.version import get_build_version
    print(f"[build] {get_build_version()}", file=sys.stderr)
except Exception:
    pass

import argparse
import json
import csv
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy/pandas types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        elif isinstance(obj, pd.Timedelta):
            return str(obj)
        return super().default(obj)

# Import V3 modules
from core.data_ingestion import get_prices_with_provenance
from core.preprocessing import compute_returns
from core.utils.metrics import annualized_metrics
from core.macro.regime import regime_features, label_regimes, current_regime, load_macro_data
from core.recommendation_engine import (
    generate_candidates,
    DEFAULT_OBJECTIVES,
    ObjectiveConfig,
    _class_of_symbol,
)

# Utility to load catalog
try:
    from core.utils.env_tools import load_config
    CFG = load_config()
    from core.utils import load_json
    CAT = load_json(str(ROOT / "config/assets_catalog.json"))
except Exception:
    import yaml
    with open(ROOT / "config/config.yaml") as f:
        CFG = yaml.safe_load(f)
    with open(ROOT / "config/assets_catalog.json") as f:
        CAT = json.load(f)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate portfolio candidates with optional stress tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Portfolio configuration
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced",
        choices=list(DEFAULT_OBJECTIVES.keys()),
        help="Portfolio objective (default: balanced)"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="SPY,QQQ,TLT,IEF,GLD",
        help="Comma-separated ticker list (default: SPY,QQQ,TLT,IEF,GLD)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2015-01-01",
        help="Start date for historical data (default: 2015-01-01)"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        default="5y",
        choices=["1y", "3y", "5y", "10y"],
        help="Analysis horizon (default: 5y)"
    )
    
    # Candidate generation
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=8,
        help="Number of candidates to generate (default: 8, range: 3-12)"
    )
    parser.add_argument(
        "--satellite-caps",
        type=str,
        default=None,
        help="Comma-separated satellite caps (e.g., 0.20,0.25,0.30)"
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default=None,
        help="Comma-separated optimizer methods (e.g., hrp,max_sharpe,min_var)"
    )
    
    # Stress testing
    parser.add_argument(
        "--shock",
        action="append",
        default=[],
        help="Apply shock: equity-10%%, rates+100bp, gold+5%% (repeatable)"
    )
    parser.add_argument(
        "--macro-override",
        type=str,
        default=None,
        help='Force regime label: {"regime": "Tightening"}'
    )
    
    # Output
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output file stem (default: dev/artifacts/scenario_<objective>_<timestamp>)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to stderr"
    )
    
    return parser.parse_args()


def log(msg: str, verbose: bool = False, force: bool = False):
    """Log to stderr if verbose is enabled or force=True."""
    if verbose or force:
        print(msg, file=sys.stderr)


def parse_shock(shock_str: str) -> Tuple[str, str, float]:
    """
    Parse shock string into (asset_class, direction, magnitude).
    
    Examples:
        "equity-10%" -> ("equity", "-", 0.10)
        "rates+100bp" -> ("rates", "+", 1.00)  # 100bp = 1%
        "gold+5%" -> ("gold", "+", 0.05)
    
    Returns:
        (asset_class, direction, magnitude)
    """
    shock_str = shock_str.strip().lower()
    
    # Parse direction
    if "+" in shock_str:
        direction = "+"
        parts = shock_str.split("+")
    elif "-" in shock_str:
        direction = "-"
        parts = shock_str.split("-")
    else:
        raise ValueError(f"Invalid shock format: {shock_str}")
    
    asset_class = parts[0].strip()
    magnitude_str = parts[1].strip()
    
    # Parse magnitude
    if magnitude_str.endswith("bp"):
        # Basis points (100bp = 1%)
        magnitude = float(magnitude_str[:-2]) / 100.0 / 100.0
    elif magnitude_str.endswith("%"):
        # Percentage
        magnitude = float(magnitude_str[:-1]) / 100.0
    else:
        # Assume decimal
        magnitude = float(magnitude_str)
    
    return asset_class, direction, magnitude


def apply_shocks(
    returns: pd.DataFrame,
    shocks: List[str],
    catalog: dict,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Apply deterministic shocks to returns.
    
    Shock Model:
        - "equity-X%": Multiply first row of equity returns by (1 - X)
        - "rates+Xbp": Adjust bond returns based on duration proxy
          Duration proxy: TLT ~17yr, IEF ~7yr, SHY ~2yr
          100bp rise → TLT -17%, IEF -7%, SHY -2%
        - "gold+X%": Multiply first row of gold returns by (1 + X)
    
    Args:
        returns: DataFrame of daily returns (columns=tickers, index=dates)
        shocks: List of shock strings (e.g., ["equity-10%", "rates+100bp"])
        catalog: Asset catalog for classification
        verbose: Enable logging
    
    Returns:
        Modified returns DataFrame
    """
    if not shocks:
        return returns
    
    returns = returns.copy()
    
    # Build symbol -> class mapping from catalog
    symbol_to_class = {}
    if "assets" in catalog:
        for asset in catalog["assets"]:
            symbol_to_class[asset["symbol"]] = asset["class"]
    else:
        # Fallback: catalog is dict of symbol -> metadata
        for symbol, metadata in catalog.items():
            if isinstance(metadata, dict) and "asset_class" in metadata:
                symbol_to_class[symbol] = metadata["asset_class"]
            elif isinstance(metadata, dict) and "class" in metadata:
                symbol_to_class[symbol] = metadata["class"]
    
    for shock_str in shocks:
        try:
            asset_class, direction, magnitude = parse_shock(shock_str)
            
            log(f"Applying shock: {shock_str} -> {asset_class} {direction}{magnitude*100:.1f}%", verbose, force=True)
            
            # Identify affected tickers
            if asset_class == "equity":
                affected = [
                    s for s in returns.columns
                    if symbol_to_class.get(s) in {
                        "public_equity", "public_equity_intl", "public_equity_em", "public_equity_sector"
                    }
                ]
            elif asset_class == "rates" or asset_class == "bonds":
                affected = [
                    s for s in returns.columns
                    if symbol_to_class.get(s) in {
                        "treasury_long", "treasury_short", "corporate_bond", "high_yield", "tax_eff_muni", "bond"
                    }
                ]
            elif asset_class == "gold":
                affected = [
                    s for s in returns.columns
                    if symbol_to_class.get(s) in {"gold", "commodity"} or s.upper() in {"GLD", "IAU"}
                ]
            else:
                log(f"Warning: Unknown asset class '{asset_class}', skipping shock", verbose, force=True)
                continue
            
            if not affected:
                log(f"Warning: No tickers found for asset class '{asset_class}'", verbose, force=True)
                continue
            
            # Apply shock to first row (t=0)
            for ticker in affected:
                if asset_class in ["rates", "bonds"]:
                    # Duration-based adjustment for rates
                    # Simple proxy: TLT ~17yr, IEF ~7yr, SHY ~2yr
                    duration_map = {
                        "TLT": 17.0,
                        "IEF": 7.0,
                        "SHY": 2.0,
                        "BND": 6.5,
                        "AGG": 6.5,
                        "LQD": 8.0,
                        "HYG": 4.0,
                        "MUB": 5.0,
                    }
                    duration = duration_map.get(ticker.upper(), 7.0)  # Default 7yr
                    
                    # 100bp rate increase → -(duration * 100bp) price impact
                    # magnitude is already in decimal (100bp = 0.01)
                    rate_change = magnitude if direction == "+" else -magnitude
                    price_impact = -duration * rate_change
                    
                    returns.loc[returns.index[0], ticker] *= (1 + price_impact)
                    log(f"  {ticker}: duration={duration:.1f}yr, impact={price_impact*100:.2f}%", verbose)
                else:
                    # Direct shock for equity/gold
                    shock_multiplier = (1 - magnitude) if direction == "-" else (1 + magnitude)
                    returns.loc[returns.index[0], ticker] *= shock_multiplier
                    log(f"  {ticker}: shock={shock_multiplier-1:.2%}", verbose)
        
        except Exception as e:
            log(f"Error applying shock '{shock_str}': {e}", verbose, force=True)
    
    return returns


def compute_horizon_metrics(
    returns: pd.Series,
    horizon: str
) -> dict:
    """
    Compute metrics for a specific horizon.
    
    Args:
        returns: Portfolio daily returns
        horizon: "1y", "3y", "5y", or "10y"
    
    Returns:
        Dict with CAGR, Volatility, Sharpe, MaxDD for the horizon
    """
    horizon_days = {
        "1y": 252,
        "3y": 252 * 3,
        "5y": 252 * 5,
        "10y": 252 * 10,
    }
    
    days = horizon_days.get(horizon, 252 * 5)
    
    # Take most recent N days
    if len(returns) > days:
        returns_horizon = returns.tail(days)
    else:
        returns_horizon = returns
    
    return annualized_metrics(returns_horizon)


def generate_custom_candidates(
    returns: pd.DataFrame,
    objective_cfg: ObjectiveConfig,
    catalog: dict,
    n_candidates: int = 8,
    custom_sat_caps: Optional[List[float]] = None,
    custom_optimizers: Optional[List[str]] = None,
    verbose: bool = False
) -> List[dict]:
    """
    Generate candidates with optional custom satellite caps and optimizers.
    
    If custom parameters are provided, expand the candidate grid accordingly.
    """
    # Use default generate_candidates if no customization
    if custom_sat_caps is None and custom_optimizers is None:
        return generate_candidates(
            returns,
            objective_cfg,
            catalog=catalog,
            n_candidates=n_candidates
        )
    
    # Otherwise, manually build variant grid
    from core.recommendation_engine import _optimize_with_method, _apply_objective_constraints
    
    # Subset to objective universe
    all_symbols = list(returns.columns)
    if objective_cfg.universe_filter:
        if callable(objective_cfg.universe_filter):
            filtered_symbols = objective_cfg.universe_filter(all_symbols, catalog)
        else:
            filtered_symbols = [s for s in all_symbols if s in objective_cfg.universe_filter]
    else:
        filtered_symbols = all_symbols
    
    rets = returns[filtered_symbols].dropna(how="all")
    rets = rets.loc[:, rets.std() > 0]
    
    if rets.empty:
        return []
    
    symbols = list(rets.columns)
    
    # Build variant space
    optimizers = custom_optimizers or ["hrp", "max_sharpe", "min_var", "risk_parity", "equal_weight"]
    sat_caps = custom_sat_caps or [0.20, 0.25, 0.30, 0.35]
    
    candidates = []
    
    for opt_method in optimizers:
        for sat_cap in sat_caps:
            if len(candidates) >= n_candidates:
                break
            
            variant_name = f"{opt_method.upper()} - Sat {int(sat_cap*100)}%"
            
            try:
                w = _optimize_with_method(
                    rets=rets,
                    symbols=symbols,
                    method=opt_method,
                    catalog=catalog,
                    objective_cfg=objective_cfg
                )
                
                if not w or sum(w.values()) == 0:
                    log(f"Skipping {variant_name}: empty weights", verbose)
                    continue
                
                # Apply constraints
                w = _apply_objective_constraints(
                    w,
                    symbols=symbols,
                    catalog=catalog,
                    core_min=objective_cfg.bounds.get("core_min", 0.65),
                    sat_max_total=sat_cap,
                    sat_max_single=objective_cfg.bounds.get("sat_max_single", 0.07)
                )
                
                if not w or sum(w.values()) == 0:
                    log(f"Skipping {variant_name}: failed constraints", verbose)
                    continue
                
                # Compute returns
                weights_vec = pd.Series(w).reindex(rets.columns).fillna(0.0)
                port_ret = (rets * weights_vec).sum(axis=1)
                
                metrics = annualized_metrics(port_ret)
                
                candidates.append({
                    "name": variant_name,
                    "weights": w,
                    "metrics": metrics,
                    "notes": f"{objective_cfg.notes} | Optimizer: {opt_method}, Sat cap: {sat_cap:.0%}",
                    "optimizer": opt_method,
                    "sat_cap": sat_cap,
                })
                
                log(f"Generated candidate: {variant_name}", verbose)
                
            except Exception as e:
                log(f"Error generating {variant_name}: {e}", verbose)
                continue
        
        if len(candidates) >= n_candidates:
            break
    
    # Sort by Sharpe
    candidates = sorted(candidates, key=lambda x: x["metrics"].get("Sharpe", -999), reverse=True)
    
    if candidates:
        candidates[0]["shortlist"] = True
    
    return candidates[:n_candidates]


def main():
    """Main CLI entry point."""
    args = parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    log(f"Portfolio Scenario Runner - Objective: {args.objective}", args.verbose, force=True)
    log(f"Tickers: {args.tickers}", args.verbose, force=True)
    log(f"Start: {args.start}, Horizon: {args.horizon}", args.verbose, force=True)
    
    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    # Parse optional custom parameters
    custom_sat_caps = None
    if args.satellite_caps:
        custom_sat_caps = [float(x.strip()) for x in args.satellite_caps.split(",")]
        log(f"Custom satellite caps: {custom_sat_caps}", args.verbose)
    
    custom_optimizers = None
    if args.optimizers:
        custom_optimizers = [x.strip().lower() for x in args.optimizers.split(",")]
        log(f"Custom optimizers: {custom_optimizers}", args.verbose)
    
    # Validate n_candidates
    if not (3 <= args.n_candidates <= 12):
        log("Error: --n-candidates must be between 3 and 12", args.verbose, force=True)
        sys.exit(1)
    
    # Load objective config
    objective_cfg = DEFAULT_OBJECTIVES.get(args.objective)
    if not objective_cfg:
        log(f"Error: Unknown objective '{args.objective}'", args.verbose, force=True)
        sys.exit(1)
    
    # Step 1: Load prices
    log("Loading price data...", args.verbose, force=True)
    try:
        prices, provenance = get_prices_with_provenance(tickers, start=args.start)
        
        if prices.empty:
            log("Error: No price data returned", args.verbose, force=True)
            sys.exit(1)
        
        log(f"Loaded {len(prices)} rows for {len(prices.columns)} tickers", args.verbose)
    except Exception as e:
        log(f"Error loading prices: {e}", args.verbose, force=True)
        sys.exit(1)
    
    # Step 2: Compute returns
    log("Computing returns...", args.verbose)
    try:
        returns = compute_returns(prices)
        
        if returns.empty:
            log("Error: Returns computation failed", args.verbose, force=True)
            sys.exit(1)
        
        log(f"Returns shape: {returns.shape}", args.verbose)
    except Exception as e:
        log(f"Error computing returns: {e}", args.verbose, force=True)
        sys.exit(1)
    
    # Step 3: Load macro data and compute regime
    log("Computing macro regime...", args.verbose)
    try:
        macro_df = load_macro_data()
        
        if not macro_df.empty:
            features = regime_features(macro_df)
            labels = label_regimes(features, method="rule_based")
            regime = current_regime(features=features, labels=labels)
        else:
            regime = "Unknown"
        
        # Apply macro override if provided
        if args.macro_override:
            override = json.loads(args.macro_override)
            if "regime" in override:
                regime = override["regime"]
                log(f"Regime overridden to: {regime}", args.verbose, force=True)
        
        log(f"Current regime: {regime}", args.verbose, force=True)
    except Exception as e:
        log(f"Error computing regime: {e}", args.verbose)
        regime = "Unknown"
    
    # Step 4: Apply shocks
    if args.shock:
        log("Applying shocks...", args.verbose, force=True)
        returns = apply_shocks(returns, args.shock, CAT, args.verbose)
    
    # Step 5: Generate candidates
    log(f"Generating {args.n_candidates} candidates...", args.verbose, force=True)
    try:
        # Use main generate_candidates for V3 ranking diversity (if no custom params)
        if custom_sat_caps is None and custom_optimizers is None:
            from core.recommendation_engine import generate_candidates
            candidates = generate_candidates(
                returns,
                objective_cfg,
                catalog=CAT,
                n_candidates=args.n_candidates,
                seed=args.seed
            )
        else:
            # Use custom variant generator when overrides provided
            candidates = generate_custom_candidates(
                returns,
                objective_cfg,
                catalog=CAT,
                n_candidates=args.n_candidates,
                custom_sat_caps=custom_sat_caps,
                custom_optimizers=custom_optimizers,
                verbose=args.verbose
            )
        
        if not candidates:
            log("Warning: No candidates generated, using equal-weight fallback", args.verbose, force=True)
            # Fallback to equal weight
            n = len(tickers)
            candidates = [{
                "name": "Equal Weight (Fallback)",
                "weights": {t: 1.0/n for t in tickers},
                "metrics": annualized_metrics((returns * (1.0/n)).sum(axis=1)),
                "notes": "Fallback portfolio",
                "optimizer": "equal_weight",
                "sat_cap": 1.0,
            }]
        
        log(f"Generated {len(candidates)} candidates", args.verbose, force=True)
    except Exception as e:
        log(f"Error generating candidates: {e}", args.verbose, force=True)
        sys.exit(1)
    
    # Step 6: Compute horizon metrics and score
    log("Computing horizon metrics and scores...", args.verbose)
    
    # Check if candidates already have scores (from generate_candidates with RANK_DIVERSITY)
    has_existing_scores = all("score" in c for c in candidates)
    
    for cand in candidates:
        # Reconstruct portfolio returns
        weights_vec = pd.Series(cand["weights"]).reindex(returns.columns).fillna(0.0)
        port_ret = (returns * weights_vec).sum(axis=1)
        
        # Compute metrics for each horizon
        horizon_metrics = {}
        for h in ["1y", "3y", "5y", "10y"]:
            horizon_metrics[h] = compute_horizon_metrics(port_ret, h)
        
        cand["horizon_metrics"] = horizon_metrics
        
        # Only recompute score if not already present (preserves V3 diversity scoring)
        if not has_existing_scores:
            # Compute composite score using stated horizon
            stated_metrics = horizon_metrics.get(args.horizon, cand["metrics"])
            sharpe = stated_metrics.get("Sharpe", 0.0)
            maxdd = stated_metrics.get("MaxDD", 0.0)
            
            # Score = Sharpe - 0.2 * |MaxDD|
            score = sharpe - 0.2 * abs(maxdd)
            cand["score"] = score
    
    # Sort by score
    candidates = sorted(candidates, key=lambda x: x.get("score", -999), reverse=True)
    
    # Step 7: Generate receipts
    receipts = []
    for ticker in tickers:
        prov_info = provenance.get(ticker, {})
        
        if ticker in prices.columns:
            ticker_prices = prices[ticker].dropna()
            first_date = ticker_prices.index[0] if len(ticker_prices) > 0 else None
            last_date = ticker_prices.index[-1] if len(ticker_prices) > 0 else None
            nan_rate = prices[ticker].isna().sum() / len(prices) if len(prices) > 0 else 1.0
        else:
            first_date = None
            last_date = None
            nan_rate = 1.0
        
        receipts.append({
            "ticker": ticker,
            "provider": list(prov_info.keys())[0] if prov_info else "unknown",
            "first": first_date.strftime("%Y-%m-%d") if first_date else None,
            "last": last_date.strftime("%Y-%m-%d") if last_date else None,
            "nan_rate": round(nan_rate, 4),
        })
    
    # Step 8: Prepare output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out:
        out_stem = args.out
    else:
        out_stem = f"dev/artifacts/scenario_{args.objective}_{timestamp}"
    
    out_stem = str(ROOT / out_stem)
    
    # Ensure output directory exists
    Path(out_stem).parent.mkdir(parents=True, exist_ok=True)
    
    # JSON output
    json_output = {
        "objective": args.objective,
        "tickers": tickers,
        "start_date": args.start,
        "horizon": args.horizon,
        "n_candidates": len(candidates),
        "current_regime": regime,
        "shocks_applied": args.shock,
        "candidates": [
            {
                "name": c["name"],
                "weights": c["weights"],
                "metrics": {
                    "full": c["metrics"],
                    **{h: c["horizon_metrics"][h] for h in ["1y", "3y", "5y", "10y"]}
                },
                "notes": c["notes"],
                "optimizer": c["optimizer"],
                "sat_cap": c["sat_cap"],
                "score": round(c["score"], 4),
                "shortlist": c.get("shortlist", False),
            }
            for c in candidates
        ],
        "receipts_sample": receipts,
    }
    
    with open(f"{out_stem}.json", "w") as f:
        json.dump(json_output, f, indent=2, cls=NumpyEncoder)
    
    log(f"Wrote {out_stem}.json", args.verbose, force=True)
    
    # Weights CSV
    weights_rows = []
    for c in candidates:
        row = {"candidate": c["name"]}
        row.update(c["weights"])
        weights_rows.append(row)
    
    weights_df = pd.DataFrame(weights_rows).fillna(0.0)
    weights_df.to_csv(f"{out_stem}_weights.csv", index=False)
    log(f"Wrote {out_stem}_weights.csv", args.verbose, force=True)
    
    # Metrics CSV
    metrics_rows = []
    for c in candidates:
        row = {"candidate": c["name"], "score": c["score"]}
        
        # Add metrics for each horizon
        for h in ["1y", "3y", "5y", "10y"]:
            hm = c["horizon_metrics"][h]
            row[f"CAGR_{h}"] = round(hm.get("CAGR", 0.0), 4)
            row[f"Vol_{h}"] = round(hm.get("Volatility", 0.0), 4)
            row[f"Sharpe_{h}"] = round(hm.get("Sharpe", 0.0), 4)
            row[f"MaxDD_{h}"] = round(hm.get("MaxDD", 0.0), 4)
        
        metrics_rows.append(row)
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f"{out_stem}_metrics.csv", index=False)
    log(f"Wrote {out_stem}_metrics.csv", args.verbose, force=True)
    
    # Step 9: Console summary table (top 5)
    print("\n" + "="*80)
    print(f"TOP 5 CANDIDATES - Objective: {args.objective.upper()}, Horizon: {args.horizon.upper()}")
    print("="*80)
    print(f"{'Rank':<5} {'Name':<30} {'Sharpe':<10} {'MaxDD':<10} {'Score':<10} {'*':<3}")
    print("-"*80)
    
    for i, c in enumerate(candidates[:5], 1):
        hm = c["horizon_metrics"][args.horizon]
        sharpe = hm.get("Sharpe", 0.0)
        maxdd = hm.get("MaxDD", 0.0)
        score = c["score"]
        shortlist = "⭐" if c.get("shortlist") else ""
        
        print(f"{i:<5} {c['name']:<30} {sharpe:>9.2f} {maxdd:>9.1%} {score:>9.2f} {shortlist:<3}")
    
    print("="*80)
    print(f"\nOutput files: {out_stem}.*")
    print(f"Current Regime: {regime}")
    if args.shock:
        print(f"Shocks Applied: {', '.join(args.shock)}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

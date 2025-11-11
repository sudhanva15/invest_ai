#!/usr/bin/env python3
"""
Sensitivity Grid Runner - Test portfolios across objectives × caps × shocks.

USAGE:
    python3 dev/run_sensitivity.py --objectives balanced,growth --caps 0.20,0.25,0.30 --shocks none,equity-10%

This tool runs multiple scenarios programmatically and aggregates top-1 results
into a tidy CSV for analysis. Feature flag: default OFF (run manually).

Args:
    --objectives: Comma-separated objectives (default: balanced,growth)
    --caps: Comma-separated satellite caps (default: 0.20,0.25,0.30,0.35)
    --shocks: Comma-separated shocks (default: none)
    --n-candidates: Candidates per run (default: 6)
    --seed: Random seed (default: 42)
    --out: Output CSV path (default: dev/artifacts/sensitivity_<timestamp>.csv)
    --verbose: Enable verbose logging

Output CSV columns:
    objective, cap, shock, top_name, top_sharpe, top_maxdd, top_score
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import csv
import subprocess
import json
import tempfile

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))

import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run sensitivity grid across objectives × caps × shocks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--objectives",
        type=str,
        default="balanced,growth",
        help="Comma-separated objectives (default: balanced,growth)"
    )
    parser.add_argument(
        "--caps",
        type=str,
        default="0.20,0.25,0.30,0.35",
        help="Comma-separated satellite caps (default: 0.20,0.25,0.30,0.35)"
    )
    parser.add_argument(
        "--shocks",
        type=str,
        default="none",
        help="Comma-separated shocks (default: none)"
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=6,
        help="Number of candidates per run (default: 6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV path (default: dev/artifacts/sensitivity_<timestamp>.csv)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def run_single_scenario(
    objective: str,
    cap: float,
    shock: str,
    n_candidates: int,
    seed: int,
    verbose: bool = False
) -> dict:
    """
    Run a single scenario using run_scenarios.py and extract top-1 result.
    
    Returns:
        Dict with keys: top_name, top_sharpe, top_maxdd, top_score
    """
    # Create temp output path
    with tempfile.TemporaryDirectory() as tmpdir:
        out_stem = Path(tmpdir) / "scenario"
        
        # Build command
        cmd = [
            sys.executable,
            str(ROOT / "dev" / "run_scenarios.py"),
            "--objective", objective,
            "--n-candidates", str(n_candidates),
            "--seed", str(seed),
            "--out", str(out_stem)
        ]
        
        # Add shock if not "none"
        if shock != "none":
            cmd.extend(["--shock", shock])
        
        # Add satellite cap (use custom flag)
        cmd.extend(["--satellite-caps", str(cap)])
        
        if verbose:
            print(f"  Running: {objective} | cap={cap:.0%} | shock={shock}", file=sys.stderr)
        
        # Run subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            if verbose:
                print(f"  Error: {result.stderr[:200]}", file=sys.stderr)
            return {
                "top_name": "ERROR",
                "top_sharpe": None,
                "top_maxdd": None,
                "top_score": None
            }
        
        # Read JSON output
        json_path = Path(str(out_stem) + ".json")
        if not json_path.exists():
            return {
                "top_name": "MISSING",
                "top_sharpe": None,
                "top_maxdd": None,
                "top_score": None
            }
        
        with open(json_path) as f:
            data = json.load(f)
        
        # Extract top candidate
        if not data.get("candidates"):
            return {
                "top_name": "EMPTY",
                "top_sharpe": None,
                "top_maxdd": None,
                "top_score": None
            }
        
        top = data["candidates"][0]
        
        # Get 5y metrics (or full metrics if horizon not available)
        metrics = top.get("metrics", {})
        if isinstance(metrics, dict) and "5y" in metrics:
            metrics_5y = metrics["5y"]
        elif isinstance(metrics, dict) and "full" in metrics:
            metrics_5y = metrics["full"]
        else:
            metrics_5y = metrics
        
        return {
            "top_name": top.get("name", "Unknown"),
            "top_sharpe": metrics_5y.get("Sharpe"),
            "top_maxdd": metrics_5y.get("MaxDD"),
            "top_score": top.get("score")
        }


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse parameters
    objectives = [o.strip() for o in args.objectives.split(",")]
    caps = [float(c.strip()) for c in args.caps.split(",")]
    shocks = [s.strip() for s in args.shocks.split(",")]
    
    # Generate output path
    if args.out:
        out_path = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "dev" / "artifacts" / f"sensitivity_{timestamp}.csv"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Sensitivity Grid Runner", file=sys.stderr)
    print(f"Objectives: {objectives}", file=sys.stderr)
    print(f"Caps: {caps}", file=sys.stderr)
    print(f"Shocks: {shocks}", file=sys.stderr)
    print(f"N-candidates: {args.n_candidates}", file=sys.stderr)
    print(f"Seed: {args.seed}", file=sys.stderr)
    print(f"", file=sys.stderr)
    
    # Run grid
    results = []
    total = len(objectives) * len(caps) * len(shocks)
    count = 0
    
    for obj in objectives:
        for cap in caps:
            for shock in shocks:
                count += 1
                print(f"[{count}/{total}] {obj} | cap={cap:.0%} | shock={shock}", file=sys.stderr)
                
                top_result = run_single_scenario(
                    objective=obj,
                    cap=cap,
                    shock=shock,
                    n_candidates=args.n_candidates,
                    seed=args.seed,
                    verbose=args.verbose
                )
                
                results.append({
                    "objective": obj,
                    "cap": cap,
                    "shock": shock,
                    **top_result
                })
    
    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "objective", "cap", "shock",
            "top_name", "top_sharpe", "top_maxdd", "top_score"
        ])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nWrote {len(results)} results to {out_path}", file=sys.stderr)
    
    # Print pivot summary
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"SENSITIVITY SUMMARY", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    
    df = pd.DataFrame(results)
    
    # Pivot: objectives × shocks, aggregated by mean score
    if "top_score" in df.columns and df["top_score"].notna().any():
        pivot = df.pivot_table(
            values="top_score",
            index="objective",
            columns="shock",
            aggfunc="mean"
        )
        print(f"\nMean Top Score by Objective × Shock:", file=sys.stderr)
        print(pivot.to_string(), file=sys.stderr)
    
    # Count by optimizer name
    if "top_name" in df.columns:
        print(f"\nTop Optimizer Distribution:", file=sys.stderr)
        optimizer_counts = df["top_name"].str.split(" - ").str[0].value_counts()
        print(optimizer_counts.to_string(), file=sys.stderr)
    
    print(f"\n{'='*80}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

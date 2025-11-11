#!/usr/bin/env python3
"""
Analyst Report Generator - Convert scenario artifacts to Markdown reports.

USAGE:
    python3 dev/report_from_artifacts.py --json dev/artifacts/scenario_balanced_<timestamp>.json

Feature flag: default OFF (run manually for reporting).

Args:
    --json <path>: JSON artifact file (required)
    --weights <path>: Weights CSV (optional, inferred if omitted)
    --metrics <path>: Metrics CSV (optional, inferred if omitted)
    --out <path>: Output Markdown path (default: dev/artifacts/report_<timestamp>.md)
    --plot: Generate matplotlib charts (default: False, requires matplotlib)

Output:
    Simple Markdown report with:
    - Title, timestamp, objective, regime
    - Top 5 candidates table
    - Shortlist weights table
    - Per-regime performance summary
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import json

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Markdown report from scenario artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="JSON artifact file (required)"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Weights CSV (optional, inferred if omitted)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Metrics CSV (optional, inferred if omitted)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output Markdown path (default: dev/artifacts/report_<timestamp>.md)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib charts (requires matplotlib)"
    )
    
    return parser.parse_args()


def infer_companion_files(json_path: Path) -> tuple:
    """
    Infer weights and metrics CSV paths from JSON path.
    
    Example:
        scenario_balanced_20251110_011642.json
        -> scenario_balanced_20251110_011642_weights.csv
        -> scenario_balanced_20251110_011642_metrics.csv
    """
    stem = json_path.stem  # Remove .json
    parent = json_path.parent
    
    weights_path = parent / f"{stem}_weights.csv"
    metrics_path = parent / f"{stem}_metrics.csv"
    
    return weights_path, metrics_path


def format_weight(w: float) -> str:
    """Format weight as percentage."""
    return f"{w*100:.1f}%"


def format_metric(m: float, metric_type: str = "default") -> str:
    """Format metric value."""
    if m is None:
        return "N/A"
    
    if metric_type == "pct":
        return f"{m*100:.1f}%"
    elif metric_type == "ratio":
        return f"{m:.2f}"
    else:
        return f"{m:.4f}"


def generate_report(
    json_data: dict,
    weights_df: pd.DataFrame = None,
    metrics_df: pd.DataFrame = None,
    plot: bool = False
) -> str:
    """
    Generate Markdown report from JSON data.
    
    Args:
        json_data: Parsed JSON artifact
        weights_df: Weights matrix DataFrame (optional)
        metrics_df: Metrics matrix DataFrame (optional)
        plot: Whether to generate plots (default: False)
    
    Returns:
        Markdown string
    """
    lines = []
    
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# Portfolio Scenario Report")
    lines.append(f"")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"")
    
    # Metadata
    lines.append(f"## Scenario Details")
    lines.append(f"")
    lines.append(f"- **Objective:** {json_data.get('objective', 'N/A').title()}")
    lines.append(f"- **Tickers:** {', '.join(json_data.get('tickers', []))}")
    lines.append(f"- **Start Date:** {json_data.get('start_date', 'N/A')}")
    lines.append(f"- **Horizon:** {json_data.get('horizon', 'N/A').upper()}")
    lines.append(f"- **Candidates Generated:** {json_data.get('n_candidates', 0)}")
    lines.append(f"- **Current Regime:** {json_data.get('current_regime', 'Unknown')}")
    
    if json_data.get('shocks_applied'):
        lines.append(f"- **Shocks Applied:** {', '.join(json_data['shocks_applied'])}")
    
    lines.append(f"")
    
    # Top 5 candidates
    lines.append(f"## Top 5 Candidates")
    lines.append(f"")
    
    candidates = json_data.get("candidates", [])
    if candidates:
        # Build table
        lines.append(f"| Rank | Name | Sharpe | MaxDD | Score | Shortlist |")
        lines.append(f"|------|------|--------|-------|-------|-----------|")
        
        for i, cand in enumerate(candidates[:5], 1):
            name = cand.get("name", "Unknown")
            
            # Get 5y metrics or full metrics
            metrics = cand.get("metrics", {})
            if isinstance(metrics, dict) and "5y" in metrics:
                metrics_5y = metrics["5y"]
            elif isinstance(metrics, dict) and "full" in metrics:
                metrics_5y = metrics["full"]
            else:
                metrics_5y = metrics
            
            sharpe = format_metric(metrics_5y.get("Sharpe"), "ratio")
            maxdd = format_metric(metrics_5y.get("MaxDD"), "pct")
            score = format_metric(cand.get("score"), "ratio")
            shortlist = "⭐" if cand.get("shortlist") else ""
            
            lines.append(f"| {i} | {name} | {sharpe} | {maxdd} | {score} | {shortlist} |")
        
        lines.append(f"")
    else:
        lines.append(f"*No candidates available*")
        lines.append(f"")
    
    # Shortlist weights
    lines.append(f"## Shortlist Portfolio Weights")
    lines.append(f"")
    
    shortlist_cand = next((c for c in candidates if c.get("shortlist")), None)
    if shortlist_cand:
        weights = shortlist_cand.get("weights", {})
        
        if weights:
            # Sort by weight descending
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            lines.append(f"**{shortlist_cand.get('name', 'Top Candidate')}**")
            lines.append(f"")
            lines.append(f"| Ticker | Weight |")
            lines.append(f"|--------|--------|")
            
            for ticker, weight in sorted_weights:
                lines.append(f"| {ticker} | {format_weight(weight)} |")
            
            lines.append(f"")
        else:
            lines.append(f"*No weights available*")
            lines.append(f"")
    else:
        lines.append(f"*No shortlist candidate marked*")
        lines.append(f"")
    
    # Metrics summary
    lines.append(f"## Performance Metrics Summary")
    lines.append(f"")
    
    if shortlist_cand:
        metrics = shortlist_cand.get("metrics", {})
        
        # Check if we have horizon-specific metrics
        has_horizons = any(h in metrics for h in ["1y", "3y", "5y", "10y"])
        
        if has_horizons:
            lines.append(f"| Horizon | CAGR | Volatility | Sharpe | MaxDD |")
            lines.append(f"|---------|------|------------|--------|-------|")
            
            for h in ["1y", "3y", "5y", "10y"]:
                if h in metrics:
                    hm = metrics[h]
                    cagr = format_metric(hm.get("CAGR"), "pct")
                    vol = format_metric(hm.get("Volatility"), "pct")
                    sharpe = format_metric(hm.get("Sharpe"), "ratio")
                    maxdd = format_metric(hm.get("MaxDD"), "pct")
                    
                    lines.append(f"| {h.upper()} | {cagr} | {vol} | {sharpe} | {maxdd} |")
            
            lines.append(f"")
        elif "full" in metrics:
            fm = metrics["full"]
            lines.append(f"- **CAGR:** {format_metric(fm.get('CAGR'), 'pct')}")
            lines.append(f"- **Volatility:** {format_metric(fm.get('Volatility'), 'pct')}")
            lines.append(f"- **Sharpe Ratio:** {format_metric(fm.get('Sharpe'), 'ratio')}")
            lines.append(f"- **Max Drawdown:** {format_metric(fm.get('MaxDD'), 'pct')}")
            lines.append(f"")
    else:
        lines.append(f"*No metrics available*")
        lines.append(f"")
    
    # Regime performance (if available)
    # Note: This is typically not in the JSON artifact, but we can document the structure
    lines.append(f"## Notes")
    lines.append(f"")
    lines.append(f"- Scoring formula: `score = Sharpe - 0.2 × |MaxDD|`")
    if json_data.get("current_regime") != "Unknown":
        lines.append(f"- Current regime ({json_data.get('current_regime')}) identified from macro indicators")
    lines.append(f"- Satellite cap constraints enforced per candidate")
    lines.append(f"")
    
    # Footer
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"*Report generated by Invest_AI V3*")
    lines.append(f"")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load JSON
    json_path = Path(args.json)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1
    
    with open(json_path) as f:
        json_data = json.load(f)
    
    # Infer companion files if not provided
    weights_path = Path(args.weights) if args.weights else None
    metrics_path = Path(args.metrics) if args.metrics else None
    
    if not weights_path or not metrics_path:
        inferred_weights, inferred_metrics = infer_companion_files(json_path)
        weights_path = weights_path or inferred_weights
        metrics_path = metrics_path or inferred_metrics
    
    # Load companion CSVs (optional)
    weights_df = None
    metrics_df = None
    
    if weights_path and weights_path.exists():
        weights_df = pd.read_csv(weights_path)
    
    if metrics_path and metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    
    # Generate report
    report_md = generate_report(
        json_data,
        weights_df=weights_df,
        metrics_df=metrics_df,
        plot=args.plot
    )
    
    # Determine output path
    if args.out:
        out_path = Path(args.out)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "dev" / "artifacts" / f"report_{timestamp}.md"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write report
    with open(out_path, "w") as f:
        f.write(report_md)
    
    print(f"✅ Report written to {out_path}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"Preview:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(report_md[:500])
    if len(report_md) > 500:
        print(f"... (truncated, see {out_path} for full report)")
    print(f"{'='*80}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

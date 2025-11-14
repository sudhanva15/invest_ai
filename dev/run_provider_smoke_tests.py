#!/usr/bin/env python3
"""
Provider smoke tests for Invest_AI data layer.

Runs a series of provider fetch tests and prints summary stats.
Exits 0 if >=35 tickers validated, else 1.

Usage:
  .venv/bin/python dev/run_provider_smoke_tests.py
"""
import sys
import json
from datetime import date
from pathlib import Path
import warnings

# Treat warnings as errors to simulate strict CI
warnings.filterwarnings("error")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sources.fetch import fetch_price_history, fetch_multiple
from core.data_sources.tiingo import is_tiingo_enabled
from core.universe_validate import build_validated_universe


def test_provider_fetches():
    """Run small fetch tests on each provider/fallback path."""
    results = {
        "tiingo": None,
        "stooq": None,
        "yfinance": None,
    }

    # Tiingo
    tiingo_ok = False
    try:
        if is_tiingo_enabled(ping=False):
            df = fetch_price_history("SPY", start=date(2020,1,1), end=date(2023,12,31))
            tiingo_ok = not df.empty
    except Exception:
        tiingo_ok = False
    results["tiingo"] = tiingo_ok

    # Stooq (cache-only symbols that we know exist)
    stooq_ok = False
    try:
        df = fetch_price_history("VTI", start=date(2020,1,1), end=date(2023,12,31))
        stooq_ok = not df.empty
    except Exception:
        stooq_ok = False
    results["stooq"] = stooq_ok

    # yfinance fallback (should not crash; may be empty in some envs)
    yf_ok = False
    try:
        df = fetch_price_history("VNQ", start=date(2020,1,1), end=date(2023,12,31))
        # Accept empty but not crash
        yf_ok = df is not None
    except Exception:
        yf_ok = False
    results["yfinance"] = yf_ok

    return results


def test_batch_and_cache():
    """Batch fetch and ensure cache metadata recorded."""
    syms = ["SPY","QQQ","BND","TLT"]
    prices, metadata = fetch_multiple(syms, start=date(2020,1,1), end=date(2023,12,31))
    return {
        "shape": tuple(prices.shape),
        "providers": {k: v.get("provider") for k, v in metadata.items()},
    }


def run_universe_build():
    valid, path = build_validated_universe()
    return len(valid), path


def main():
    print("Running provider smoke tests...\n")
    prov = test_provider_fetches()
    print("Provider quick checks:")
    for k, v in prov.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")

    batch = test_batch_and_cache()
    print("\nBatch fetch:")
    print(f"  shape: {batch['shape']}")
    print(f"  providers: {batch['providers']}")

    valid_count, snap_path = run_universe_build()
    print("\nUniverse build:")
    print(f"  valid_count: {valid_count}")
    print(f"  snapshot: {snap_path}")

    # Enhanced breakdown: provider, asset class, tier from snapshot
    try:
        with open(snap_path) as f:
            payload = json.load(f)
        records = payload.get("records", {})
        valid_syms = set(payload.get("valid_symbols", []))
        metrics = payload.get("metrics", {})
        
        prov_counts = {"tiingo": 0, "stooq": 0, "yfinance": 0, "unknown": 0}
        tiingo_symbols = []
        for sym, rec in records.items():
            if sym in valid_syms:
                prov = (rec.get("provider") or "unknown").lower()
                if prov not in prov_counts:
                    prov = "unknown"
                prov_counts[prov] += 1
                if prov == "tiingo":
                    tiingo_symbols.append(sym)
        
        print("\nProvider coverage:")
        for k in ["tiingo","stooq","yfinance","unknown"]:
            print(f"  {k}: {prov_counts[k]} valid")
        
        # Show first 5 Tiingo symbols for diagnostic visibility
        if tiingo_symbols:
            print(f"\n  Tiingo symbols (first 5): {', '.join(tiingo_symbols[:5])}")
        
        # Asset class & tier breakdown
        asset_class_counts = metrics.get("asset_class_counts", {})
        tier_counts = metrics.get("tier_counts", {})
        
        if asset_class_counts:
            print("\nAsset class breakdown:")
            for cls, cnt in sorted(asset_class_counts.items()):
                print(f"  {cls}: {cnt} ETFs")
        
        if tier_counts:
            print("\nTier breakdown:")
            for tier, cnt in sorted(tier_counts.items()):
                print(f"  {tier}: {cnt} ETFs")
    except Exception:
        pass

    # Exit status
    if valid_count >= 35:
        print("\n✓ Smoke tests PASS (coverage target met)")
        return 0
    else:
        print("\n⚠ Smoke tests PARTIAL (coverage below target; enable Tiingo to increase)")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Warning as w:
        # Treat warnings as failure in smoke
        print(f"Warning raised as error: {w}")
        sys.exit(1)

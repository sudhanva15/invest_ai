#!/usr/bin/env python3
"""
Backfill Stooq cache by downloading missing ETF data from Yahoo Finance.

This script:
1. Reads assets_catalog.json to get all ETF symbols (or --assets list)
2. Checks which ones are missing from data/raw/
3. Downloads from a chosen provider (tiingo|yfinance) with safe fallbacks
4. Saves to data/raw/{SYMBOL}.csv in Stooq-compatible format

Run:
    python dev/backfill_cache.py [--limit N] [--start YYYY-MM-DD] [--assets SPY,QQQ] [--provider-only tiingo]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import logging
import pandas as pd


def load_catalog_symbols(catalog_path: Path) -> list[str]:
    """Load all symbols from assets catalog."""
    with open(catalog_path) as f:
        data = json.load(f)
    
    symbols = []
    for asset in data.get("assets", []):
        sym = asset.get("symbol", "").strip().upper()
        if sym:
            symbols.append(sym)
    
    return sorted(symbols)


def get_missing_symbols(symbols: list[str], cache_dir: Path) -> list[str]:
    """Return symbols that don't have cache files."""
    missing = []
    for sym in symbols:
        cache_file = cache_dir / f"{sym}.csv"
        if not cache_file.exists():
            missing.append(sym)
    return missing


def download_with_yfinance_workaround(symbol: str, start: str = "2010-01-01") -> pd.DataFrame:
    """
    Download data from Yahoo Finance with timezone error workaround.
    
    The YFTzMissingError occurs when yfinance can't determine timezone.
    Workaround: Use Ticker.history() instead of yf.download() and handle errors gracefully.
    """
    try:
        import yfinance as yf
        
        ticker = yf.Ticker(symbol)
        
        # Use history() method which is more robust than download()
        df = ticker.history(start=start, auto_adjust=False)
        
        if df.empty:
            print(f"  ✗ {symbol}: Empty response from yfinance")
            return pd.DataFrame()
        
        # Normalize to Stooq schema
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open", 
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        
        # Convert date to string format
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        # Select columns in Stooq order
        columns = ["date", "open", "high", "low", "close", "volume"]
        df = df[columns]
        
        print(f"  ✓ {symbol}: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
        return df
    
    except Exception as e:
        print(f"  ✗ {symbol}: {type(e).__name__}: {str(e)[:80]}")
        return pd.DataFrame()

def download_with_tiingo(symbol: str, start: str = "2010-01-01") -> pd.DataFrame:
    """Download data from Tiingo (if enabled) and normalize to Stooq CSV schema."""
    try:
        from core.data_sources.tiingo import fetch_daily, is_tiingo_enabled
        if not is_tiingo_enabled(ping=False):
            print("  ✗ Tiingo disabled (no API key)")
            return pd.DataFrame()
        df = fetch_daily(symbol, start=start)
        if df is None or df.empty:
            print(f"  ✗ {symbol}: Empty response from Tiingo")
            return pd.DataFrame()
        # Convert to Stooq CSV schema order
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
        out = out.rename(columns={"adj_close": "close"}) if "close" not in out.columns and "adj_close" in out.columns else out
        cols = [c for c in ["date","open","high","low","close","volume"] if c in out.columns]
        if set(["date","close"]).issubset(set(cols)):
            out = out[cols]
            print(f"  ✓ {symbol}: {len(out)} rows from Tiingo")
            return out
        print(f"  ✗ {symbol}: Missing required columns from Tiingo response")
        return pd.DataFrame()
    except Exception as e:
        print(f"  ✗ {symbol}: Tiingo error {type(e).__name__}: {str(e)[:80]}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Backfill Stooq cache with missing ETF data")
    parser.add_argument("--limit", type=int, help="Limit number of symbols to download")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    parser.add_argument("--assets", help="Comma-separated list of tickers to backfill (overrides catalog)")
    parser.add_argument("--provider-only", choices=["tiingo","yfinance"], help="Use only the specified provider")
    args = parser.parse_args()
    
    # Paths
    catalog_path = Path("config/assets_catalog.json")
    cache_dir = Path("data/raw")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load symbols
    if args.assets:
        symbols = [s.strip().upper() for s in args.assets.split(",") if s.strip()]
        print(f"Using provided assets list: {len(symbols)} symbols")
    else:
        print("Loading catalog...")
        symbols = load_catalog_symbols(catalog_path)
        print(f"  Found {len(symbols)} symbols in catalog")
    
    # Find missing
    if args.force:
        missing = symbols
        print(f"  Force mode: will download all {len(missing)} symbols")
    else:
        missing = get_missing_symbols(symbols, cache_dir)
        print(f"  Missing from cache: {len(missing)} symbols")
    
    if not missing:
        print("\n✓ Cache is complete!")
        return 0
    
    # Apply limit if specified
    if args.limit:
        missing = missing[:args.limit]
        print(f"  Limited to first {len(missing)} symbols")
    
    # Download
    print(f"\nDownloading {len(missing)} symbols...")
    success_count = 0
    
    for i, symbol in enumerate(missing, 1):
        print(f"[{i}/{len(missing)}] {symbol}...")
        # Provider selection
        df = pd.DataFrame()
        if args.provider_only == "tiingo":
            df = download_with_tiingo(symbol, start=args.start)
        elif args.provider_only == "yfinance" or df.empty:
            # Use yfinance as fallback/default
            df = download_with_yfinance_workaround(symbol, start=args.start)
        
        if not df.empty:
            # Save to cache
            cache_file = cache_dir / f"{symbol}.csv"
            df.to_csv(cache_file, index=False)
            success_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"✓ Downloaded: {success_count}/{len(missing)}")
    print(f"✗ Failed: {len(missing) - success_count}/{len(missing)}")
    print("=" * 60)
    
    # Re-check coverage
    remaining = get_missing_symbols(symbols, cache_dir)
    print(f"\nCache coverage: {len(symbols) - len(remaining)}/{len(symbols)} ({100 * (len(symbols) - len(remaining)) / len(symbols):.1f}%)")
    
    return 0


if __name__ == "__main__":
    exit(main())

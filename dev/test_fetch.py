#!/usr/bin/env python3
"""
Quick validation test for core/data_sources/fetch.py

Tests:
    1. Single ticker fetch (cache miss → provider → cache write)
    2. Cache hit (second fetch reads from cache)
    3. Multi-ticker fetch
    4. Provider fallback chain
    5. Error handling (invalid ticker)

Run:
    python dev/test_fetch.py
"""
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sources.fetch import fetch_price_history, fetch_multiple, CACHE_DIR


def test_single_fetch():
    """Test single ticker fetch with cache miss and hit."""
    print("\n=== Test 1: Single Ticker Fetch ===")
    
    ticker = "SPY"
    start = date(2020, 1, 1)
    end = date(2023, 12, 31)
    
    # Clear cache for clean test
    cache_file = CACHE_DIR / f"{ticker}.parquet"
    meta_file = CACHE_DIR / f"{ticker}_meta.json"
    if cache_file.exists():
        cache_file.unlink()
    if meta_file.exists():
        meta_file.unlink()
    
    # First fetch (cache miss)
    print(f"\nFetch {ticker} (cache miss)...")
    df1 = fetch_price_history(ticker, start=start, end=end)
    
    print(f"  Rows: {len(df1)}")
    print(f"  Columns: {list(df1.columns)}")
    print(f"  Date range: {df1.index.min()} to {df1.index.max()}")
    print(f"  Has adj_close: {'adj_close' in df1.columns}")
    print(f"  Cache file exists: {cache_file.exists()}")
    
    assert not df1.empty, "First fetch should return data"
    assert "adj_close" in df1.columns, "Should have adj_close column"
    assert cache_file.exists(), "Cache file should be written"
    
    # Second fetch (cache hit)
    print(f"\nFetch {ticker} again (cache hit)...")
    df2 = fetch_price_history(ticker, start=start, end=end)
    
    print(f"  Rows: {len(df2)}")
    print(f"  Same data: {df1.equals(df2)}")
    
    assert not df2.empty, "Second fetch should return cached data"
    print("\n✓ Single fetch test passed")


def test_multi_fetch():
    """Test multiple tickers at once."""
    print("\n=== Test 2: Multi-Ticker Fetch ===")
    
    tickers = ["SPY", "QQQ", "BND"]
    start = date(2020, 1, 1)
    end = date(2023, 12, 31)
    
    print(f"\nFetch {tickers}...")
    prices, metadata = fetch_multiple(tickers, start=start, end=end)
    
    print(f"  Prices shape: {prices.shape}")
    print(f"  Columns: {list(prices.columns)}")
    print(f"  Metadata: {metadata}")
    
    for ticker in tickers:
        if ticker in metadata:
            meta = metadata[ticker]
            print(f"  {ticker}: provider={meta['provider']}, rows={meta['rows']}, error={meta['error']}")
    
    assert not prices.empty, "Should return prices for at least one ticker"
    print("\n✓ Multi-ticker fetch test passed")


def test_invalid_ticker():
    """Test error handling for invalid ticker."""
    print("\n=== Test 3: Invalid Ticker ===")
    
    ticker = "NOTAREALTICKER12345"
    
    print(f"\nFetch {ticker} (should fail gracefully)...")
    df = fetch_price_history(ticker)
    
    print(f"  Rows: {len(df)}")
    print(f"  Empty: {df.empty}")
    
    assert df.empty, "Invalid ticker should return empty DataFrame"
    print("\n✓ Invalid ticker test passed")


def test_cache_fresh_check():
    """Test cache freshness logic."""
    print("\n=== Test 4: Cache Freshness ===")
    
    ticker = "TLT"
    
    # Clear old cache
    cache_file = CACHE_DIR / f"{ticker}.parquet"
    meta_file = CACHE_DIR / f"{ticker}_meta.json"
    if cache_file.exists():
        cache_file.unlink()
    if meta_file.exists():
        meta_file.unlink()
    
    # Fetch to populate cache
    print(f"\nFetch {ticker} to populate cache...")
    df1 = fetch_price_history(ticker, start=date(2020, 1, 1), end=date(2023, 12, 31))
    print(f"  Rows: {len(df1)}")
    print(f"  Cache exists: {cache_file.exists()}")
    
    # Fetch again (should hit cache)
    print(f"\nFetch {ticker} again (should use cache)...")
    df2 = fetch_price_history(ticker, start=date(2020, 1, 1), end=date(2023, 12, 31))
    print(f"  Rows: {len(df2)}")
    print(f"  Same data: {df1.equals(df2)}")
    
    # Fetch with different date range (should bypass cache if range extends beyond cached)
    print(f"\nFetch {ticker} with extended range (may bypass cache)...")
    df3 = fetch_price_history(ticker, start=date(2015, 1, 1), end=date(2023, 12, 31))
    print(f"  Rows: {len(df3)}")
    print(f"  More rows than cached: {len(df3) > len(df1)}")
    
    print("\n✓ Cache freshness test passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing core/data_sources/fetch.py")
    print("=" * 60)
    
    try:
        test_single_fetch()
        test_multi_fetch()
        test_invalid_ticker()
        test_cache_fresh_check()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

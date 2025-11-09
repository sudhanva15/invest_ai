#!/usr/bin/env python3
"""Quick diagnostic for data loading"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.utils.env_tools import load_env_once
load_env_once('.env')

print("Testing data fetch...")

# Test 1: Direct stooq
try:
    from core.data_sources import stooq
    df = stooq.fetch_daily("SPY")
    print(f"✓ Stooq direct: {len(df)} rows" if df is not None and len(df) > 0 else "✗ Stooq: empty")
except Exception as e:
    print(f"✗ Stooq failed: {e}")

# Test 2: Router
try:
    from core.data_sources.router_smart import fetch_union
    df, prov = fetch_union("SPY", return_provenance=True)
    print(f"✓ Router: {len(df)} rows, prov={prov}" if df is not None and len(df) > 0 else f"✗ Router: empty, prov={prov}")
except Exception as e:
    print(f"✗ Router failed: {e}")

# Test 3: get_prices
try:
    from core.data_ingestion import get_prices
    df = get_prices(["SPY"])
    print(f"✓ get_prices: {df.shape}" if not df.empty else "✗ get_prices: empty")
    if not df.empty:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index: {df.index[0]} to {df.index[-1]}")
except Exception as e:
    print(f"✗ get_prices failed: {e}")
    import traceback
    traceback.print_exc()

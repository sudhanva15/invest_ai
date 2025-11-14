#!/usr/bin/env python3
"""
Quick Tiingo ETF debug utility.

Checks a set of ETFs against Tiingo's daily prices endpoint and prints a
clear per-ticker status including HTTP status code, row count, and date span.

Usage:
  .venv/bin/python dev/debug_tiingo_etfs.py

Requires TIINGO_API_KEY via env or config (apis.tiingo_api_key).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import requests
import pandas as pd
import io

# Make repo importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data_sources.tiingo import _get_token_layered

ETFS = [
    "SPY","IVV","VOO","VTI","QQQ","BND","AGG","GLD","VNQ",
]

BASE = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"


def normalize_tiingo_csv(text: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.StringIO(text))
        if df.empty:
            return pd.DataFrame()
        # date, close, high, low, open, volume, adjClose, ...
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].notna()].sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()


def check_symbol(sym: str, token: str, start: str | None = None) -> dict:
    """
    Check a symbol via Tiingo API.
    
    KEY CHANGE: start=None by default to fetch FULL HISTORY.
    This allows Tiingo to return 10-20+ years of data instead of truncating to 2010.
    """
    url = BASE.format(ticker=sym.lower())
    params = {"format": "csv", "token": token}
    # Only add startDate if explicitly provided; otherwise get full history
    if start:
        params["startDate"] = start
    
    try:
        r = requests.get(url, params=params, timeout=20)
        status = r.status_code
        if status != 200:
            return {
                "symbol": sym,
                "http": status,
                "ok": False,
                "rows": 0,
                "reason": r.text[:200].replace("\n", " ") if r.text else f"HTTP {status}",
            }
        df = normalize_tiingo_csv(r.text)
        if df.empty:
            return {"symbol": sym, "http": status, "ok": False, "rows": 0, "reason": "empty"}
        
        rows = len(df)
        # Flag suspiciously small responses
        if rows < 100:
            return {
                "symbol": sym,
                "http": status,
                "ok": False,
                "rows": rows,
                "reason": f"too_small ({rows} rows < 100)",
            }
        
        return {
            "symbol": sym,
            "http": status,
            "ok": True,
            "rows": rows,
            "start": df["date"].min().date().isoformat(),
            "end": df["date"].max().date().isoformat(),
        }
    except Exception as e:
        return {"symbol": sym, "http": None, "ok": False, "rows": 0, "reason": f"{type(e).__name__}: {e}"}


def main():
    token = _get_token_layered()
    if not token:
        print("❌ Tiingo: missing API key (TIINGO_API_KEY)")
        print("   Set in .env or config/credentials.env")
        return 1
    
    print("Debugging Tiingo ETFs (start=None for full history)...")
    print("=" * 80)
    
    oks = 0
    for sym in ETFS:
        res = check_symbol(sym, token, start=None)  # Get full history
        if res.get("ok"):
            oks += 1
            print(f"[{sym}] status=ok rows={res['rows']} first={res['start']} last={res['end']}")
        else:
            print(f"[{sym}] status=fail http={res['http']} reason={res.get('reason')}")
    
    print("=" * 80)
    print(f"\n📊 Summary: {oks}/{len(ETFS)} OK")
    
    if oks >= 6:
        print(f"✅ Tiingo is working well")
        return 0
    else:
        print(f"⚠️  Tiingo coverage low")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

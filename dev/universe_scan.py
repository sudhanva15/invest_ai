#!/usr/bin/env python3
"""
Scan the configured assets catalog, validate data coverage, and write a snapshot.
Also print a one-line summary for diagnostics.
"""
from pathlib import Path
import sys

# Ensure repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.universe_validate import build_validated_universe


def main():
    cat = Path("config/assets_catalog.json")
    out = Path("data/outputs/universe_snapshot.json")
    valid, snap = build_validated_universe(cat, start=None, end=None, snapshot_path=out)
    # Load payload to get totals
    import json
    payload = json.loads(Path(snap).read_text())
    total = payload.get("universe_size", 0)
    v = payload.get("valid_count", len(valid))
    d = payload.get("dropped_count", total - v)
    print(f"Universe: {total} tickers (Valid: {v}, Dropped: {d})")
    print(f"Snapshot: {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import os, json, time
from datetime import datetime, timedelta
from typing import Dict
from sec_edgar_downloader import Downloader

CACHE_DIR = os.path.join("data","edgar")
os.makedirs(CACHE_DIR, exist_ok=True)

# Map of symbol->CIK should live in your catalog later; minimal helper here:
_CIK_INDEX = os.path.join(CACHE_DIR, "cik_index.json")

def _load_cik_index() -> Dict[str,str]:
    if os.path.exists(_CIK_INDEX):
        return json.load(open(_CIK_INDEX))
    return {}

def _save_cik_index(idx: Dict[str,str]):
    with open(_CIK_INDEX, "w") as f:
        json.dump(idx, f, indent=2)

def update_cik(symbol: str, cik: str):
    idx = _load_cik_index()
    idx[symbol.upper()] = cik
    _save_cik_index(idx)

def filings_snapshot(symbol: str, lookback_days: int=365) -> dict:
    """
    Download listing of 10-K/10-Q in lookback window and return counts+last dates.
    Note: sec-edgar-downloader stores raw files; we just summarize metadata.
    """
    idx = _load_cik_index()
    cik = idx.get(symbol.upper())
    if not cik:
        # Users can set manually for now (future: auto-resolve)
        return {"symbol": symbol, "error": "No CIK mapping. Call update_cik(symbol, cik)."}

    dl = Downloader()
    out = {"symbol": symbol, "cik": cik, "counts": {}, "last_dates": {}}
    cutoff = datetime.utcnow() - timedelta(days=lookback_days)

    for form in ["10-K", "10-Q", "8-K"]:
        # Download metadata only (no PDFs) by limiting to small page-size via downloader’s internal behavior
        try:
            # This call saves files under ./sec_edgar_filings; we summarize and ignore contents
            dl.get(form, cik, amount=5)  # small pull
        except Exception:
            pass

        filings_dir = os.path.join("sec_edgar_filings", cik, form.replace("-","_"))
        count, last_date = 0, None
        if os.path.isdir(filings_dir):
            for root, _, files in os.walk(filings_dir):
                for fn in files:
                    if fn.endswith(".txt") or fn.endswith(".html"):
                        # filenames include date like YYYY-MM-DD
                        parts = fn.split("_")
                        for p in parts:
                            if len(p)==10 and p[4]=="-" and p[7]=="-":
                                try:
                                    d = datetime.strptime(p, "%Y-%m-%d")
                                    if d >= cutoff:
                                        count += 1
                                        last_date = max(last_date, d) if last_date else d
                                except Exception:
                                    pass
        out["counts"][form] = count
        out["last_dates"][form] = last_date.isoformat() if last_date else None

    # Cache JSON
    cache_path = os.path.join(CACHE_DIR, f"{symbol.upper()}.json")
    with open(cache_path,"w") as f:
        json.dump(out, f, indent=2)
    return out

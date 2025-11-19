"""Utility helpers for consistent asset-class mapping.

Priority chain (Phase 3):
1. Universe snapshot records (if provided externally)
2. Explicit catalog entries (config/assets_catalog.json "assets" list)
3. Derived fallbacks (use "asset_class" then "class")
4. Heuristic symbol buckets (hard-coded common ETFs)

This centralizes logic previously duplicated in UI components so
engine receipts and portfolio payloads can inject stable
`asset_class` values for every symbol.
"""
from __future__ import annotations
from typing import Dict, Iterable, Optional

def _heuristic_class(symbol: str) -> str:
    s = symbol.upper()
    # Commodities / alternatives
    if s in {"GLD","IAU","GLDM"}: return "commodity"
    if s in {"DBC","PDBC","GSG","GCC"}: return "commodity"
    if s in {"VNQ","SCHH","IYR"}: return "reit"
    # Cash / bills
    if s in {"BIL","SHY"}: return "cash"
    # Bonds
    if s in {"AGG","BND","LQD","HYG","MUB","TLT","IEF","IEI","SHY","TIP","SCHP","BSV","BIV","VGIT","VGLT","VGSH","EDV","ZROZ","GOVT","IGSB","IGIB","JNK","BNDX","EMB"}: return "bond"
    # Equity broad / intl
    if s in {"SPY","VOO","IVV","VTI","ITOT","SCHB","SPTM","RSP","SPLG","SCHX"}: return "equity"
    if s in {"VEA","EFA","VXUS","VEU","ACWI","ACWX","IEFA","IEMG"}: return "equity"
    # Equity style / factor / growth/value
    if s in {"QQQ","QQQM","IWM","IJR","IJH","IWB","VUG","VTV","VBK","VBR","AVUV","MTUM","QUAL","USMV","SPLV","SCHD","VIG","VYM","DGRO","SPYG","SPYV","IWF","IWD","VLUE","SIZE"}: return "equity"
    # Equity sector ETFs (treat as equity bucket for diversification math)
    if s in {"XLF","XLK","XLY","XLP","XLV","XLI","XLU","XLE","XLRE","XLB","XLC"}: return "equity"
    # International country / EM single-country (still equity)
    if s in {"VWO","EEM","EWJ","INDA","EWU","EWQ","EWC","EWA","EWG","EWH","EWY","EWT","EWZ","EWS","EZA","EWW","EWM","EWD","EWL","EWI","EWP","VSS","SCZ"}: return "equity"
    return "unknown"

def build_symbol_metadata_map(
    catalog: Optional[dict],
    universe_records: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, str]]:
    """Return mapping: SYMBOL -> {asset_class: str, core_or_satellite: str}

    Parameters
    ----------
    catalog : dict | None
        Catalog loaded from assets_catalog.json (expects an "assets" list).
    universe_records : dict | None
        Optional snapshot objects having `.symbol`, `.asset_class`, `.core_or_satellite` attrs.
    """
    out: Dict[str, Dict[str, str]] = {}

    # 1. Universe snapshot records (highest precedence)
    if universe_records:
        for sym, rec in universe_records.items():
            try:
                s = str(getattr(rec, "symbol", sym)).upper()
                ac = str(getattr(rec, "asset_class", "") or getattr(rec, "class", "") or "unknown")
                cs = str(getattr(rec, "core_or_satellite", "satellite") or "satellite")
                out[s] = {"asset_class": ac, "core_or_satellite": cs}
            except Exception:
                continue

    # 2/3. Catalog assets (skip if already present from snapshot)
    try:
        assets = (catalog or {}).get("assets", []) if isinstance(catalog, dict) else []
        for a in assets:
            try:
                s = str(a.get("symbol", "")).upper()
                if not s:
                    continue
                if s in out:  # already from snapshot
                    continue
                ac = str(a.get("asset_class") or a.get("class") or _heuristic_class(s) or "unknown")
                cs = str(a.get("core_or_satellite", "satellite") or "satellite")
                out[s] = {"asset_class": ac, "core_or_satellite": cs}
            except Exception:
                continue
    except Exception:
        pass

    # 4. Heuristic fallbacks for any remaining referenced symbols (caller may merge later)
    return out

def map_asset_classes(symbols: Iterable[str], meta_map: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    """Convenience: return {symbol: asset_class} with heuristic fallback."""
    result: Dict[str, str] = {}
    for sym in symbols:
        s = str(sym).upper()
        ac = meta_map.get(s, {}).get("asset_class") or _heuristic_class(s) or "unknown"
        result[sym] = ac
    return result

__all__ = ["build_symbol_metadata_map", "map_asset_classes"]

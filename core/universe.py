"""Universe construction utilities.

Minimal curated universe expansion beyond ETFs with caps & eligibility.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

from core.env_tools import is_demo_mode
from core.demo_data import (
    get_demo_universe_frame,
    get_demo_universe_symbols,
    get_demo_caps,
    get_demo_constraints,
)

_DEF_PATH = Path("config/assets_catalog.json")
DEMO_MODE = is_demo_mode()

def load_assets_catalog(path: str | Path = _DEF_PATH) -> pd.DataFrame:
    if DEMO_MODE:
        df = get_demo_universe_frame().copy()
        caps = get_demo_caps()
        constraints = get_demo_constraints()
    else:
        p = Path(path)
        data = json.loads(p.read_text())
        df = pd.DataFrame(data["assets"]).copy()
        caps = data.get("caps", {})
        constraints = data.get("constraints", {})
    if "symbol" not in df.columns:
        raise ValueError("Catalog missing symbol column")
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df.drop_duplicates("symbol")
    df.index = df["symbol"]
    df.index.name = "symbol"
    # Attach caps & constraints metadata using df.attrs (recommended pandas approach)
    df.attrs["caps"] = caps
    df.attrs["constraints"] = constraints
    # Legacy fallback: set _caps/_constraints for older code, suppress warnings
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            df._caps = caps
            df._constraints = constraints
        except Exception:
            pass
    return df

def class_caps_from_catalog(df: pd.DataFrame) -> Dict[str, float]:
    caps = {}
    if hasattr(df, "_caps") and isinstance(getattr(df, "_caps"), dict):
        caps = df._caps.get("asset_class", {})
    elif isinstance(df.attrs.get("caps"), dict):
        caps = df.attrs["caps"].get("asset_class", {})
    return {k: float(v) for k,v in caps.items()}

def _is_single_stock(row) -> bool:
    return str(row.get("class", "")) == "single_stock"

FALLBACK_CORE = [
    "SPY","VTI","QQQ","DIA","IWM","VEA","VXUS","VWO",
    "TLT","IEF","AGG","BND","MUB","BIL","GLD","DBC"
]

def build_universe(
    objective: str,
    risk_profile: Optional[str],
    allow_single_names: bool,
    sectors_caps: Optional[Dict[str, float]],
    catalog: Optional[pd.DataFrame] = None,
) -> List[str]:
    """Filter catalog into ordered universe list.

    Ordering: core first (risk_bucket==core) then satellites.
    """
    if catalog is None:
        catalog = load_assets_catalog()
    df = catalog.copy()
    # Basic eligibility filter (retail only for now)
    df = df[df["eligibility"].isin(["retail"])].copy()
    # Objective coarse filters (light, tunable)
    if objective in {"preserve"}:
        df = df[~df["class"].isin(["single_stock"])]  # exclude single stocks
    if not allow_single_names:
        df = df[~df["class"].isin(["single_stock"])]
    # Sector caps enforcement (drop excess single names in over-cap sectors)
    if sectors_caps:
        keep_symbols = []
        for sector, cap in sectors_caps.items():
            sector_rows = df[df["sector"] == sector]
            if sector_rows.empty:
                continue
            # Sort deterministic by symbol then take up to int(cap * N)
            max_n = max(1, int(cap * len(sector_rows)))
            keep_symbols.extend(sorted(sector_rows.head(max_n)["symbol"]))
        if keep_symbols:
            df_sector_keep = df[df["sector"].isna() | df["symbol"].isin(keep_symbols)]
            df = df_sector_keep
    # Order core vs satellite
    core = df[df.get("risk_bucket")=="core"]["symbol"].tolist() if "risk_bucket" in df.columns else []
    sat  = df[df.get("risk_bucket")=="satellite"]["symbol"].tolist() if "risk_bucket" in df.columns else []
    universe = core + sat if (core or sat) else df["symbol"].tolist()

    # Fallback if universe too small
    if len(universe) < 12:
        # Keep only those present in catalog
        have = set(df["symbol"].tolist())
        universe = [s for s in FALLBACK_CORE if s in have]
    return universe

def get_validated_universe() -> List[str]:
    """
    Get the current validated universe from snapshot (runtime helper).
    
    This is a convenience wrapper around load_valid_universe() that:
    - Returns just the symbol list (most common use case)
    - Falls back to empty list if snapshot doesn't exist
    - Logs warnings but doesn't raise exceptions
    
    Returns:
        List of validated ticker symbols (e.g., ["SPY", "QQQ", ...])
        Empty list if snapshot doesn't exist or is invalid
    """
    if DEMO_MODE:
        return get_demo_universe_symbols()
    try:
        from core.universe_validate import load_valid_universe
        valid_syms, _, _ = load_valid_universe()
        return valid_syms
    except FileNotFoundError:
        import logging
        logging.getLogger(__name__).info(
            "Universe snapshot not found. Using empty universe. "
            "Run build_validated_universe() to create snapshot."
        )
        return []
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load universe snapshot: {e}")
        return []


__all__ = ["load_assets_catalog", "build_universe", "class_caps_from_catalog", "get_validated_universe"]

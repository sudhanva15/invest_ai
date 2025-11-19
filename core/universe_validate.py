"""Data-driven universe validation utilities.

Builds a validated tradable universe by probing data availability per symbol
using the centralized fetch pipeline (Stooq → Tiingo → yfinance fallback).

Outputs a snapshot JSON with validation metrics and per-symbol provider tracking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.universe import load_assets_catalog
from core.utils.env_tools import load_config
from core.data_sources.fetch import fetch_multiple
from core.preprocessing import compute_returns
from core.env_tools import is_demo_mode
from core.demo_data import (
    get_demo_universe_frame,
    load_demo_price_history,
)

DEMO_MODE = is_demo_mode()


@dataclass
class SymbolValidation:
    symbol: str
    start: Optional[str]
    end: Optional[str]
    history_years: float
    missing_pct: float
    n_obs: int
    provider: str
    valid: bool
    reason: Optional[str] = None
    # Provider diagnostics
    provider_error: Optional[str] = None
    # Liquidity & catalog fields
    median_volume: Optional[float] = None
    asset_class: Optional[str] = None
    region: Optional[str] = None
    core_or_satellite: Optional[str] = None


def _years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return float((end - start).days) / 365.25 if (pd.notna(start) and pd.notna(end)) else 0.0


def _demo_prices_wide(symbols: List[str]) -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = load_demo_price_history(sym)
        if df.empty:
            continue
        ser = df[["date", "adj_close"]].rename(columns={"adj_close": sym})
        frames.append(ser.set_index("date"))
    if not frames:
        return pd.DataFrame()
    prices = frames[0].copy()
    for frame in frames[1:]:
        prices = prices.join(frame, how="outer")
    return prices.sort_index()


def _demo_validate_catalog(
    catalog_df: pd.DataFrame,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    core_min_years: float = 10.0,
    sat_min_years: float = 7.0,
    max_missing_pct: float = 10.0,
    min_median_volume: float = 0.0,
) -> Tuple[Dict[str, SymbolValidation], List[str], List[str]]:
    symbols: List[str] = catalog_df["symbol"].astype(str).str.upper().tolist()
    records: Dict[str, SymbolValidation] = {}
    valid: List[str] = []
    dropped: List[str] = []

    for sym in symbols:
        try:
            df_full = load_demo_price_history(sym, start=start, end=end)
        except Exception:
            df_full = pd.DataFrame()

        cat_row = catalog_df[catalog_df["symbol"] == sym]
        asset_class = (
            str(cat_row["asset_class"].values[0])
            if not cat_row.empty and "asset_class" in cat_row.columns
            else None
        )
        region = (
            str(cat_row["region"].values[0])
            if not cat_row.empty and "region" in cat_row.columns
            else None
        )
        core_or_satellite = (
            str(cat_row["core_or_satellite"].values[0])
            if not cat_row.empty and "core_or_satellite" in cat_row.columns
            else None
        )

        if df_full.empty:
            rec = SymbolValidation(
                symbol=sym,
                start=None,
                end=None,
                history_years=0.0,
                missing_pct=100.0,
                n_obs=0,
                provider="demo",
                valid=False,
                reason="no_data",
                provider_error=None,
                median_volume=None,
                asset_class=asset_class,
                region=region,
                core_or_satellite=core_or_satellite,
            )
            records[sym] = rec
            dropped.append(sym)
            continue

        df_full = df_full.sort_values("date")
        idx = pd.to_datetime(df_full["date"], errors="coerce")
        st = idx.min()
        en = idx.max()
        years = _years_between(st, en)
        try:
            expected = len(pd.bdate_range(st, en)) if pd.notna(st) and pd.notna(en) else 0
        except Exception:
            expected = 0
        got = int(df_full.shape[0])
        missing_pct = float(max(0.0, (1.0 - (got / expected)) * 100.0)) if expected > 0 else 0.0
        median_volume = None
        if "volume" in df_full.columns:
            vol_series = df_full["volume"].dropna()
            if not vol_series.empty:
                median_volume = float(vol_series.median())

        tier = core_or_satellite if core_or_satellite else "satellite"
        min_years = core_min_years if tier and tier.lower() == "core" else sat_min_years
        history_ok = years >= (min_years - 0.01)
        sparsity_ok = missing_pct <= max_missing_pct
        liquidity_ok = (
            (median_volume is not None and median_volume >= min_median_volume)
            if min_median_volume
            else True
        )
        is_valid = history_ok and sparsity_ok and (got > 0) and liquidity_ok
        reason = None
        if not is_valid:
            r = []
            if not history_ok:
                r.append(f"years<{min_years:.0f}")
            if not sparsity_ok:
                r.append("missing>")
            if not liquidity_ok:
                r.append(f"vol<{min_median_volume}")
            if got <= 0:
                r.append("noobs")
            reason = ",".join(r) if r else "invalid"

        rec = SymbolValidation(
            symbol=sym,
            start=st.strftime("%Y-%m-%d") if pd.notna(st) else None,
            end=en.strftime("%Y-%m-%d") if pd.notna(en) else None,
            history_years=round(years, 2),
            missing_pct=round(missing_pct, 2),
            n_obs=got,
            provider="demo",
            valid=is_valid,
            reason=reason,
            provider_error=None,
            median_volume=int(median_volume) if median_volume is not None else None,
            asset_class=asset_class,
            region=region,
            core_or_satellite=core_or_satellite,
        )
        records[sym] = rec
        (valid if is_valid else dropped).append(sym)

    return records, valid, dropped


def _demo_metrics(
    catalog_df: pd.DataFrame,
    valid: List[str],
    records: Dict[str, SymbolValidation],
) -> Dict:
    metrics: Dict[str, Optional[float] | Dict | None] = {
        "avg_volatility": None,
        "avg_correlation": None,
        "sector_exposure": {},
        "asset_class_counts": {},
        "tier_counts": {},
        "history_years_distribution": None,
        "volume_distribution": None,
    }

    df_valid = catalog_df[catalog_df["symbol"].isin(valid)].copy()
    if not df_valid.empty:
        if "sector" in df_valid.columns:
            metrics["sector_exposure"] = (
                df_valid["sector"].fillna("unknown").value_counts().to_dict()
            )
        if "asset_class" in df_valid.columns:
            metrics["asset_class_counts"] = (
                df_valid["asset_class"].fillna("unknown").value_counts().to_dict()
            )
        if "core_or_satellite" in df_valid.columns:
            metrics["tier_counts"] = (
                df_valid["core_or_satellite"].fillna("unknown").value_counts().to_dict()
            )

    history_years = [rec.history_years for rec in records.values() if rec.valid]
    if history_years:
        metrics["history_years_distribution"] = {
            "min": round(min(history_years), 2),
            "median": round(float(np.median(history_years)), 2),
            "max": round(max(history_years), 2),
        }

    volumes = [rec.median_volume for rec in records.values() if rec.valid and rec.median_volume]
    if volumes:
        metrics["volume_distribution"] = {
            "min": int(min(volumes)),
            "median": int(float(np.median(volumes))),
            "max": int(max(volumes)),
        }

    prices = _demo_prices_wide(valid)
    if not prices.empty:
        try:
            rets = compute_returns(prices)
            vols = rets.std(ddof=0) * np.sqrt(252.0)
            metrics["avg_volatility"] = float(np.nanmean(vols.values)) if len(vols) else None
            corr = rets.corr()
            if corr is not None and corr.size > 0:
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                metrics["avg_correlation"] = float(np.nanmean(upper.values))
        except Exception:
            pass

    return metrics


def validate_catalog(
    catalog_df: pd.DataFrame,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    core_min_years: float = 10.0,
    sat_min_years: float = 7.0,
    max_missing_pct: float = 10.0,
    min_median_volume: float = 0.0,
) -> Tuple[Dict[str, SymbolValidation], List[str], List[str]]:
    """Validate each catalog symbol for data coverage using centralized fetch.

    Returns a mapping of symbol->validation record, plus (valid_symbols, dropped_symbols).
    """
    if DEMO_MODE:
        return _demo_validate_catalog(
            catalog_df,
            start=start,
            end=end,
            core_min_years=core_min_years,
            sat_min_years=sat_min_years,
            max_missing_pct=max_missing_pct,
            min_median_volume=min_median_volume,
        )
    symbols: List[str] = catalog_df["symbol"].astype(str).str.upper().tolist()
    records: Dict[str, SymbolValidation] = {}
    valid: List[str] = []
    dropped: List[str] = []

    # Convert date strings to date objects if provided
    start_date = pd.to_datetime(start).date() if start else None
    end_date = pd.to_datetime(end).date() if end else None

    # Fetch all prices + volume using new centralized fetch pipeline
    prices, metadata = fetch_multiple(symbols, start=start_date, end=end_date)
    
    # For volume computation, fetch full OHLCV frames per symbol
    # Only compute median_volume when volume column exists; gracefully handle missing data
    from core.data_sources.fetch import fetch_price_history
    volume_cache = {}
    for sym in symbols:
        try:
            df_full = fetch_price_history(sym, start=start_date, end=end_date)
            if not df_full.empty and "volume" in df_full.columns:
                # Only cache if volume column has meaningful data
                vol_col = df_full["volume"].dropna()
                if not vol_col.empty and (vol_col > 0).any():
                    volume_cache[sym] = vol_col
        except Exception:
            # Gracefully skip symbols that fail volume fetch
            pass

    for sym in symbols:
        meta = metadata.get(sym, {})
        provider = meta.get("provider", "none")
        provider_error = meta.get("error")
        
        # Get catalog metadata for this symbol
        cat_row = catalog_df[catalog_df["symbol"] == sym]
        asset_class = str(cat_row["asset_class"].values[0]) if not cat_row.empty and "asset_class" in cat_row.columns else None
        region = str(cat_row["region"].values[0]) if not cat_row.empty and "region" in cat_row.columns else None
        core_or_satellite = str(cat_row["core_or_satellite"].values[0]) if not cat_row.empty and "core_or_satellite" in cat_row.columns else None
        
        # If symbol is in prices DataFrame, extract its column
        if sym in prices.columns:
            s = prices[sym].dropna()
        else:
            s = pd.Series(dtype=float)
        
        # Compute coverage metrics
        if s.empty:
            rec = SymbolValidation(
                symbol=sym,
                start=None,
                end=None,
                history_years=0.0,
                missing_pct=100.0,
                n_obs=0,
                provider=provider or "none",
                valid=False,
                reason="no_data",
                provider_error=provider_error,
                median_volume=None,
                asset_class=asset_class,
                region=region,
                core_or_satellite=core_or_satellite,
            )
            records[sym] = rec
            dropped.append(sym)
            continue

        idx = s.index if isinstance(s.index, pd.DatetimeIndex) else pd.to_datetime(s.index, errors="coerce")
        idx = idx.sort_values()
        st = idx.min()
        en = idx.max()
        years = _years_between(st, en)
        
        # Missing pct relative to this symbol's own expected trading days between st and en
        try:
            expected = len(pd.bdate_range(st, en)) if pd.notna(st) and pd.notna(en) else 0
        except Exception:
            expected = 0
        got = int(s.shape[0])
        missing_pct = float(max(0.0, (1.0 - (got / expected)) * 100.0)) if expected > 0 else 0.0
        n_obs = got
        
        # Compute median volume from volume cache
        median_volume = None
        if sym in volume_cache:
            vol_series = volume_cache[sym].dropna()
            if not vol_series.empty:
                median_volume = float(vol_series.median())

        # Thresholds based on core/satellite bucket (use core_or_satellite from catalog metadata)
        tier = core_or_satellite if core_or_satellite else "satellite"
        min_years = core_min_years if tier.lower() == "core" else sat_min_years

        # Apply all validation checks including liquidity
        # Use small epsilon (0.01y ≈ 3.65 days) to handle floating point precision
        history_ok = years >= (min_years - 0.01)
        sparsity_ok = missing_pct <= max_missing_pct
        liquidity_ok = (median_volume is not None and median_volume >= min_median_volume) if min_median_volume else True
        
        is_valid = history_ok and sparsity_ok and (n_obs > 0) and liquidity_ok
        
        # Determine failure reason
        reason = None
        if not is_valid:
            r = []
            if not history_ok:
                r.append(f"years<{min_years:.0f}")
            if not sparsity_ok:
                r.append("missing>")
            if not liquidity_ok:
                r.append(f"vol<{min_median_volume}")
            if n_obs <= 0:
                r.append("noobs")
            reason = ",".join(r) if r else "invalid"

        rec = SymbolValidation(
            symbol=sym,
            start=st.strftime("%Y-%m-%d") if pd.notna(st) else None,
            end=en.strftime("%Y-%m-%d") if pd.notna(en) else None,
            history_years=round(years, 2),
            missing_pct=round(missing_pct, 2),
            n_obs=n_obs,
            provider=provider or "unknown",
            valid=is_valid,
            reason=reason,
            provider_error=provider_error,
            median_volume=int(median_volume) if median_volume is not None else None,
            asset_class=asset_class,
            region=region,
            core_or_satellite=core_or_satellite,
        )
        records[sym] = rec
        (valid if is_valid else dropped).append(sym)

    return records, valid, dropped


def write_universe_snapshot(
    out_path: Path,
    *,
    catalog_path: Path,
    records: Dict[str, SymbolValidation],
    valid: List[str],
    dropped: List[str],
    prices: Optional[pd.DataFrame] = None,
) -> Path:
    """Write a universe snapshot JSON including aggregate stats.
    Returns the path written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate metrics
    avg_vol, avg_corr = None, None
    sector_exposure: Dict[str, int] = {}
    asset_class_counts: Dict[str, int] = {}
    tier_counts: Dict[str, int] = {}
    
    # Collect valid records for aggregation
    valid_records = [records[sym] for sym in valid if sym in records]
    
    # Asset class and tier breakdown
    for rec in valid_records:
        if rec.asset_class:
            asset_class_counts[rec.asset_class] = asset_class_counts.get(rec.asset_class, 0) + 1
        if rec.core_or_satellite:
            tier_counts[rec.core_or_satellite] = tier_counts.get(rec.core_or_satellite, 0) + 1
    
    # History years distribution (min, median, max)
    history_years = [rec.history_years for rec in valid_records if rec.history_years]
    history_dist = None
    if history_years:
        history_dist = {
            "min": round(min(history_years), 2),
            "median": round(np.median(history_years), 2),
            "max": round(max(history_years), 2),
        }
    
    # Median volume statistics (for valid symbols with volume data)
    volumes = [rec.median_volume for rec in valid_records if rec.median_volume is not None]
    volume_dist = None
    if volumes:
        volume_dist = {
            "min": int(min(volumes)),
            "median": int(np.median(volumes)),
            "max": int(max(volumes)),
        }

    # Sector exposure from catalog (legacy field)
    try:
        df = load_assets_catalog(catalog_path)
        if "sector" in df.columns:
            sector_exposure = (
                df.loc[df.index.isin(valid)]["sector"].fillna("unknown").value_counts().to_dict()
            )
    except Exception:
        pass

    # Volatility/correlation from prices if provided
    if prices is not None and not prices.empty:
        try:
            rets = compute_returns(prices)
            # Annualized vol per asset
            vols = rets.std(ddof=0) * np.sqrt(252.0)
            avg_vol = float(np.nanmean(vols.values)) if len(vols) else None
            # Average pairwise correlation
            corr = rets.corr()
            if corr is not None and corr.size > 0:
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                avg_corr = float(np.nanmean(upper.values))
        except Exception:
            pass

    payload = {
        "catalog": str(catalog_path),
        "universe_size": len(records),
        "valid_count": len(valid),
        "dropped_count": len(dropped),
        "valid_symbols": valid,
        "dropped_symbols": dropped,
        "records": {k: asdict(v) for k, v in records.items()},
        "metrics": {
            "avg_volatility": avg_vol,
            "avg_correlation": avg_corr,
            "sector_exposure": sector_exposure,
            "asset_class_counts": asset_class_counts,
            "tier_counts": tier_counts,
            "history_years_distribution": history_dist,
            "volume_distribution": volume_dist,
        },
    }

    out_path.write_text(json.dumps(payload, indent=2))

    # Also write a lightweight metrics snapshot for dashboards
    metrics_path = out_path.parent / "universe_metrics.json"
    metrics_path.write_text(json.dumps(payload.get("metrics", {}), indent=2))
    return out_path


def build_validated_universe(
    catalog_path: Path | str = Path("config/assets_catalog.json"),
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    snapshot_path: Path | str = Path("data/outputs/universe_snapshot.json"),
) -> Tuple[List[str], Path]:
    """High-level helper: validate catalog and write snapshot.
    Returns (valid_symbols, snapshot_path).
    """
    cat_path = Path(catalog_path)
    catalog = load_assets_catalog(cat_path)

    # Load optional thresholds from config
    try:
        cfg = load_config(Path("config/config.yaml"))
        uni_cfg = cfg.get("universe", {}) if isinstance(cfg, dict) else {}
        core_min_years = float(uni_cfg.get("core_min_years", 10.0))
        sat_min_years = float(uni_cfg.get("sat_min_years", 7.0))
        max_missing_pct = float(uni_cfg.get("max_missing_pct", 10.0))
        min_median_volume = uni_cfg.get("min_median_volume")  # optional liquidity filter
    except Exception:
        core_min_years, sat_min_years, max_missing_pct = 10.0, 7.0, 10.0
        min_median_volume = None

    # KEY CHANGE: Allow start=None to fetch full available history from providers
    # Do NOT force a default_start_date here - let providers (esp. Tiingo) return all data
    # This allows validation based on "any 5+ years" rather than "5+ years since 2010"
    # If caller explicitly provides a start date, honor it; otherwise pass None to fetch pipeline

    records, valid, dropped = validate_catalog(
        catalog,
        start=start,
        end=end,
        core_min_years=core_min_years,
        sat_min_years=sat_min_years,
        max_missing_pct=max_missing_pct,
        min_median_volume=min_median_volume,
    )

    # Fetch prices for valid symbols to compute aggregate metrics
    start_date = pd.to_datetime(start).date() if start else None
    end_date = pd.to_datetime(end).date() if end else None
    prices_wide, _ = fetch_multiple(valid, start=start_date, end=end_date)

    snap_path = write_universe_snapshot(
        Path(snapshot_path),
        catalog_path=cat_path,
        records=records,
        valid=valid,
        dropped=dropped,
        prices=prices_wide,
    )
    return valid, snap_path


def load_valid_universe(
    snapshot_path: Path | str | None = None,
) -> Tuple[List[str], Dict[str, SymbolValidation], Dict]:
    """
    Load the validated ETF universe from the last saved snapshot.
    
    This is the RUNTIME function that should be used by the UI and portfolio engine.
    It does NOT re-query providers or re-validate symbols - just loads the snapshot.
    
    Args:
        snapshot_path: Path to universe_snapshot.json. 
                      If None, defaults to data/outputs/universe_snapshot.json
    
    Returns:
        (valid_symbols, records_dict, metrics_dict)
        
        valid_symbols: List of validated ticker symbols
        records_dict: {symbol: SymbolValidation} - per-symbol validation metadata
        metrics_dict: Aggregate metrics (asset class counts, provider breakdown, etc.)
    
    Raises:
        FileNotFoundError: If snapshot file doesn't exist
        ValueError: If snapshot is invalid/corrupted
    """
    if DEMO_MODE:
        catalog = get_demo_universe_frame()
        records, valid, _ = _demo_validate_catalog(catalog)
        metrics = _demo_metrics(catalog, valid, records)
        return valid, records, metrics

    if snapshot_path is None:
        snapshot_path = Path("data/outputs/universe_snapshot.json")
    else:
        snapshot_path = Path(snapshot_path)
    
    if not snapshot_path.exists():
        raise FileNotFoundError(
            f"Universe snapshot not found: {snapshot_path}\n"
            f"Run build_validated_universe() first to generate it."
        )
    
    try:
        data = json.loads(snapshot_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in universe snapshot: {e}")
    
    # Extract key fields
    valid_symbols = data.get("valid_symbols", [])
    if not valid_symbols:
        raise ValueError(f"No valid symbols in universe snapshot: {snapshot_path}")
    
    # Reconstruct SymbolValidation objects from dict records
    records_raw = data.get("records", {})
    records = {}
    for sym, rec_dict in records_raw.items():
        try:
            records[sym] = SymbolValidation(**rec_dict)
        except (TypeError, KeyError) as e:
            # Skip malformed records but log warning
            import logging
            logging.getLogger(__name__).warning(f"Skipping malformed record for {sym}: {e}")
    
    metrics = data.get("metrics", {})
    
    return valid_symbols, records, metrics


__all__ = [
    "SymbolValidation",
    "validate_catalog",
    "write_universe_snapshot",
    "build_validated_universe",
    "load_valid_universe",
]

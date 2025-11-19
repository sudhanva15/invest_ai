"""
Enhanced Universe Configuration Loader

Loads and manages the investment universe from config/universe.yaml.
Complements existing universe.py with YAML-based configuration.
"""

from __future__ import annotations
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import logging

_log = logging.getLogger(__name__)


def get_universe_yaml_path() -> Path:
    """Get path to universe.yaml config file."""
    # Try from package root
    candidates = [
        Path(__file__).parent.parent / "config" / "universe.yaml",
        Path("config") / "universe.yaml",
        Path("../config") / "universe.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    
    raise FileNotFoundError(
        "universe.yaml not found. Searched: " + ", ".join(str(c) for c in candidates)
    )


def load_universe_from_yaml(
    asset_classes: Optional[List[str]] = None,
    min_history_years: Optional[float] = None,
    core_only: bool = False,
) -> pd.DataFrame:
    """
    Load investment universe from config/universe.yaml.
    
    Args:
        asset_classes: Filter to specific asset classes (e.g. ['equity', 'bond'])
        min_history_years: Minimum required data history
        core_only: If True, only return core assets (exclude satellites)
    
    Returns:
        DataFrame with columns:
            - symbol: Ticker symbol
            - name: Asset name
            - asset_class: One of: equity, bond, reit, commodity, cash
            - subtype: More specific classification
            - region: Geographic region
            - is_etf: Boolean
            - provider: Data provider (stooq, tiingo, etc.)
            - core_satellite: 'core' or 'satellite'
            - max_weight: Maximum portfolio weight
            - min_history_years: Minimum data history requirement
    
    Example:
        >>> universe = load_universe_from_yaml(asset_classes=['equity', 'bond'])
        >>> symbols = universe['symbol'].tolist()
    """
    universe_path = get_universe_yaml_path()
    
    try:
        with open(universe_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        _log.error(f"Failed to load universe.yaml: {e}")
        raise
    
    if 'assets' not in config:
        raise ValueError("universe.yaml missing 'assets' key")
    
    assets = config['assets']
    
    # Convert to DataFrame
    df = pd.DataFrame(assets)
    
    # Validate required columns
    required_cols = ['symbol', 'name', 'asset_class', 'subtype', 'region', 
                     'is_etf', 'provider', 'core_satellite', 'max_weight']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Universe assets missing required columns: {missing}")
    
    # Apply filters
    if asset_classes:
        df = df[df['asset_class'].isin(asset_classes)]
    
    if min_history_years is not None:
        df = df[df.get('min_history_years', 0) >= min_history_years]
    
    if core_only:
        df = df[df['core_satellite'] == 'core']
    
    # Reset index
    df = df.reset_index(drop=True)
    
    _log.info(f"Loaded universe: {len(df)} assets after filters")
    
    return df


def get_asset_class_info() -> Dict[str, Dict]:
    """
    Get metadata for each asset class.
    
    Returns:
        Dict mapping asset_class -> {label, description, typical_return, typical_volatility}
    """
    universe_path = get_universe_yaml_path()
    
    with open(universe_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config.get('asset_class_info', {})


def get_symbols_by_asset_class(
    asset_classes: Optional[List[str]] = None,
    core_only: bool = False
) -> Dict[str, List[str]]:
    """
    Get symbols grouped by asset class.
    
    Args:
        asset_classes: Filter to specific asset classes
        core_only: If True, only include core assets
    
    Returns:
        Dict mapping asset_class -> list of symbols
    
    Example:
        >>> by_class = get_symbols_by_asset_class(['equity', 'bond'])
        >>> equity_symbols = by_class['equity']
    """
    universe = load_universe_from_yaml(asset_classes=asset_classes, core_only=core_only)
    
    result = {}
    for ac in universe['asset_class'].unique():
        symbols = universe[universe['asset_class'] == ac]['symbol'].tolist()
        result[ac] = symbols
    
    return result


def get_asset_metadata(symbol: str) -> Optional[Dict]:
    """
    Get metadata for a specific asset.
    
    Args:
        symbol: Ticker symbol
    
    Returns:
        Dict with asset metadata, or None if not found
    """
    universe = load_universe_from_yaml()
    
    row = universe[universe['symbol'] == symbol]
    if len(row) == 0:
        return None
    
    return row.iloc[0].to_dict()


def validate_universe_yaml() -> List[str]:
    """
    Validate universe configuration.
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        universe = load_universe_from_yaml()
    except Exception as e:
        return [f"Failed to load universe: {e}"]
    
    # Check for duplicate symbols
    duplicates = universe[universe.duplicated(subset=['symbol'], keep=False)]
    if len(duplicates) > 0:
        errors.append(f"Duplicate symbols: {duplicates['symbol'].tolist()}")
    
    # Check asset_class values
    valid_classes = {'equity', 'bond', 'reit', 'commodity', 'cash'}
    invalid = universe[~universe['asset_class'].isin(valid_classes)]
    if len(invalid) > 0:
        errors.append(f"Invalid asset_class values: {invalid['asset_class'].unique().tolist()}")
    
    # Check core_satellite values
    valid_cs = {'core', 'satellite'}
    invalid_cs = universe[~universe['core_satellite'].isin(valid_cs)]
    if len(invalid_cs) > 0:
        errors.append(f"Invalid core_satellite values: {invalid_cs['core_satellite'].unique().tolist()}")
    
    # Check max_weight bounds
    invalid_weight = universe[(universe['max_weight'] <= 0) | (universe['max_weight'] > 1.0)]
    if len(invalid_weight) > 0:
        errors.append(f"Invalid max_weight values for: {invalid_weight['symbol'].tolist()}")
    
    # Check for reasonable asset class coverage
    asset_classes = set(universe['asset_class'].unique())
    if 'equity' not in asset_classes:
        errors.append("Universe missing equity assets")
    if 'bond' not in asset_classes:
        errors.append("Universe missing bond assets")
    
    return errors


if __name__ == "__main__":
    # Quick validation
    print("="*70)
    print("UNIVERSE VALIDATION")
    print("="*70)
    
    errors = validate_universe_yaml()
    if errors:
        print("❌ ERRORS FOUND:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ Universe configuration valid")
    
    # Print summary
    universe = load_universe_from_yaml()
    print(f"\n📊 Universe Summary:")
    print(f"  Total assets: {len(universe)}")
    print(f"\n  By asset class:")
    for ac, count in universe['asset_class'].value_counts().items():
        print(f"    {ac}: {count}")
    print(f"\n  Core vs Satellite:")
    for cs, count in universe['core_satellite'].value_counts().items():
        print(f"    {cs}: {count}")

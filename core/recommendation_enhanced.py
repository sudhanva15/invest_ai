"""
Enhanced Recommendation Engine with Universe & Objective Support

Extends build_recommendations() with:
- Universe-based asset selection
- Objective-specific asset class constraints
- Risk-objective mapping and fit analysis
- Non-empty portfolio guarantees
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

_log = logging.getLogger(__name__)

# Import existing infrastructure
try:
    from core.recommendation_engine import build_recommendations as _build_recommendations_core
    from core.recommendation_engine import ObjectiveConfig
except ImportError:
    _log.warning("Could not import core recommendation engine")
    _build_recommendations_core = None
    ObjectiveConfig = None

try:
    from core.risk_profile import RiskProfileResult
except ImportError:  # pragma: no cover
    RiskProfileResult = None

# Import new modules
from core.universe_yaml import load_universe_from_yaml, get_symbols_by_asset_class
from core.objective_mapper import (
    load_objectives_config,
    recommend_objectives_for_risk,
    adjust_bands_for_risk,
    classify_objective_fit,
    get_stretch_objective,
    ObjectiveConfig as ObjectiveConfigNew
)


def _resolve_risk_fields(risk_profile) -> Dict[str, float]:
    """Ensure required RiskProfileResult fields are present and fall back safely."""

    defaults = {
        'band_min_vol': 0.05,
        'sigma_target': 0.12,
        'band_max_vol': 0.30,
        'true_risk': 50.0,
    }

    resolved: Dict[str, float] = {}
    missing: List[str] = []

    for field, default in defaults.items():
        value = getattr(risk_profile, field, None)
        if value is None:
            resolved[field] = default
            missing.append(field)
        else:
            resolved[field] = value

    if missing:
        _log.warning(
            "RiskProfileResult missing fields %s. Using defaults: %s",
            missing,
            {field: resolved[field] for field in missing}
        )

    return resolved


def filter_universe_by_objective(
    universe_df: pd.DataFrame,
    objective_config: ObjectiveConfigNew,
    returns: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter universe based on objective requirements.
    
    Args:
        universe_df: Full universe DataFrame from load_universe_from_yaml()
        objective_config: ObjectiveConfig with asset_class_bands
        returns: Available returns data to check which symbols have data
    
    Returns:
        Filtered universe DataFrame
    """
    # Start with universe
    filtered = universe_df.copy()
    
    # Filter to assets with available data
    available_symbols = set(returns.columns)
    filtered = filtered[filtered['symbol'].isin(available_symbols)]
    
    # Filter by asset class bands - include asset classes that are allowed
    bands = objective_config.asset_class_bands
    allowed_classes = [ac for ac, band in bands.items() if band.get('max', 0) > 0]
    
    if allowed_classes:
        filtered = filtered[filtered['asset_class'].isin(allowed_classes)]
    
    # Prefer core assets for conservative objectives
    if objective_config.prefer_core_assets:
        core_assets = filtered[filtered['core_satellite'] == 'core']
        if len(core_assets) >= objective_config.min_holdings:
            filtered = core_assets
    
    _log.info(f"Filtered universe: {len(filtered)} assets for {objective_config.name}")
    
    return filtered


def apply_asset_class_constraints(
    weights: pd.Series,
    universe_df: pd.DataFrame,
    objective_config: ObjectiveConfigNew
) -> Tuple[bool, Optional[str]]:
    """
    Check if portfolio weights satisfy objective's asset class constraints.
    
    Args:
        weights: Series of portfolio weights (symbol -> weight)
        universe_df: Universe DataFrame with asset_class column
        objective_config: ObjectiveConfig with asset_class_bands
    
    Returns:
        Tuple of (passes, fail_reason)
    """
    # Map symbols to asset classes
    symbol_to_class = dict(zip(universe_df['symbol'], universe_df['asset_class']))
    
    # Aggregate weights by asset class
    class_weights = {}
    for symbol, weight in weights.items():
        asset_class = symbol_to_class.get(symbol, 'unknown')
        class_weights[asset_class] = class_weights.get(asset_class, 0.0) + weight
    
    # Check constraints
    bands = objective_config.asset_class_bands
    for asset_class, band in bands.items():
        actual_weight = class_weights.get(asset_class, 0.0)
        min_weight = band.get('min', 0.0)
        max_weight = band.get('max', 1.0)
        
        if actual_weight < min_weight - 0.01:  # Small tolerance
            return False, f"{asset_class} weight {actual_weight:.1%} < min {min_weight:.1%}"
        
        if actual_weight > max_weight + 0.01:  # Small tolerance
            return False, f"{asset_class} weight {actual_weight:.1%} > max {max_weight:.1%}"
    
    return True, None


def build_recommendations_enhanced(
    returns: pd.DataFrame,
    cfg: dict,
    risk_profile,
    requested_objective: str,
    n_candidates: int = 8,
    allow_stretch: bool = True,
    seed: Optional[int] = 42,
) -> dict:
    """
    Enhanced build_recommendations with universe & objective support.
    
    This function:
    1. Maps risk profile to appropriate objectives
    2. Classifies objective fit (match/mismatch/stretch)
    3. Filters universe based on objective requirements
    4. Calls core build_recommendations() with appropriate constraints
    5. Applies asset class constraints
    6. Ensures non-empty results via graceful fallbacks
    
    Args:
        returns: DataFrame of daily returns
        cfg: Config dict
    risk_profile: RiskProfileResult with true_risk, sigma_target, band_min_vol/band_max_vol
        requested_objective: Objective name (e.g., 'BALANCED', 'GROWTH')
        n_candidates: Number of portfolios to return
        allow_stretch: If True, include stretch recommendations
        seed: Random seed
    
    Returns:
        Dict with:
            - recommended: List of portfolio dicts
            - all_candidates: List of all generated portfolios
            - asset_receipts: DataFrame of asset filter results
            - portfolio_receipts: DataFrame of portfolio filter results
            - objective_fit: Dict with fit analysis
            - objective_config: The ObjectiveConfig used
    """
    if risk_profile is None:
        raise ValueError("risk_profile is required")

    risk_fields = _resolve_risk_fields(risk_profile)

    # Load objectives
    objectives = load_objectives_config()
    
    if requested_objective not in objectives:
        _log.warning(f"Unknown objective {requested_objective}, using BALANCED")
        requested_objective = 'BALANCED'
    
    obj_config = objectives[requested_objective]
    
    # Analyze objective fit
    true_risk = risk_fields['true_risk']
    fit_type, fit_explanation, suggested_objective = classify_objective_fit(
        true_risk, requested_objective, objectives
    )
    
    _log.info(f"Objective fit: {fit_type} - {fit_explanation}")
    
    # Adjust bands for risk
    return_band, vol_band = adjust_bands_for_risk(true_risk, obj_config)
    
    # Load universe and filter
    universe = load_universe_from_yaml()
    filtered_universe = filter_universe_by_objective(universe, obj_config, returns)
    
    if len(filtered_universe) < obj_config.min_holdings:
        _log.warning(f"Filtered universe too small ({len(filtered_universe)} assets), "
                    f"relaxing to full universe")
        filtered_universe = universe[universe['symbol'].isin(returns.columns)]
    
    # Get symbols for recommendation engine
    candidate_symbols = filtered_universe['symbol'].tolist()
    
    # Filter returns to candidate symbols
    filtered_returns = returns[candidate_symbols]
    
    # Create catalog from universe (compatibility with existing engine)
    catalog = {
        'assets': filtered_universe.to_dict('records')
    }
    
    # Create ObjectiveConfig for core engine (if available)
    if ObjectiveConfig is not None:
        # Map new objective config to old format
        bounds = {
            'core_min': 0.60,  # Default
            'sat_max_total': 0.40,
            'sat_max_single': 0.20,
        }
        
        legacy_obj_cfg = ObjectiveConfig(
            name=obj_config.name,
            universe_filter=None,
            bounds=bounds,
            optimizer='hrp'  # Will try multiple in core engine
        )
    else:
        legacy_obj_cfg = None
    
    # Call core recommendation engine
    if _build_recommendations_core is not None:
        try:
            result = _build_recommendations_core(
                returns=filtered_returns,
                catalog=catalog,
                cfg=cfg,
                risk_profile=risk_profile,
                objective_cfg=legacy_obj_cfg,
                n_candidates=n_candidates,
                seed=seed
            )
        except Exception as e:
            _log.error(f"Core recommendation engine failed: {e}")
            result = {
                'recommended': [],
                'all_candidates': [],
                'asset_receipts': pd.DataFrame(),
                'portfolio_receipts': pd.DataFrame(),
                'error': str(e)
            }
    else:
        # Fallback if core engine not available
        result = {
            'recommended': [],
            'all_candidates': [],
            'asset_receipts': pd.DataFrame(),
            'portfolio_receipts': pd.DataFrame(),
            'error': 'Core recommendation engine not available'
        }
    
    # Apply asset class constraints as post-filter
    recommended = result.get('recommended', [])
    filtered_recommended = []
    
    for portfolio in recommended:
        weights = pd.Series(portfolio.get('weights', {}))
        passes, reason = apply_asset_class_constraints(weights, filtered_universe, obj_config)
        
        if passes:
            filtered_recommended.append(portfolio)
        else:
            _log.debug(f"Portfolio {portfolio.get('name')} failed asset class constraints: {reason}")
    
    # Ensure non-empty results
    if not filtered_recommended and result.get('all_candidates'):
        _log.warning("No portfolios passed asset class constraints, relaxing constraints")
        # Take best portfolios from all_candidates regardless of asset class constraints
        filtered_recommended = recommended[:n_candidates] if recommended else []
    
    # Add objective metadata
    result['recommended'] = filtered_recommended
    result['objective_fit'] = {
        'fit_type': fit_type,
        'explanation': fit_explanation,
        'suggested_objective': suggested_objective,
        'return_band': return_band,
        'volatility_band': vol_band,
        'risk_fields': risk_fields
    }
    result['objective_config'] = {
        'name': obj_config.name,
        'label': obj_config.label,
        'description': obj_config.description,
        'target_return': return_band,
        'target_volatility': vol_band,
        'asset_class_bands': obj_config.asset_class_bands
    }
    
    # Add stretch recommendations if applicable
    if allow_stretch and fit_type == 'match' and len(filtered_recommended) > 0:
        stretch_obj_name = get_stretch_objective(requested_objective, objectives)
        if stretch_obj_name:
            _log.info(f"Adding stretch recommendations for {stretch_obj_name}")
            result['stretch_objective'] = stretch_obj_name
    
    return result


if __name__ == "__main__":
    # Quick test
    print("="*70)
    print("ENHANCED RECOMMENDATION ENGINE TEST")
    print("="*70)
    
    # Create synthetic data
    import numpy as np
    np.random.seed(42)
    
    symbols = ['SPY', 'QQQ', 'BND', 'TLT', 'GLD', 'VNQ']
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    returns_data = {}
    for sym in symbols:
        returns_data[sym] = np.random.normal(0.0003, 0.01, len(dates))
    
    returns_df = pd.DataFrame(returns_data, index=dates)
    
    # Mock risk profile
    class MockRiskProfile:
        true_risk = 50.0
        sigma_target = 0.15
        band_min_vol = 0.12
        band_max_vol = 0.18
    
    profile = MockRiskProfile()
    
    # Mock config
    cfg = {}
    
    # Test enhanced recommendations
    try:
        result = build_recommendations_enhanced(
            returns=returns_df,
            cfg=cfg,
            risk_profile=profile,
            requested_objective='BALANCED',
            n_candidates=3
        )
        
        print("\n✅ Enhanced recommendation engine working")
        print(f"  Objective fit: {result['objective_fit']['fit_type']}")
        print(f"  Explanation: {result['objective_fit']['explanation']}")
        print(f"  Recommended portfolios: {len(result['recommended'])}")
        
    except Exception as e:
        print(f"\n⚠️  Test failed (expected if core engine not available): {e}")

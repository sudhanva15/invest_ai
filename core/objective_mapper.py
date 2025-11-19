"""
Objective Mapper - Risk to Objective Mapping

Maps user risk profiles to appropriate investment objectives and adjusts
target bands based on risk tolerance. Handles mismatch/match/stretch cases.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

_log = logging.getLogger(__name__)


@dataclass
class ObjectiveConfig:
    """Configuration for a single investment objective."""
    name: str
    label: str
    description: str
    risk_score_min: float
    risk_score_max: float
    risk_score_optimal: float
    target_return_min: float
    target_return_max: float
    target_return_target: float
    target_vol_min: float
    target_vol_max: float
    target_vol_target: float
    asset_class_bands: Dict[str, Dict[str, float]]
    min_holdings: int
    max_holdings: int
    prefer_core_assets: bool


def get_objectives_path() -> Path:
    """Get path to objectives.yaml config file."""
    candidates = [
        Path(__file__).parent.parent / "config" / "objectives.yaml",
        Path("config") / "objectives.yaml",
        Path("../config") / "objectives.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    
    raise FileNotFoundError(
        "objectives.yaml not found. Searched: " + ", ".join(str(c) for c in candidates)
    )


def load_objectives_config() -> Dict[str, ObjectiveConfig]:
    """
    Load all objective configurations.
    
    Returns:
        Dict mapping objective name -> ObjectiveConfig
    """
    config_path = get_objectives_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    objectives = {}
    for name, obj_data in config.get('objectives', {}).items():
        objectives[name] = ObjectiveConfig(
            name=name,
            label=obj_data['label'],
            description=obj_data['description'],
            risk_score_min=obj_data['risk_score_min'],
            risk_score_max=obj_data['risk_score_max'],
            risk_score_optimal=obj_data['risk_score_optimal'],
            target_return_min=obj_data['target_return']['min'],
            target_return_max=obj_data['target_return']['max'],
            target_return_target=obj_data['target_return']['target'],
            target_vol_min=obj_data['target_volatility']['min'],
            target_vol_max=obj_data['target_volatility']['max'],
            target_vol_target=obj_data['target_volatility']['target'],
            asset_class_bands=obj_data['asset_class_bands'],
            min_holdings=obj_data.get('min_holdings', 8),
            max_holdings=obj_data.get('max_holdings', 20),
            prefer_core_assets=obj_data.get('prefer_core_assets', False),
        )
    
    return objectives


def recommend_objectives_for_risk(
    true_risk: float,
    objectives_config: Optional[Dict[str, ObjectiveConfig]] = None
) -> List[str]:
    """
    Recommend appropriate objectives for a given risk score.
    
    Args:
        true_risk: Risk score on 0-100 scale
        objectives_config: Optional pre-loaded objectives (will load if not provided)
    
    Returns:
        List of objective names (e.g., ['BALANCED', 'GROWTH_PLUS_INCOME'])
        Sorted by distance from optimal risk score
    
    Example:
        >>> recommend_objectives_for_risk(50)
        ['BALANCED', 'GROWTH_PLUS_INCOME']
        >>> recommend_objectives_for_risk(20)
        ['CONSERVATIVE', 'BALANCED']
    """
    if objectives_config is None:
        objectives_config = load_objectives_config()
    
    # Find objectives where risk score falls within min-max range
    matching = []
    for name, obj in objectives_config.items():
        if obj.risk_score_min <= true_risk <= obj.risk_score_max:
            distance = abs(true_risk - obj.risk_score_optimal)
            matching.append((name, distance))
    
    # Sort by distance to optimal (closest first)
    matching.sort(key=lambda x: x[1])
    
    recommended = [name for name, _ in matching]
    
    # If no perfect matches, find the closest objective
    if not recommended:
        closest = min(
            objectives_config.items(),
            key=lambda x: min(
                abs(true_risk - x[1].risk_score_min),
                abs(true_risk - x[1].risk_score_max),
                abs(true_risk - x[1].risk_score_optimal)
            )
        )
        recommended = [closest[0]]
    
    _log.info(f"Risk score {true_risk:.1f} -> recommended objectives: {recommended}")
    
    return recommended


def adjust_bands_for_risk(
    true_risk: float,
    objective_config: ObjectiveConfig,
    config_path: Optional[Path] = None
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Adjust return and volatility bands based on how far true_risk is from objective's optimal.
    
    Args:
        true_risk: Risk score on 0-100 scale
        objective_config: The ObjectiveConfig to adjust
        config_path: Optional path to objectives.yaml (will find if not provided)
    
    Returns:
        Tuple of (return_band, vol_band) where:
            return_band = (min_return, max_return)
            vol_band = (min_vol, max_vol)
    
    Example:
        >>> obj = load_objectives_config()['BALANCED']
        >>> return_band, vol_band = adjust_bands_for_risk(40, obj)
        >>> # For risk=40 when optimal=50, returns are slightly reduced
    """
    if config_path is None:
        config_path = get_objectives_path()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get adjustment parameters
    adj = config.get('risk_adjustment', {})
    transition_zone = adj.get('transition_zone', 15)
    
    # Calculate distance from optimal
    distance = true_risk - objective_config.risk_score_optimal
    
    # Determine multipliers
    if abs(distance) <= transition_zone:
        # Within transition zone - no adjustment
        return_mult = 1.0
        vol_mult = 1.0
    elif distance < 0:
        # Below optimal - reduce targets
        return_mult = adj.get('below_optimal', {}).get('return_multiplier', 0.85)
        vol_mult = adj.get('below_optimal', {}).get('volatility_multiplier', 0.80)
    else:
        # Above optimal - increase targets
        return_mult = adj.get('above_optimal', {}).get('return_multiplier', 1.15)
        vol_mult = adj.get('above_optimal', {}).get('volatility_multiplier', 1.20)
    
    # Apply adjustments
    return_min = objective_config.target_return_min * return_mult
    return_max = objective_config.target_return_max * return_mult
    
    vol_min = objective_config.target_vol_min * vol_mult
    vol_max = objective_config.target_vol_max * vol_mult
    
    # Ensure sensible bounds
    return_min = max(0.01, return_min)  # At least 1% return min
    vol_min = max(0.01, vol_min)  # At least 1% vol min
    
    return_band = (return_min, return_max)
    vol_band = (vol_min, vol_max)
    
    _log.debug(f"Adjusted bands for risk={true_risk:.1f} (opt={objective_config.risk_score_optimal:.1f}): "
               f"return={return_band}, vol={vol_band}")
    
    return return_band, vol_band


def classify_objective_fit(
    true_risk: float,
    requested_objective: str,
    objectives_config: Optional[Dict[str, ObjectiveConfig]] = None
) -> Tuple[str, str, Optional[str]]:
    """
    Classify whether requested objective fits user's risk profile.
    
    Args:
        true_risk: Risk score on 0-100 scale
        requested_objective: Objective name requested by user
        objectives_config: Optional pre-loaded objectives
    
    Returns:
        Tuple of (fit_type, explanation, suggested_objective):
            fit_type: 'match', 'mismatch', or 'stretch'
            explanation: Human-readable reason
            suggested_objective: Alternative objective (if mismatch), else None
    
    Examples:
        >>> classify_objective_fit(50, 'BALANCED')
        ('match', 'Risk profile aligns well with Balanced objective', None)
        
        >>> classify_objective_fit(25, 'AGGRESSIVE')
        ('mismatch', 'Risk score too low for Aggressive', 'CONSERVATIVE')
    """
    if objectives_config is None:
        objectives_config = load_objectives_config()
    
    if requested_objective not in objectives_config:
        return ('mismatch', f'Unknown objective: {requested_objective}', None)
    
    obj = objectives_config[requested_objective]
    recommended = recommend_objectives_for_risk(true_risk, objectives_config)
    
    # Case 1: Perfect or good match
    if requested_objective in recommended:
        distance = abs(true_risk - obj.risk_score_optimal)
        if distance <= 15:
            return ('match', f'Risk profile aligns well with {obj.label} objective', None)
        else:
            return ('match', f'Risk profile is compatible with {obj.label} objective', None)
    
    # Case 2: Mismatch - outside comfort zone
    if true_risk < obj.risk_score_min:
        # User's risk too low for objective
        suggested = recommended[0] if recommended else 'CONSERVATIVE'
        return (
            'mismatch',
            f'Your risk score ({true_risk:.0f}) is below the recommended range '
            f'({obj.risk_score_min:.0f}-{obj.risk_score_max:.0f}) for {obj.label}',
            suggested
        )
    
    if true_risk > obj.risk_score_max:
        # User's risk too high for objective
        suggested = recommended[0] if recommended else 'GROWTH'
        return (
            'mismatch',
            f'Your risk score ({true_risk:.0f}) is above the recommended range '
            f'({obj.risk_score_min:.0f}-{obj.risk_score_max:.0f}) for {obj.label}',
            suggested
        )
    
    # Case 3: Stretch - user could handle more risk
    # (This would be determined by financial capacity, not just risk score)
    # For now, treat as match
    return ('match', f'Risk profile fits {obj.label} objective', None)


def get_stretch_objective(
    current_objective: str,
    objectives_config: Optional[Dict[str, ObjectiveConfig]] = None
) -> Optional[str]:
    """
    Get the next more aggressive objective for stretch scenarios.
    
    Args:
        current_objective: Current objective name
        objectives_config: Optional pre-loaded objectives
    
    Returns:
        Name of more aggressive objective, or None if already most aggressive
    
    Example:
        >>> get_stretch_objective('BALANCED')
        'GROWTH_PLUS_INCOME'
    """
    if objectives_config is None:
        objectives_config = load_objectives_config()
    
    # Define objective progression from conservative to aggressive
    progression = ['CONSERVATIVE', 'BALANCED', 'GROWTH_PLUS_INCOME', 'GROWTH', 'AGGRESSIVE']
    
    try:
        current_idx = progression.index(current_objective)
        if current_idx < len(progression) - 1:
            return progression[current_idx + 1]
    except (ValueError, IndexError):
        pass
    
    return None


if __name__ == "__main__":
    # Quick test
    print("="*70)
    print("OBJECTIVES CONFIGURATION TEST")
    print("="*70)
    
    objectives = load_objectives_config()
    print(f"\n✅ Loaded {len(objectives)} objectives:")
    for name, obj in objectives.items():
        print(f"  - {name}: {obj.label} (risk {obj.risk_score_min:.0f}-{obj.risk_score_max:.0f})")
    
    # Test risk mapping
    print("\n📊 Risk Score Mapping:")
    for risk in [20, 50, 80]:
        recommended = recommend_objectives_for_risk(risk)
        print(f"  Risk {risk}: {', '.join(recommended)}")
    
    # Test objective fit classification
    print("\n🎯 Objective Fit Classification:")
    test_cases = [
        (50, 'BALANCED'),
        (25, 'AGGRESSIVE'),
        (80, 'CONSERVATIVE'),
    ]
    for risk, obj in test_cases:
        fit_type, explanation, suggested = classify_objective_fit(risk, obj)
        print(f"  Risk {risk} + {obj}: {fit_type}")
        print(f"    → {explanation}")
        if suggested:
            print(f"    → Suggested: {suggested}")

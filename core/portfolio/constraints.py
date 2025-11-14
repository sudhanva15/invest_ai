"""Core/Satellite portfolio weight constraints and adjustment utilities."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd


def validate_weights(
    weights: Dict[str, float],
    core_symbols: List[str],
    satellite_symbols: Optional[List[str]] = None,
    core_min: float = 0.65,
    satellite_max: float = 0.35,
    single_max: Optional[float] = None,
    tolerance: float = 1e-6
) -> List[str]:
    """Check if weights satisfy core/satellite constraints.
    
    Args:
        weights: Portfolio weights to validate
        core_symbols: List of core asset symbols
        satellite_symbols: Optional list of satellite assets
        core_min: Minimum core allocation (65% = 0.65)
        satellite_max: Maximum satellite allocation (35% = 0.35)
        single_max: Maximum satellite position size, optional
        tolerance: Numerical tolerance for float comparisons
        
    Returns:
        List of constraint violation messages, empty if all pass
    """
    if not weights:
        return ["No weights provided"]
        
    violations = []
    
    # Check normalization first 
    total = sum(weights.values())
    if abs(total - 1.0) > tolerance:
        violations.append(f"Weights sum to {total:.6f}, expected 1.0")
        if total == 0:
            return violations
    
    # Check core minimum
    core_sum = sum(weights.get(s, 0) for s in core_symbols)
    if core_sum < (core_min - tolerance):
        violations.append(
            f"Core allocation {core_sum:.1%} below {core_min:.1%} minimum"
        )
    
    # Handle satellites if specified
    if satellite_symbols:
        sat_sum = sum(weights.get(s, 0) for s in satellite_symbols)
        
        # Check satellite maximum
        if sat_sum > (satellite_max + tolerance):
            violations.append(
                f"Satellite allocation {sat_sum:.1%} above {satellite_max:.1%} maximum"
            )
        
        # Check individual satellite caps if specified
        if single_max:
            for s in satellite_symbols:
                if s in weights and weights[s] > (single_max + tolerance):
                    violations.append(
                        f"Satellite position {s} ({weights[s]:.1%}) exceeds {single_max:.1%} cap"
                    )
    
    return violations


def apply_weight_constraints(
    weights: Dict[str, float],
    core_symbols: List[str],
    satellite_symbols: Optional[List[str]] = None,
    core_min: float = 0.65,
    satellite_max: float = 0.35,
    single_max: Optional[float] = None  # Optional for core-only
) -> Dict[str, float]:
    """Apply core-satellite constraints to portfolio weights.
    
    This function enforces several constraints on a portfolio allocation:
    1. Single position cap (satellites only) if specified
    2. Core allocation must be at least core_min
    3. Satellite allocation cannot exceed satellite_max
    4. Final weights are normalized to sum to 1.0
    
    Args:
        weights: Original weights as {symbol: weight} dict
        core_symbols: List of core asset symbols
        satellite_symbols: Optional list of satellite asset symbols
        core_min: Minimum allocation to core assets (0.65 = 65%)
        satellite_max: Maximum allocation to satellites (0.35 = 35%)
        single_max: Optional max for satellite positions
        
    Returns:
        Dict mapping symbols to adjusted weights that satisfy all constraints
    """
    # Input validation
    if not weights or not core_symbols:
        return {}
    if not all(isinstance(v, (int, float)) for v in weights.values()):
        raise ValueError("All weights must be numeric")
    if any(v < 0 for v in weights.values()):
        raise ValueError("All weights must be non-negative")
    
    # Work with a fresh copy
    result = {k: float(v) for k, v in weights.items()}
    
    # Step 1: Initial normalization
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
    
    # Step 2: Handle core/satellite allocation
    if satellite_symbols:
        # First ensure satellite allocation is within maximum
        sat_sum = sum(result.get(s, 0) for s in satellite_symbols)
        if sat_sum > satellite_max:
            scale = satellite_max / sat_sum
            for s in satellite_symbols:
                if s in result:
                    result[s] *= scale
    
        # Then cap individual satellite positions if needed
        if single_max:
            sat_total = 0
            for s in satellite_symbols:
                if s in result and result[s] > single_max:
                    result[s] = single_max
                if s in result:
                    sat_total += result[s]
                    
            # Re-normalize satellite allocation if needed
            if sat_total > satellite_max:
                scale = satellite_max / sat_total
                for s in satellite_symbols:
                    if s in result:
                        result[s] *= scale
    
    # Step 3: Ensure minimum core allocation
    core_sum = sum(result.get(s, 0) for s in core_symbols)
    if core_sum < core_min:
        if core_sum > 0:
            # Scale up existing core positions proportionally
            scale = core_min / core_sum
            for s in core_symbols:
                if s in result:
                    result[s] *= scale
        else:
            # No core allocation present: seed an equal split across core to satisfy core_min
            n_core = max(1, len(core_symbols))
            eq_core = core_min / n_core
            for s in core_symbols:
                # Initialize missing core symbols to their equal share
                result[s] = eq_core

        # Scale down non-core (satellites) proportionally to fit remaining budget
        if satellite_symbols:
            remain = max(0.0, 1.0 - core_min)
            sat_sum = sum(result.get(s, 0) for s in satellite_symbols)
            if sat_sum > 0:
                scale = remain / sat_sum
                for s in satellite_symbols:
                    if s in result:
                        result[s] *= scale
    
    # Step 4: Final cleanup and normalization
    total = sum(result.values())
    if total > 0:
        result = {k: v / total for k, v in result.items()}
        
        # One final satellite position cap check
        if satellite_symbols and single_max:
            for s in satellite_symbols:
                if s in result and result[s] > single_max:
                    excess = result[s] - single_max
                    result[s] = single_max
                    
                    # Redistribute excess to core proportionally
                    core_total = sum(result.get(s, 0) for s in core_symbols)
                    if core_total > 0:
                        for s in core_symbols:
                            if s in result:
                                result[s] += excess * (result[s] / core_total)
        
        # Remove tiny weights
        result = {k: v for k, v in result.items() if abs(v) > 1e-6}
    
    return result



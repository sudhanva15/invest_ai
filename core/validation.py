"""Lightweight validation for UI and runtime checks.

This module provides fast validation checks suitable for calling from the UI
or during portfolio generation. For comprehensive validation, see dev/validate_simulations.py.
"""

from __future__ import annotations
from typing import Optional
import pandas as pd


def run_light_validator(
    objective: Optional[str] = None,
    universe_size: Optional[int] = None,
    returns: Optional[pd.DataFrame] = None,
) -> bool:
    """Run lightweight validation checks suitable for UI calls.
    
    This is a fast subset of checks from dev/validate_simulations.py,
    designed to be called from the Streamlit UI without blocking.
    
    Args:
        objective: Objective name (e.g., "balanced", "growth"). Optional.
        universe_size: Number of assets in universe. Optional.
        returns: Returns DataFrame for basic checks. Optional.
    
    Returns:
        True if all checks pass, False otherwise.
    
    TODO: Expand this to call more sophisticated checks as needed:
        - Price data freshness (< 30 days stale)
        - Macro data availability
        - Candidate generation smoke test (at least 3 valid candidates)
        - Receipt generation health
        
    For now, this performs basic sanity checks:
        - Universe not empty
        - Returns DataFrame has sufficient data (if provided)
        - No critical configuration errors
    """
    try:
        # Check 1: Universe size
        if universe_size is not None and universe_size < 3:
            return False
        
        # Check 2: Returns data quality (if provided)
        if returns is not None:
            if returns.empty:
                return False
            
            # Check for sufficient history (at least 1 year = 252 trading days)
            if len(returns) < 252:
                return False
            
            # Check for excessive NaN ratio (> 20% fails)
            nan_ratio = returns.isna().sum().sum() / (len(returns) * len(returns.columns))
            if nan_ratio > 0.20:
                return False
            
            # Check for at least 3 tickers with valid data
            valid_tickers = (returns.notna().sum() > 126).sum()
            if valid_tickers < 3:
                return False
        
        # Check 3: Objective validity (if provided)
        if objective is not None:
            # Import here to avoid circular dependencies
            try:
                from core.recommendation_engine import DEFAULT_OBJECTIVES
                if objective not in DEFAULT_OBJECTIVES and objective not in {"balanced", "growth", "income"}:
                    return False
            except ImportError:
                pass
        
        # All checks passed
        return True
        
    except Exception:
        # If any check raises an exception, treat as failure
        return False


def get_validation_message(passed: bool) -> str:
    """Get a user-friendly message for validation status.
    
    Args:
        passed: Whether validation passed.
    
    Returns:
        User-friendly status message.
    """
    if passed:
        return "✓ Basic validation checks passed"
    else:
        return "⚠ Some validation checks failed; treat results as rough guidance"

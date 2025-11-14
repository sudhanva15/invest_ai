from __future__ import annotations
from typing import Dict, Tuple


def compute_risk_score(answers: Dict[str, float]) -> float:
    """
    Compute a user risk score in [0, 100] from 8 questionnaire responses.

    Expected keys in `answers` (each value in [0, 100]):
      - q1_time_horizon
      - q2_loss_tolerance
      - q3_reaction_20_drop
      - q4_income_stability
      - q5_dependence_on_money
      - q6_investing_experience
      - q7_safety_net
      - q8_goal_type

    Weights:
      Q1: 0.20, Q2: 0.20, Q3: 0.15, Q4: 0.10,
      Q5: 0.15, Q6: 0.10, Q7: 0.05, Q8: 0.05

    Returns a float in [0, 100]. Missing questions default to 50.
    
    UI Integration:
        UI code should use the map_*_choice() functions to convert MCQ selections
        to numeric scores before calling this function. Example:
        
            from core.risk_profile import map_time_horizon_choice, compute_risk_score
            
            horizon_choice = st.radio("Time horizon?", ["0–3 years", "3–7 years", ...])
            answers = {
                "q1_time_horizon": map_time_horizon_choice(horizon_choice),
                ...
            }
            risk_score = compute_risk_score(answers)
    """
    # Default mid-point if a response is missing
    default = 50.0
    q = {
        "q1_time_horizon": float(answers.get("q1_time_horizon", default)),
        "q2_loss_tolerance": float(answers.get("q2_loss_tolerance", default)),
        "q3_reaction_20_drop": float(answers.get("q3_reaction_20_drop", default)),
        "q4_income_stability": float(answers.get("q4_income_stability", default)),
        "q5_dependence_on_money": float(answers.get("q5_dependence_on_money", default)),
        "q6_investing_experience": float(answers.get("q6_investing_experience", default)),
        "q7_safety_net": float(answers.get("q7_safety_net", default)),
        "q8_goal_type": float(answers.get("q8_goal_type", default)),
    }

    # Clamp each to [0, 100]
    for k in q:
        q[k] = max(0.0, min(100.0, q[k]))

    weights = {
        "q1_time_horizon": 0.20,
        "q2_loss_tolerance": 0.20,
        "q3_reaction_20_drop": 0.15,
        "q4_income_stability": 0.10,
        "q5_dependence_on_money": 0.15,
        "q6_investing_experience": 0.10,
        "q7_safety_net": 0.05,
        "q8_goal_type": 0.05,
    }

    score = 0.0
    for k, w in weights.items():
        score += w * q[k]

    # Ensure [0, 100]
    return max(0.0, min(100.0, float(score)))


# ============================================================================
# MCQ Mapping Functions (UI → numeric scores)
# ============================================================================
# These functions map textual multiple-choice answers to numeric scores [0, 100]
# that feed into compute_risk_score. UI code should call these mapping functions
# and pass the numeric results to compute_risk_score(answers).


def map_time_horizon_choice(choice: str) -> float:
    """
    Map time horizon choice to score [0, 100].
    
    Longer horizons → higher score (more time to recover from downturns).
    
    Args:
        choice: One of "0–3 years", "3–7 years", "7–15 years", "15+ years"
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "0–3 years": 0.0,
        "3–7 years": 40.0,
        "7–15 years": 70.0,
        "15+ years": 100.0,
    }
    return mapping.get(choice, 50.0)  # Default to middle if unknown


def map_loss_tolerance_choice(choice: str) -> float:
    """
    Map loss tolerance choice to score [0, 100].
    
    Higher tolerance → higher score.
    
    Args:
        choice: One of "Very low", "Low", "Medium", "High", "Very high"
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Very low": 0.0,
        "Low": 25.0,
        "Medium": 50.0,
        "High": 75.0,
        "Very high": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_reaction_20_drop_choice(choice: str) -> float:
    """
    Map behavioral reaction to 20% drop to score [0, 100].
    
    Calmer reactions → higher score.
    
    Args:
        choice: Behavioral statement about reaction to market drop
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Sell everything immediately": 0.0,
        "Sell some to reduce risk": 20.0,
        "Hold and wait": 50.0,
        "Hold and might buy more": 80.0,
        "Definitely buy more (opportunity)": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_income_stability_choice(choice: str) -> float:
    """
    Map income stability to score [0, 100].
    
    More stable income → higher score (can handle volatility).
    
    Args:
        choice: One of "Very unstable", "Unstable", "Moderate", "Stable", "Very stable"
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Very unstable": 0.0,
        "Unstable": 25.0,
        "Moderate": 50.0,
        "Stable": 75.0,
        "Very stable": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_dependence_on_money_choice(choice: str) -> float:
    """
    Map dependence on invested money to score [0, 100].
    
    Less dependence → higher score (can afford to take risks).
    
    Args:
        choice: Statement about financial dependence
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Critical for living expenses": 0.0,
        "Important but not critical": 30.0,
        "Helpful but have other income": 60.0,
        "Nice-to-have growth money": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_investing_experience_choice(choice: str) -> float:
    """
    Map investing experience to score [0, 100].
    
    More experience → higher score (comfortable with volatility).
    
    Args:
        choice: One of "Beginner", "Some experience", "Experienced", "Advanced"
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Beginner (first time)": 0.0,
        "Some experience (< 3 years)": 35.0,
        "Experienced (3-10 years)": 70.0,
        "Advanced (10+ years)": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_safety_net_choice(choice: str) -> float:
    """
    Map safety net strength to score [0, 100].
    
    Stronger safety net → higher score (can afford temporary losses).
    
    Args:
        choice: Statement about emergency fund/insurance
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "No emergency fund or insurance": 0.0,
        "Small emergency fund (< 3 months)": 30.0,
        "Moderate safety net (3-6 months)": 60.0,
        "Strong safety net (6+ months)": 100.0,
    }
    return mapping.get(choice, 50.0)


def map_goal_type_choice(choice: str) -> float:
    """
    Map investment goal type to score [0, 100].
    
    More growth-oriented → higher score.
    
    Args:
        choice: One of "Capital preservation", "Income generation", "Balanced growth", "Aggressive growth"
    
    Returns:
        Numeric score in [0, 100]
    """
    mapping = {
        "Capital preservation (safety first)": 0.0,
        "Income generation (steady returns)": 30.0,
        "Balanced growth (moderate risk)": 60.0,
        "Aggressive growth (max returns)": 100.0,
    }
    return mapping.get(choice, 50.0)


def compute_credibility_score(
    history_years: float,
    robustness_score: float,
    num_assets: int,
    validator_passed: bool,
) -> float:
    """
    Combine sub-scores into a credibility score in [0, 100].

    Args:
        history_years: Number of years of historical data used.
        robustness_score: Score from compute_simple_robustness_score on segmented CAGRs
                         (expected [0, 100]). Higher = more consistent across time segments.
        num_assets: Number of assets in the portfolio.
        validator_passed: Whether portfolio passed validation checks.

    Sub-scores:
        - history_score  = min(history_years / 20.0, 1.0) * 100
        - universe_score = min(num_assets / 20.0, 1.0) * 100
        - validator_score = 100 if validator_passed else 50

    Final credibility = 0.30*history_score + 0.30*robustness_score + 0.20*universe_score + 0.20*validator_score

    Example:
        For a portfolio with 10 years of data, 80 robustness, 12 assets, and passing validation:
        history_score = (10/20) * 100 = 50
        robustness_score = 80
        universe_score = (12/20) * 100 = 60
        validator_score = 100
        credibility = 0.30*50 + 0.30*80 + 0.20*60 + 0.20*100 = 71.0
    """
    h = max(0.0, float(history_years))
    r = max(0.0, min(100.0, float(robustness_score)))
    n = max(0, int(num_assets))

    history_score = min(h / 20.0, 1.0) * 100.0
    universe_score = min(n / 20.0, 1.0) * 100.0
    validator_score = 100.0 if bool(validator_passed) else 50.0

    credibility = (
        0.30 * history_score
        + 0.30 * r
        + 0.20 * universe_score
        + 0.20 * validator_score
    )
    return max(0.0, min(100.0, float(credibility)))


def compute_outcome_band(mu: float, sigma: float, credibility: float) -> Tuple[float, float]:
    """Compute a symmetric outcome band around mu, scaled by sigma and credibility."""
    base_band = float(sigma)
    scale = 1.0 + (100.0 - float(credibility)) / 100.0
    low = float(mu) - base_band * scale
    high = float(mu) + base_band * scale
    return float(low), float(high)


def compute_simple_robustness_score(segment_cagrs: list[float]) -> float:
    """
    Compute a robustness score in [0, 100] from segmented CAGR values.

    segment_cagrs: CAGR values for 3–4 equal time segments of the same portfolio.
    Returns a 0–100 score where:
    - 100 means segments are very consistent (low variability).
    - Lower scores mean more variability between segments.

    If segment_cagrs is empty, returns 50.0 as a neutral default.

    Mapping logic:
    - std <= 0.02 (2 percentage points) ≈ near 100
    - std >= 0.20 (20 percentage points) ≈ near 0
    - Linear interpolation between these thresholds
    """
    if not segment_cagrs:
        return 50.0

    import statistics
    if len(segment_cagrs) == 1:
        return 100.0  # single segment is perfectly consistent

    mean_cagr = statistics.mean(segment_cagrs)
    std_cagr = statistics.stdev(segment_cagrs)

    # Map std to score: lower std → higher score
    # std_low = 0.02 → score ~100
    # std_high = 0.20 → score ~0
    std_low = 0.02
    std_high = 0.20

    if std_cagr <= std_low:
        score = 100.0
    elif std_cagr >= std_high:
        score = 0.0
    else:
        # Linear interpolation
        score = 100.0 * (1.0 - (std_cagr - std_low) / (std_high - std_low))

    return max(0.0, min(100.0, float(score)))


def compute_robustness_from_curve(equity_curve, n_segments: int = 3) -> float:
    """
    Compute robustness score from an equity curve by segmenting and analyzing CAGR consistency.
    
    Args:
        equity_curve: pandas Series representing cumulative portfolio growth (1.0 = start)
        n_segments: Number of equal-length segments to split the curve into (default 3)
    
    Returns:
        Robustness score in [0, 100]. Higher = more consistent growth across segments.
        Returns 50.0 if curve too short or invalid.
    
    Usage:
        # After computing portfolio returns and equity curve
        curve = (1 + portfolio_returns).cumprod()
        robustness = compute_robustness_from_curve(curve, n_segments=3)
    """
    try:
        import pandas as pd
        
        if not isinstance(equity_curve, pd.Series) or len(equity_curve) < 126:
            # Need at least ~6 months of data for meaningful segmentation
            return 50.0
        
        curve = equity_curve.dropna()
        if len(curve) < n_segments * 30:
            # Need at least ~30 days per segment for CAGR
            return 50.0
        
        segment_cagrs = []
        n = len(curve)
        step = n // n_segments
        
        for i in range(n_segments):
            start_idx = i * step
            end_idx = (i + 1) * step if i < n_segments - 1 else n
            
            segment = curve.iloc[start_idx:end_idx]
            if len(segment) < 2:
                continue
            
            # Compute annualized CAGR for this segment
            start_val = float(segment.iloc[0])
            end_val = float(segment.iloc[-1])
            
            if start_val <= 0:
                continue
            
            # CAGR = (end/start)^(252/days) - 1
            days = len(segment)
            if days > 0:
                cagr = float((end_val / start_val) ** (252.0 / days) - 1.0)
                segment_cagrs.append(cagr)
        
        return compute_simple_robustness_score(segment_cagrs)
    
    except Exception:
        return 50.0

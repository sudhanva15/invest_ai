from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass

__all__ = [
    "RiskProfileResult",
    "compute_risk_profile",
    "compute_risk_score",
    "compute_risk_score_questionnaire",
    "compute_risk_score_facts",
    "risk_label",
    "map_true_risk_to_vol_band",
    "map_true_risk_to_cagr_band",
]


# ============================================================================
# RiskProfileResult – First-class risk profile container
# ============================================================================

@dataclass(frozen=True, init=False)
class RiskProfileResult:
    """
    Immutable risk profile with normalized field names expected by Phase 3/4.

    Required fields:
      - questionnaire_score, facts_score, combined_score, slider_score, true_risk
      - vol_min, vol_target, vol_max
      - cagr_min, cagr_target (Phase 3: expected growth range)
      - label, horizon_years, objective

    Backward-compatibility properties are provided for legacy code paths:
      - score_questionnaire, score_facts, score_combined
      - sigma_target, band_min_vol, band_max_vol
    """
    questionnaire_score: float
    facts_score: float
    combined_score: float
    slider_score: float
    true_risk: float
    vol_min: float
    vol_target: float
    vol_max: float
    cagr_min: float
    cagr_target: float
    label: str
    horizon_years: int
    objective: str

    def __init__(self, **kwargs: Any):  # type: ignore[no-untyped-def]
        """Allow construction with both new and legacy field names.

        Legacy names accepted:
          - score_questionnaire -> questionnaire_score
          - score_facts -> facts_score
          - score_combined -> combined_score
          - sigma_target -> vol_target
          - band_min_vol -> vol_min
          - band_max_vol -> vol_max
        
        Phase 3 additions:
          - cagr_min, cagr_target (auto-computed from true_risk if not provided)
        """
        q = kwargs.get("questionnaire_score", kwargs.get("score_questionnaire"))
        f = kwargs.get("facts_score", kwargs.get("score_facts"))
        c = kwargs.get("combined_score", kwargs.get("score_combined"))
        s = kwargs.get("slider_score", kwargs.get("slider_value", c))
        tr = kwargs.get("true_risk", kwargs.get("true_risk_score", c))
        vmin = kwargs.get("vol_min", kwargs.get("band_min_vol"))
        vtgt = kwargs.get("vol_target", kwargs.get("sigma_target"))
        vmax = kwargs.get("vol_max", kwargs.get("band_max_vol"))
        cagr_min_arg = kwargs.get("cagr_min")
        cagr_target_arg = kwargs.get("cagr_target")
        label = kwargs.get("label", "Moderate")
        horizon = kwargs.get("horizon_years", kwargs.get("horizon", 10))
        objective = kwargs.get("objective", "Balanced")

        # Basic validation/conversion
        def _f(x, default=0.0):
            try:
                return float(x)
            except Exception:
                return float(default)
        def _i(x, default=10):
            try:
                return int(x)
            except Exception:
                return int(default)
        
        # Compute CAGR band if not provided
        tr_val = _f(tr, c)
        if cagr_min_arg is None or cagr_target_arg is None:
            cagr_min_val, cagr_target_val = map_true_risk_to_cagr_band(tr_val)
        else:
            cagr_min_val = _f(cagr_min_arg)
            cagr_target_val = _f(cagr_target_arg)
        
        # Set attributes (frozen dataclass workaround)
        object.__setattr__(self, "questionnaire_score", _f(q))
        object.__setattr__(self, "facts_score", _f(f))
        object.__setattr__(self, "combined_score", _f(c))
        object.__setattr__(self, "slider_score", _f(s, c))
        object.__setattr__(self, "true_risk", tr_val)
        object.__setattr__(self, "vol_min", _f(vmin))
        object.__setattr__(self, "vol_target", _f(vtgt))
        object.__setattr__(self, "vol_max", _f(vmax))
        object.__setattr__(self, "cagr_min", cagr_min_val)
        object.__setattr__(self, "cagr_target", cagr_target_val)
        object.__setattr__(self, "label", str(label))
        object.__setattr__(self, "horizon_years", _i(horizon))
        object.__setattr__(self, "objective", str(objective))

    # ---- Backward compatibility read-only aliases ----
    @property
    def score_questionnaire(self) -> float:
        return self.questionnaire_score

    @property
    def score_facts(self) -> float:
        return self.facts_score

    @property
    def score_combined(self) -> float:
        return self.combined_score

    @property
    def sigma_target(self) -> float:
        return self.vol_target

    @property
    def band_min_vol(self) -> float:
        return self.vol_min

    @property
    def band_max_vol(self) -> float:
        return self.vol_max


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


# ============================================================================
# Income-Based Risk Scoring (Capacity)
# ============================================================================

def compute_risk_score_facts(income_profile: dict) -> float:
    """
    Compute capacity-based risk score from financial profile [0, 100].
    
    Combines 4 financial factors (each 0-25 points):
    1. Income stability (very unstable=5, very stable=25)
    2. Emergency fund (none=0, 6+ months=25)
    3. Investable surplus ratio (vs annual expenses)
    4. Debt burden (vs annual income)
    
    Args:
        income_profile: Dict with keys:
            - income_stability: str ("Very stable", "Stable", etc.)
            - emergency_fund_months: float (months of expenses covered)
            - investable_amount: float ($ available to invest)
            - monthly_expenses: float ($ per month)
            - outstanding_debt: float ($ total debt)
            - annual_income: float ($ per year)
    
    Returns:
        Score in [0, 100]. Higher = greater financial capacity for risk.
    
    Formula (from audit Section 13.10.2):
        S_facts = S_stability + S_efund + S_surplus + S_debt
        Each component ∈ [0, 25], then clamped to [0, 100]
    """
    score = 0.0
    
    # 1. Income stability (0-25 pts)
    stability_map = {
        "Very stable": 25.0,
        "Stable": 20.0,
        "Moderate": 15.0,
        "Unstable": 10.0,
        "Very unstable": 5.0,
    }
    score += stability_map.get(income_profile.get("income_stability", "Moderate"), 15.0)
    
    # 2. Emergency fund (0-25 pts)
    efund_months = float(income_profile.get("emergency_fund_months", 0))
    if efund_months >= 6.0:
        score += 25.0
    elif efund_months >= 3.0:
        score += 15.0
    elif efund_months >= 1.0:
        score += 8.0
    else:
        score += 0.0
    
    # 3. Investable surplus (0-25 pts)
    investable = float(income_profile.get("investable_amount", 0))
    monthly_exp = float(income_profile.get("monthly_expenses", 1))
    annual_exp = monthly_exp * 12.0
    if annual_exp > 0:
        surplus_ratio = investable / annual_exp
        if surplus_ratio >= 2.0:
            score += 25.0
        elif surplus_ratio >= 1.0:
            score += 20.0
        elif surplus_ratio >= 0.5:
            score += 12.0
        elif surplus_ratio >= 0.2:
            score += 8.0
        else:
            score += 3.0
    else:
        score += 10.0  # neutral if expenses unknown
    
    # 4. Debt burden (0-25 pts) - lower debt = higher score
    debt = float(income_profile.get("outstanding_debt", 0))
    annual_income = float(income_profile.get("annual_income", 1))
    if annual_income > 0:
        debt_ratio = debt / annual_income
        if debt_ratio < 0.1:
            score += 25.0
        elif debt_ratio < 0.5:
            score += 20.0
        elif debt_ratio < 1.5:
            score += 15.0
        elif debt_ratio < 3.0:
            score += 8.0
        else:
            score += 0.0
    else:
        score += 10.0  # neutral if income unknown
    
    return max(0.0, min(100.0, float(score)))


def risk_label(score: float) -> str:
    """
    Map risk score to qualitative label.
    
    Args:
        score: Risk score in [0, 100]
    
    Returns:
        Label: "Very Conservative", "Conservative", "Moderate", 
               "Growth-Oriented", or "Aggressive"
    """
    if score < 20:
        return "Very Conservative"
    elif score < 40:
        return "Conservative"
    elif score < 60:
        return "Moderate"
    elif score < 80:
        return "Growth-Oriented"
    else:
        return "Aggressive"


# ============================================================================
# Main Risk Profile Computation (Phase 3 signature)
# ============================================================================

def compute_risk_score_questionnaire(questionnaire_answers: Dict[str, Any]) -> float:
    """Wrapper for legacy compute_risk_score to match Phase 3 naming."""
    return compute_risk_score(questionnaire_answers)


# ============================================================================
# TRUE_RISK → Volatility & CAGR Mapping (Phase 3 calibration)
# ============================================================================

# Calibration anchors for TRUE_RISK → CAGR mapping
# Based on historical equity/bond mix performance (1990-2025):
# - Conservative (20): 80/20 bonds/equity → ~5-6% CAGR
# - Moderate (50): 60/40 equity/bonds → ~8-9% CAGR  
# - Aggressive (80): 90/10 equity/bonds → ~10-11% CAGR
CAGR_ANCHOR_LOW = (20.0, 0.05, 0.06)   # (risk_score, cagr_min, cagr_target)
CAGR_ANCHOR_MID = (50.0, 0.08, 0.09)
CAGR_ANCHOR_HIGH = (80.0, 0.10, 0.11)

def map_true_risk_to_cagr_band(true_risk: float) -> Tuple[float, float]:
    """
    Map TRUE_RISK score [0,100] to expected (cagr_min, cagr_target) band.
    
    Uses piecewise linear interpolation between calibrated anchors:
    - TRUE_RISK ≈ 20 → 5-6% CAGR (conservative, bond-heavy)
    - TRUE_RISK ≈ 50 → 8-9% CAGR (balanced, 60/40 equity/bonds)
    - TRUE_RISK ≈ 80 → 10-11% CAGR (aggressive, equity-heavy)
    
    Args:
        true_risk: Risk score in [0, 100]
    
    Returns:
        (cagr_min, cagr_target) as decimals (e.g., 0.08 for 8%)
    
    Example:
        >>> cagr_min, cagr_target = map_true_risk_to_cagr_band(50)
        >>> print(f"Expected CAGR band: {cagr_min:.1%} - {cagr_target:.1%}")
        Expected CAGR band: 8.0% - 9.0%
    
    Usage:
        from core.risk_profile import compute_risk_profile, map_true_risk_to_cagr_band
        
        profile = compute_risk_profile(...)
        cagr_min, cagr_target = map_true_risk_to_cagr_band(profile.true_risk)
    """
    tr = max(0.0, min(100.0, float(true_risk)))
    
    # Piecewise linear interpolation
    if tr <= CAGR_ANCHOR_LOW[0]:
        # Below low anchor: use low anchor values
        return float(CAGR_ANCHOR_LOW[1]), float(CAGR_ANCHOR_LOW[2])
    
    elif tr <= CAGR_ANCHOR_MID[0]:
        # Between low and mid: interpolate
        t = (tr - CAGR_ANCHOR_LOW[0]) / (CAGR_ANCHOR_MID[0] - CAGR_ANCHOR_LOW[0])
        cagr_min = CAGR_ANCHOR_LOW[1] + t * (CAGR_ANCHOR_MID[1] - CAGR_ANCHOR_LOW[1])
        cagr_target = CAGR_ANCHOR_LOW[2] + t * (CAGR_ANCHOR_MID[2] - CAGR_ANCHOR_LOW[2])
        return float(cagr_min), float(cagr_target)
    
    elif tr <= CAGR_ANCHOR_HIGH[0]:
        # Between mid and high: interpolate
        t = (tr - CAGR_ANCHOR_MID[0]) / (CAGR_ANCHOR_HIGH[0] - CAGR_ANCHOR_MID[0])
        cagr_min = CAGR_ANCHOR_MID[1] + t * (CAGR_ANCHOR_HIGH[1] - CAGR_ANCHOR_MID[1])
        cagr_target = CAGR_ANCHOR_MID[2] + t * (CAGR_ANCHOR_HIGH[2] - CAGR_ANCHOR_MID[2])
        return float(cagr_min), float(cagr_target)
    
    else:
        # Above high anchor: use high anchor values
        return float(CAGR_ANCHOR_HIGH[1]), float(CAGR_ANCHOR_HIGH[2])


def map_true_risk_to_vol_band(score: float) -> Tuple[float, float, float]:
    """
    Map true risk score [0,100] to (vol_min, vol_target, vol_max).
    Calibrated to observed candidate distribution (Nov 2025):
      sigma_min=0.1271, sigma_max=0.2202, band=±0.02.
    """
    s = max(0.0, min(100.0, float(score)))
    sigma_min = 0.1271
    sigma_max = 0.2202
    vol_target = sigma_min + (sigma_max - sigma_min) * (s / 100.0)
    band = 0.02
    vol_min = max(0.0, vol_target - band)
    vol_max = vol_target + band
    return vol_min, vol_target, vol_max


def infer_investment_horizon(questionnaire_answers: Dict[str, Any], income_profile: Dict[str, Any]) -> int:
    """Simple heuristic for horizon; default to 10 if not provided."""
    try:
        # Prefer explicit field
        h = int(income_profile.get("horizon_years") or 0)
        if h > 0:
            return h
    except Exception:
        pass
    return 10


def compute_risk_profile(
    questionnaire_answers: Dict[str, Any],
    income_profile: Dict[str, Any],
    slider_score: Optional[float] = None,
) -> RiskProfileResult:
    """
    Phase 3 orchestrator that combines questionnaire, facts, and slider into a full profile.

    Steps:
      1) questionnaire_score = compute_risk_score_questionnaire(questionnaire_answers)
      2) facts_score = compute_risk_score_facts(income_profile)
      3) combined_score = 0.5 * questionnaire_score + 0.5 * facts_score
      4) true_risk = 0.7 * combined_score + 0.3 * slider_score (if provided, else combined_score)
      5) (vol_min, vol_target, vol_max) = map_true_risk_to_vol_band(true_risk)
      6) (cagr_min, cagr_target) = map_true_risk_to_cagr_band(true_risk)  # Phase 3
      7) label = risk_label(true_risk)
      8) horizon_years = infer_investment_horizon(...)
      9) objective = income_profile.get("objective", "Balanced")
    """
    q_score = float(compute_risk_score_questionnaire(questionnaire_answers or {}))
    f_score = float(compute_risk_score_facts(income_profile or {}))
    combined = 0.5 * q_score + 0.5 * f_score
    if slider_score is None:
        tr = combined
        slider_used = combined  # for record; indicates neutral blending
    else:
        s = max(0.0, min(100.0, float(slider_score)))
        tr = 0.7 * combined + 0.3 * s
        slider_used = s
    # Clamp
    tr = max(0.0, min(100.0, float(tr)))

    vol_min, vol_target, vol_max = map_true_risk_to_vol_band(tr)
    cagr_min, cagr_target = map_true_risk_to_cagr_band(tr)
    label = risk_label(tr)
    horizon = infer_investment_horizon(questionnaire_answers, income_profile)
    objective = str(income_profile.get("objective", "Balanced"))

    return RiskProfileResult(
        questionnaire_score=float(q_score),
        facts_score=float(f_score),
        combined_score=float(combined),
        slider_score=float(slider_used),
        true_risk=float(tr),
        vol_min=float(vol_min),
        vol_target=float(vol_target),
        vol_max=float(vol_max),
        cagr_min=float(cagr_min),
        cagr_target=float(cagr_target),
        label=str(label),
        horizon_years=int(horizon),
        objective=objective,
    )


# ============================================================================
# Contribution Plans (DCA + Lump Sum)
# ============================================================================

def compute_contribution_plans(
    agi: float,
    net_worth: float,
    risk_profile: RiskProfileResult,
    objective: str | None = None,
    horizon_years: int | None = None,
) -> list[dict]:
    """
    Generate 2-3 contribution plan scenarios based on income and risk profile.
    
    Plans vary by aggressiveness:
    - Baseline: Conservative monthly + small lump sum
    - Ambitious: Moderate monthly + medium lump sum
    - Aggressive: Higher monthly + larger lump sum (if risk allows)
    
    Args:
        agi: Adjusted Gross Income (annual)
        net_worth: Total net worth (assets - liabilities)
        risk_profile: RiskProfileResult from compute_risk_profile
        objective: Investment objective (used to adjust plans)
        horizon_years: Time horizon (longer = can be more aggressive)
    
    Returns:
        List of plan dicts, each with:
            - name: "Baseline", "Ambitious", "Aggressive"
            - lump: lump sum contribution ($)
            - monthly: monthly DCA contribution ($)
            - description: brief explanation
    
    Heuristics:
        - Monthly contribution scales with AGI and risk score
        - Lump sum scales with net worth and risk score
        - Objective modifies ranges:
            * "preservation" → lower contributions
            * "growth"/"aggressive" → higher contributions
        - Horizon affects lump sum (longer horizon = can invest more upfront)
    
    Usage:
        profile = compute_risk_profile(...)
        plans = compute_contribution_plans(
            agi=120000,
            net_worth=350000,
            risk_profile=profile,
            objective="balanced",
            horizon_years=10
        )
        
        for plan in plans:
            print(f"{plan['name']}: ${plan['lump']} lump + ${plan['monthly']}/mo")
    """
    agi = max(0.0, float(agi))
    net_worth = max(0.0, float(net_worth))
    true_risk = risk_profile.true_risk
    horizon = horizon_years or 10
    obj = objective or "balanced"
    
    # Base monthly contribution as % of AGI
    # Scale by risk: conservative=2%, moderate=4%, aggressive=8%
    monthly_pct_base = 0.02 + (true_risk / 100.0) * 0.06  # range [2%, 8%]
    
    # Adjust for objective
    obj_multiplier = {
        "preservation": 0.6,
        "income": 0.8,
        "balanced": 1.0,
        "growth": 1.2,
        "aggressive": 1.4,
    }.get(obj, 1.0)
    
    monthly_annual = (agi * monthly_pct_base * obj_multiplier) / 12.0
    
    # Lump sum as % of net worth
    # Scale by risk and horizon: conservative+short=1%, aggressive+long=5%
    lump_pct_base = 0.01 + (true_risk / 100.0) * 0.04  # range [1%, 5%]
    horizon_multiplier = min(1.5, 0.5 + (horizon / 20.0))  # cap at 1.5x for 20+ year horizons
    
    lump_base = net_worth * lump_pct_base * horizon_multiplier * obj_multiplier
    
    # Generate 3 plans
    plans = []
    
    # Baseline (conservative)
    plans.append({
        "name": "Baseline",
        "lump": round(lump_base * 0.7, 2),
        "monthly": round(monthly_annual * 0.8, 2),
        "description": "Conservative approach with lower initial investment and steady monthly contributions"
    })
    
    # Ambitious (moderate)
    plans.append({
        "name": "Ambitious",
        "lump": round(lump_base * 1.0, 2),
        "monthly": round(monthly_annual * 1.0, 2),
        "description": "Balanced approach matching your risk profile and financial capacity"
    })
    
    # Aggressive (if risk allows)
    if true_risk >= 40:  # Only offer aggressive if not too conservative
        plans.append({
            "name": "Aggressive",
            "lump": round(lump_base * 1.4, 2),
            "monthly": round(monthly_annual * 1.3, 2),
            "description": "Maximum investment pace for faster wealth accumulation"
        })
    
    return plans

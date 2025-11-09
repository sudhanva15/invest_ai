from dataclasses import dataclass
from typing import Literal, Dict, Any

# Canonical tier ordering (low → high)
TIER_ORDER: list[str] = ["retail", "accredited", "qualified_client", "qualified_purchaser", "institutional"]
_TIER_POS = {t: i for i, t in enumerate(TIER_ORDER)}

def tier_order() -> list[str]:
    """Public accessor for canonical tier order."""
    return list(TIER_ORDER)

def tier_index(tier: str) -> int:
    """Return 0-based index for a given tier; unknown tiers map to 0 (retail)."""
    return _TIER_POS.get(str(tier), 0)

def allowed_tiers_for(user_tier: str) -> list[str]:
    """
    Return a list of tiers that are <= the user's tier in seniority.
    Example: 'qualified_client' → ['retail','accredited','qualified_client']
    """
    pos = tier_index(user_tier)
    return TIER_ORDER[: pos + 1]

Tier = Literal["retail","accredited","qualified_client","qualified_purchaser","institutional"]

@dataclass
class InvestorInputs:
    income: float
    net_worth: float
    horizon_years: int
    risk_pct: int
    objective: Literal["grow","grow_income","income","tax_efficiency","preserve"]

def classify_investor(income: float, net_worth: float) -> Tier:
    """Heuristic classification by net worth / income; returns canonical Tier string."""
    if net_worth >= 25_000_000: return "institutional"
    if net_worth >= 5_000_000:  return "qualified_purchaser"
    if net_worth >= 2_200_000:  return "qualified_client"
    if net_worth >= 1_000_000 or income >= 200_000: return "accredited"
    return "retail"

def safe_investment_budget(income: float) -> Dict[str, Any]:
    """Return heuristic monthly contribution bands as a dict (baseline/ambitious/aggressive)."""
    baseline = income * 0.10 / 12.0
    ambitious = income * 0.20 / 12.0
    aggressive = income * 0.30 / 12.0
    return {
        "baseline_monthly": round(baseline, 2),
        "ambitious_monthly": round(ambitious, 2),
        "aggressive_monthly": round(aggressive, 2),
        "note": "Heuristic only; not financial advice."
    }

def min_risk_by_objective() -> Dict[str,int]:
    """
    Minimum risk% floors for objectives (0–100 scale).
    These act as guardrails so the UI cannot select an incoherent objective/risk pairing.
    """
    floors = {"preserve": 5, "income": 15, "tax_efficiency": 20, "grow_income": 35, "grow": 45}
    # Clamp to 0..100 in case of config edits
    return {k: max(0, min(100, int(v))) for k, v in floors.items()}

def coerce_risk_pct(objective: str, risk_pct: int) -> int:
    """
    Enforce a minimum risk% floor based on objective; clamps to [0,100].
    Example: objective='grow' with risk_pct=20 → returns 45 (floor).
    """
    floors = min_risk_by_objective()
    floor = floors.get(objective, floors.get("income", 15))
    rp = int(risk_pct)
    rp = max(floor, rp)
    return max(0, min(100, rp))

def validate_inputs(inp: "InvestorInputs") -> "InvestorInputs":
    """
    Return a sanitized copy of InvestorInputs:
      - risk_pct coerced by objective floors
      - horizon_years coerced to >= 0
      - income/net_worth coerced to >= 0
    """
    return InvestorInputs(
        income=max(0.0, float(inp.income)),
        net_worth=max(0.0, float(inp.net_worth)),
        horizon_years=max(0, int(inp.horizon_years)),
        risk_pct=coerce_risk_pct(inp.objective, int(inp.risk_pct)),
        objective=inp.objective,
    )

__all__ = [
    "Tier",
    "InvestorInputs",
    "classify_investor",
    "safe_investment_budget",
    "min_risk_by_objective",
    "coerce_risk_pct",
    "validate_inputs",
    "tier_order",
    "tier_index",
    "allowed_tiers_for",
]

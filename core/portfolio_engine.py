import numpy as np, pandas as pd
from typing import Any

# Defensive imports: allow runtime without pypfopt (fallback to equal-weight)
_risk_models: Any = None
_expected_returns: Any = None
_EfficientFrontier: Any = None
_objective_functions: Any = None
_HRPOpt: Any = None
try:
    from pypfopt import risk_models as _risk_models
    from pypfopt import expected_returns as _expected_returns
    from pypfopt import EfficientFrontier as _EfficientFrontier
    from pypfopt import objective_functions as _objective_functions
    from pypfopt import HRPOpt as _HRPOpt
    _PYPFOPT_AVAILABLE = True
except Exception:  # ModuleNotFoundError or partial import failure
    _PYPFOPT_AVAILABLE = False

# Robust config loader import
try:
    from .utils.env_tools import load_config  # preferred
except Exception:  # fallback if package layout differs
    from core.utils.env_tools import load_config  # type: ignore

# Import robust returns cleaning utility
try:
    from .utils.returns_cleaner import clean_prices_to_returns
except Exception:
    from core.utils.returns_cleaner import clean_prices_to_returns  # type: ignore

CFG = load_config()

def estimate_mu_cov(rets: pd.DataFrame):
    if not _PYPFOPT_AVAILABLE:
        # Dummy mean/cov so downstream logic can still execute equal-weight fallback
        mu = pd.Series([0.0]*rets.shape[1], index=rets.columns)
        S = rets.cov()
        return mu, S
    mu = _expected_returns.mean_historical_return(rets.dropna(how="all"), frequency=252)
    S  = _risk_models.sample_cov(rets.dropna(how="all"), frequency=252)
    return mu, S


def optimize_weights(
    rets: pd.DataFrame,
    method: str | None = None,
    bounds: tuple[float, float] | None = None,
    rf: float | None = None,
    cfg: dict | None = None,
) -> pd.Series:
    """
    Backward-compatible optimizer that now honors config keys under cfg['risk'].
    Order of precedence:
      1) explicit args (method, bounds, rf)
      2) cfg['risk'] {optimizer, min_weight, max_weight, risk_free_rate}
      3) module-level CFG defaults
    
    Returns are expected to be pre-cleaned; if you have prices, call
    clean_prices_to_returns() first.
    """
    cfg = cfg or CFG
    risk_cfg = (cfg or {}).get("risk", {})
    opt = (method or risk_cfg.get("optimizer") or "hrp").strip().lower()

    # derive bounds
    if bounds is None:
        wmin = float(risk_cfg.get("min_weight", 0.0))
        wmax = float(risk_cfg.get("max_weight", 0.3))
        bounds = (wmin, wmax)
    else:
        wmin, wmax = map(float, bounds)

    rfr = float(rf if rf is not None else risk_cfg.get("risk_free_rate", 0.015))

    mu, S = estimate_mu_cov(rets)
    if not _PYPFOPT_AVAILABLE:
        # Equal-weight fallback if library missing
        n = rets.shape[1]
        return pd.Series({c: 1.0/n for c in rets.columns})

    try:
        if opt == "hrp" and _PYPFOPT_AVAILABLE:
            hrp = _HRPOpt(rets.cov())
            w = hrp.optimize()
            w_series = pd.Series(w).sort_values(ascending=False)
            return _apply_constraints_if_available(w_series, rets.columns.tolist(), cfg)
        elif opt in {"ef_max_sharpe", "max_sharpe", "markowitz"}:
            ef = _EfficientFrontier(mu, S, weight_bounds=(wmin, wmax))
            ef.add_objective(_objective_functions.L2_reg, gamma=0.001)
            ef.max_sharpe(risk_free_rate=rfr)
            w = pd.Series(ef.clean_weights())
            w = w.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)
            return _apply_constraints_if_available(w, rets.columns.tolist(), cfg)
        elif opt in {"min_volatility", "ef_min_vol"}:
            ef = _EfficientFrontier(mu, S, weight_bounds=(wmin, wmax))
            ef.min_volatility()
            w = pd.Series(ef.clean_weights())
            w = w.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)
            return _apply_constraints_if_available(w, rets.columns.tolist(), cfg)
    except Exception:
        # Fall through to HRP / equal-weight
        pass

    # fallback to HRP if available else equal-weight
    if _PYPFOPT_AVAILABLE:
        try:
            hrp = _HRPOpt(rets.cov())
            w = hrp.optimize()
            w_series = pd.Series(w).sort_values(ascending=False)
            return _apply_constraints_if_available(w_series, rets.columns.tolist(), cfg)
        except Exception:
            pass
    n = rets.shape[1]
    w_eq = pd.Series({c: 1.0/n for c in rets.columns})
    return _apply_constraints_if_available(w_eq, rets.columns.tolist(), cfg)

def _apply_constraints_if_available(weights: pd.Series, symbols: list, cfg: dict | None = None) -> pd.Series:
    """Apply Core/Satellite constraints if module exists; sanitize and normalize."""
    try:
        from core.portfolio.constraints import apply_weight_constraints
        # Default allocation metadata (can be refined from catalog if needed)
        CORE = symbols  # simplistic: treat all as core by default
        SATS = []
        w_dict = {str(k): float(v) for k, v in weights.items()}
        w_dict = apply_weight_constraints(
            w_dict,
            core_symbols=CORE,
            satellite_symbols=SATS,
            core_min=0.65,
            satellite_max=0.35,
            single_max=0.07,
        )
        weights = pd.Series(w_dict)
    except Exception:
        pass
    # Sanitize: drop negatives, renormalize
    weights = weights.fillna(0.0).clip(lower=0.0)
    total = weights.sum()
    if total > 0 and np.isfinite(total):
        weights = weights / total
    else:
        # Emergency equal-weight fallback
        n = len(symbols)
        weights = pd.Series({s: 1.0/n for s in symbols})
    return weights

def portfolio_returns(rets: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = rets[weights.index]
    return (aligned * weights.values).sum(axis=1)

# === [WIRING PATCH] Engine facade with config knobs =========================
def build_portfolio(returns_df: pd.DataFrame, cfg: dict):
    """
    Facade for optimizer selection using config keys:
    risk.optimizer, risk.min_weight, risk.max_weight, risk_free_rate.
    Falls back to HRP if optimizer not recognized.
    
    Expects returns_df to be pre-cleaned; if you have prices, call
    clean_prices_to_returns(prices, winsor_p=0.005, min_non_na=252) first.
    """
    opt = (cfg.get("risk", {}).get("optimizer") or "hrp").lower()
    wmin = float(cfg.get("risk", {}).get("min_weight", 0.0))
    wmax = float(cfg.get("risk", {}).get("max_weight", 0.3))
    rfr = float(cfg.get("risk", {}).get("risk_free_rate", 0.015))

    if not _PYPFOPT_AVAILABLE:
        n = returns_df.shape[1]
        eq_w = pd.Series({c: 1.0/n for c in returns_df.columns})
        return _apply_constraints_if_available(eq_w, returns_df.columns.tolist(), cfg)

    try:
        if opt == "hrp" and _PYPFOPT_AVAILABLE:
            hrp = _HRPOpt(returns_df.cov())
            w = hrp.optimize()
            w_series = pd.Series(w).sort_values(ascending=False)
            return _apply_constraints_if_available(w_series, returns_df.columns.tolist(), cfg)
        elif opt in {"ef_max_sharpe", "max_sharpe", "markowitz"}:
            mu, S = estimate_mu_cov(returns_df)
            ef = _EfficientFrontier(mu, S, weight_bounds=(wmin, wmax))
            ef.add_objective(_objective_functions.L2_reg, gamma=0.001)
            ef.max_sharpe(risk_free_rate=rfr)
            weights = pd.Series(ef.clean_weights())
            weights = weights.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)
            return _apply_constraints_if_available(weights, returns_df.columns.tolist(), cfg)
        elif opt in {"min_volatility", "ef_min_vol"}:
            mu, S = estimate_mu_cov(returns_df)
            ef = _EfficientFrontier(mu, S, weight_bounds=(wmin, wmax))
            ef.min_volatility()
            weights = pd.Series(ef.clean_weights())
            weights = weights.fillna(0.0).clip(lower=0.0).sort_values(ascending=False)
            return _apply_constraints_if_available(weights, returns_df.columns.tolist(), cfg)
    except Exception:
        pass

    if _PYPFOPT_AVAILABLE:
        try:
            hrp = _HRPOpt(returns_df.cov())
            w = hrp.optimize()
            w_series = pd.Series(w).sort_values(ascending=False)
            return _apply_constraints_if_available(w_series, returns_df.columns.tolist(), cfg)
        except Exception:
            pass
    n = returns_df.shape[1]
    eq_w = pd.Series({c: 1.0/n for c in returns_df.columns})
    return _apply_constraints_if_available(eq_w, returns_df.columns.tolist(), cfg)

# ===========================================================================
__all__ = ["estimate_mu_cov", "optimize_weights", "portfolio_returns", "build_portfolio", "clean_prices_to_returns"]

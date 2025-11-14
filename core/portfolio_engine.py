import numpy as np, pandas as pd
from typing import Any, Optional, Dict, Tuple

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

# Import legacy cleaner (kept for strict mode and backward compatibility)
try:
    from .utils.returns_cleaner import clean_prices_to_returns as _legacy_cleaner
except Exception:
    from core.utils.returns_cleaner import clean_prices_to_returns as _legacy_cleaner  # type: ignore

CFG = load_config()

# ---------------- Robust returns cleaning (per-asset then align) ----------------
def clean_prices_to_returns(
    prices: pd.DataFrame,
    winsor_p: float = 0.005,
    min_non_na: int = 126,
    k_days: int = 1260,
    strict: bool = False,
    return_diagnostics: bool = False,
) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Robust price->returns cleaning with per-asset returns first, then outer-join alignment.

    - Coerce each column to numeric (to_numeric errors='coerce')
    - Forward-fill up to 5 consecutive missing values per asset (bounded ffill)
    - Compute pct_change per column with fill_method=None (no implicit ffills)
    - Outer-join all return series on date index
    - Select last k_days trading days window where columns have at least min_non_na points
    - Drop columns still failing min_non_na
    - Optionally winsorize tails symmetrically by winsor_p
    - Return diagnostics when requested

    strict=True switches to legacy strict behavior (min_non_na>=252, full intersection alignment).
    """
    import numpy as _np
    import pandas as _pd

    if prices is None or len(prices) == 0:
        return (_pd.DataFrame(), {"dropped_symbols": [], "kept_symbols": [], "window": (None, None)}) if return_diagnostics else _pd.DataFrame()

    if strict:
        # Delegate to legacy cleaner with stricter defaults
        out = _legacy_cleaner(prices, winsor_p=winsor_p, min_non_na=max(252, int(min_non_na)))
        diags = {"mode": "strict", "dropped_symbols": [], "kept_symbols": list(out.columns), "window": (out.index.min(), out.index.max())}
        return (out, diags) if return_diagnostics else out

    # Per-asset returns first
    returns_list = []
    kept = []
    dropped = []
    for col in prices.columns:
        s = _pd.to_numeric(prices[col], errors="coerce")
        # Bounded forward-fill small gaps (e.g., long weekends / occasional misses)
        s = s.ffill(limit=5)
        r = s.pct_change(fill_method=None)
        if r.count() < min_non_na:
            dropped.append((str(col), "too-few-points"))
            continue
        if not _np.isfinite(r).any():
            dropped.append((str(col), "non-finite"))
            continue
        returns_list.append(r.rename(str(col)))
        kept.append((str(col), int(r.notna().sum())))

    if not returns_list:
        out = _pd.DataFrame()
        diags = {"dropped_symbols": dropped, "kept_symbols": [], "window": (None, None)}
        return (out, diags) if return_diagnostics else out

    # Outer-join by index
    rets = _pd.concat(returns_list, axis=1, join="outer")
    rets.index = _pd.to_datetime(rets.index, errors="coerce")
    rets = rets.sort_index()

    # Select last k_days window
    if k_days and k_days > 0 and len(rets.index) > k_days:
        rets = rets.iloc[-k_days:]

    # Drop columns failing min_non_na within window
    ok_cols = [c for c in rets.columns if rets[c].count() >= int(min_non_na)]
    dropped += [(c, "too-few-points-window") for c in rets.columns if c not in ok_cols]
    rets = rets[ok_cols]

    if rets.empty:
        diags = {"dropped_symbols": dropped, "kept_symbols": [], "window": (None, None)}
        return (_pd.DataFrame(), diags) if return_diagnostics else _pd.DataFrame()

    # Winsorize tails if requested (column-wise)
    if winsor_p and 0 < winsor_p < 0.5:
        q_lo = rets.quantile(winsor_p)
        q_hi = rets.quantile(1 - winsor_p)
        for c in rets.columns:
            lo = q_lo.get(c)
            hi = q_hi.get(c)
            if lo is not None and hi is not None and _np.isfinite(lo) and _np.isfinite(hi):
                rets[c] = rets[c].clip(float(lo), float(hi))

    # Final sanitization
    rets = rets.replace([_np.inf, -_np.inf], _np.nan).dropna(how="any")

    window = (rets.index.min(), rets.index.max()) if not rets.empty else (None, None)
    diags = {"dropped_symbols": dropped, "kept_symbols": kept, "window": window}
    return (rets, diags) if return_diagnostics else rets

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

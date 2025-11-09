from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
 # Robust config import (prefer env_tools; fallback to core path)
try:
    from .utils.env_tools import load_config  # preferred
except Exception:  # fallback for different package layouts
    from core.utils.env_tools import load_config  # type: ignore
from .portfolio_engine import optimize_weights, portfolio_returns
from .backtesting import summarize_backtest, equity_curve

CFG = load_config()

@dataclass
class UserProfile:
    monthly_contribution: float
    horizon_years: int
    risk_level: str

def target_vol_for(level: str) -> float:
    # Default buckets if CFG['risk']['target_vol_buckets'] is absent
    default = {
        "aggressive": 0.24,
        "moderate":  0.16,
        "conservative": 0.10,
        "income": 0.12,
        "preserve": 0.07,
    }
    risk_cfg = (CFG or {}).get("risk", {})
    buckets = risk_cfg.get("target_vol_buckets", default)
    return buckets.get(level, buckets.get("moderate", 0.16))

def recommend_legacy(rets: pd.DataFrame, profile: UserProfile):
    risk_cfg = (CFG or {}).get("risk", {})
    method = (risk_cfg.get("optimizer") or "hrp")
    bounds = (float(risk_cfg.get("min_weight", 0.0)), float(risk_cfg.get("max_weight", 0.3)))
    rf = float(risk_cfg.get("risk_free_rate", 0.015))

    weights = optimize_weights(rets, method=method, bounds=bounds, rf=rf, cfg=CFG)
    port = portfolio_returns(rets, weights)
    # summarize_backtest expects a Series of portfolio returns and annual rf
    summary = summarize_backtest(port, rf=rf)
    curve = equity_curve(port)
    rec = {"weights": weights.to_dict() if hasattr(weights, "to_dict") else dict(weights),
           "metrics": summary,
           "curve": curve}
    return rec



def _class_of_symbol(symbol: str, catalog: dict) -> str:
    for a in catalog["assets"]:
        if a["symbol"].upper() == symbol.upper():
            return a.get("class", "unknown")
    return "unknown"

def _bounds_by_objective(objective: str):
    """
    Per asset-class bounds (fractions that must sum within [0,1] on the portfolio),
    tuned for sensible defaults.
    """
    # (min, max) by class
    # classes expected in catalog: public_equity, public_equity_intl, public_equity_em,
    # treasury_long, tbills, corporate_bond, high_yield, tax_eff_muni, commodities, gold, reit, etc.
    if objective == "grow":
        return {
            "equity_all_min": 0.70, "equity_all_max": 0.95,
            "bond_all_min":   0.00, "bond_all_max":   0.25,
            "cash_min":       0.00, "cash_max":       0.10,   # BIL/SHY
            "alts_min":       0.00, "alts_max":       0.20,   # GLD/DBC/VNQ bucket
        }
    if objective == "grow_income":
        return {
            "equity_all_min": 0.55, "equity_all_max": 0.85,
            "bond_all_min":   0.10, "bond_all_max":   0.35,
            "cash_min":       0.00, "cash_max":       0.10,
            "alts_min":       0.00, "alts_max":       0.20,
        }
    if objective == "income":
        return {
            "equity_all_min": 0.15, "equity_all_max": 0.45,
            "bond_all_min":   0.30, "bond_all_max":   0.70,
            "cash_min":       0.00, "cash_max":       0.20,
            "alts_min":       0.00, "alts_max":       0.15,
        }
    if objective == "preserve":
        return {
            "equity_all_min": 0.00, "equity_all_max": 0.25,
            "bond_all_min":   0.40, "bond_all_max":   0.90,
            "cash_min":       0.05, "cash_max":       0.50,
            "alts_min":       0.00, "alts_max":       0.10,
        }
    if objective == "tax_efficiency":
        return {
            "equity_all_min": 0.40, "equity_all_max": 0.80,
            "bond_all_min":   0.10, "bond_all_max":  0.40,   # will prefer MUB inside bond bucket
            "cash_min":       0.00, "cash_max":       0.10,
            "alts_min":       0.00, "alts_max":       0.20,
        }
    # default to grow
    return _bounds_by_objective("grow")

def _target_vol_from_risk_pct(objective: str, risk_pct: float) -> float:
    """
    Map [0..100] slider to target annualized volatility.
    Ranges tuned per objective (you can retune later).
    """
    rp = max(0.0, min(100.0, float(risk_pct)))
    def lerp(a,b,t): return a + (b-a)*(t/100.0)
    if objective == "grow":
        return lerp(0.14, 0.28, rp)   # 14% @ 0  → 28% @ 100
    if objective == "grow_income":
        return lerp(0.10, 0.22, rp)
    if objective == "income":
        return lerp(0.06, 0.16, rp)
    if objective == "preserve":
        return lerp(0.04, 0.10, rp)
    if objective == "tax_efficiency":
        return lerp(0.10, 0.24, rp)
    return lerp(0.12, 0.24, rp)

def _is_equity(class_name: str) -> bool:
    return class_name.startswith("public_equity") or class_name in {"public_equity", "public_equity_intl", "public_equity_em"}

def _is_bond(class_name: str) -> bool:
    return class_name in {"treasury_long","tbills","corporate_bond","high_yield","tax_eff_muni","public_bond","muni_bond"}

def _is_cash(symbol: str, class_name: str) -> bool:
    return symbol.upper() in {"BIL","SHY"} or class_name in {"tbills"}

def _is_alt(class_name: str, symbol: str) -> bool:
    return class_name in {"commodities","gold","reit","public_equity_sector","public_equity_style"} or symbol.upper() in {"GLD","DBC","VNQ"}

def _symbol_bounds_from_class_limits(symbols, catalog, limits):
    # Distribute class min/max evenly across symbols in that class (simple; can be refined).
    from collections import defaultdict
    c2s = defaultdict(list)
    for s in symbols:
        c2s[_class_of_symbol(s, catalog)].append(s)

    sbounds = {}
    # Compute totals per bucket
    equity = [s for s in symbols if _is_equity(_class_of_symbol(s, catalog))]
    bonds  = [s for s in symbols if _is_bond(_class_of_symbol(s, catalog))]
    cash   = [s for s in symbols if _is_cash(s, _class_of_symbol(s, catalog))]
    alts   = [s for s in symbols if _is_alt(_class_of_symbol(s, catalog), s)]

    def spread(items, min_total, max_total, hard_cap=None):
        n = max(1, len(items))
        min_each = min_total / n
        max_each = (hard_cap if hard_cap is not None else max_total / n)
        return {s: (min_each, max_each) for s in items}

    sbounds.update(spread(equity, limits["equity_all_min"], limits["equity_all_max"]))
    sbounds.update(spread(bonds,  limits["bond_all_min"],   limits["bond_all_max"]))
    sbounds.update(spread(cash,   limits["cash_min"],       limits["cash_max"], hard_cap=limits["cash_max"]))
    sbounds.update(spread(alts,   limits["alts_min"],       limits["alts_max"]))

    # Fill any missing with (0, 1) then clip to [0,1]
    for s in symbols:
        if s not in sbounds:
            sbounds[s] = (0.0, 1.0)
        lo, hi = sbounds[s]
        sbounds[s] = (max(0.0, lo), min(1.0, hi))
    return sbounds

def _repair_bounds(bounds_list):
    """Make bounds feasible and numerically safe for the solver.
    Ensures: 0<=lo<=hi<=1, sum(los)<=1<=sum(his). If sums violate, relax proportionally.
    """
    if not bounds_list:
        return []
    lows, highs = [], []
    for lo, hi in bounds_list:
        lo = 0.0 if lo is None else float(lo)
        hi = 1.0 if hi is None else float(hi)
        lo = max(0.0, min(1.0, lo))
        hi = max(0.0, min(1.0, hi))
        if lo > hi:
            lo, hi = hi, lo
        lows.append(lo)
        highs.append(hi)
    sum_lo, sum_hi = sum(lows), sum(highs)
    n = len(lows)
    # If minimums exceed 1, scale them down proportionally
    if sum_lo > 1.0 and sum_lo > 0:
        scale = 1.0 / sum_lo
        lows = [lo * scale for lo in lows]
        sum_lo = sum(lows)
    # If maximums sum to < 1, relax them up
    if sum_hi < 1.0:
        add = (1.0 - sum_hi) / n
        highs = [min(1.0, hi + add) for hi in highs]
        sum_hi = sum(highs)
        if sum_hi < 1.0:
            # last resort: allow all highs to be 1.0
            highs = [1.0] * n
    return list(zip(lows, highs))


from pypfopt.hierarchical_portfolio import HRPOpt

def _hrp_weights(rets_df, symbols):
    # HRP ignores per-asset bounds natively, but it is very robust and diversified.
    # We’ll compute HRP weights and normalize; later we can soft-enforce class mins if needed.
    hrp = HRPOpt(rets_df.fillna(0))
    w = hrp.optimize()
    # Coerce to {symbol: weight}
    w = {k: float(v) for k,v in w.items() if k in symbols}
    tot = sum(v for v in w.values() if v is not None)
    if tot > 0:
        w = {k: v/tot for k,v in w.items()}
    else:
        # fallback equal-weight
        n = len(symbols) or 1
        w = {symbols[i]: 1.0/n for i in range(n)}
    return w
# ---- Safe weight extraction helper ----
import numpy as np
from pypfopt import EfficientFrontier

def _safe_get_weights(ef, symbols):
    """Extract weights safely; fall back to equal-weight if optimizer failed."""
    try:
        w = ef.clean_weights()
    except Exception:
        n = len(symbols)
        return {} if n == 0 else {symbols[i]: 1.0 / n for i in range(n)}
    w = {k: max(0.0, float(v)) for k, v in w.items() if v is not None}
    total = sum(w.values())
    if not total or not np.isfinite(total):
        n = len(symbols)
        return {} if n == 0 else {symbols[i]: 1.0 / n for i in range(n)}
    return {k: v / total for k, v in w.items()}

# ---- Primary recommender (HRP/MVO with class bounds and risk slider) ----
def recommend(returns, profile, objective="grow", risk_pct=50, catalog=None, method="hrp"):
    """
    returns: DataFrame of daily returns (%), columns = symbols, index = dates
    profile: UserProfile (kept for interface; not deeply used here)
    objective: 'grow'|'grow_income'|'income'|'preserve'|'tax_efficiency'
    risk_pct: 0..100 from UI slider (maps to target annual vol per objective)
    catalog: dict with asset classes (for class-aware bounds)
    method: 'hrp' (Hierarchical Risk Parity) or 'mvo' (mean-variance)
    """
    import pandas as pd
    from pypfopt import risk_models, expected_returns

    if returns is None or len(returns) == 0:
        # Nothing to optimize – trivial empty portfolio
        return {"weights": {}, "metrics": {}, "curve": pd.Series(dtype=float)}

    # Clean and keep only live series
    rets = returns.dropna(how="all").copy()
    rets = rets.loc[:, rets.std() > 0]
    symbols = list(rets.columns)
    if not symbols:
        return {"weights": {}, "metrics": {}, "curve": pd.Series(dtype=float)}

    # Bounds derived from objective and catalog classes
    cat = catalog if catalog is not None else {"assets": []}
    limits = _bounds_by_objective(objective)
    sbounds = _symbol_bounds_from_class_limits(symbols, cat, limits)
    bounds_list = _repair_bounds([sbounds[s] for s in symbols])

    # Expected returns & covariance (with shrinkage when available)
    mu = expected_returns.mean_historical_return(rets, frequency=252)
    try:
        from pypfopt.risk_models import CovarianceShrinkage
        S = CovarianceShrinkage(rets).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(rets, frequency=252)

    target_vol = _target_vol_from_risk_pct(objective, float(risk_pct))

    # Solve for weights
    method_lc = (method or "").lower()
    if method_lc == "hrp":
        w = _hrp_weights(rets, symbols)
    else:
        ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        # Apply per-asset bounds via linear constraints
        lbs = np.array([b[0] for b in bounds_list], dtype=float)
        ubs = np.array([b[1] for b in bounds_list], dtype=float)
        try:
            ef.add_constraint(lambda w: w >= lbs)
            ef.add_constraint(lambda w: w <= ubs)
        except Exception:
            # If constraint add fails in older versions, proceed with global bounds
            pass
        solved = False
        for solve_call in (
            lambda: ef.efficient_risk(target_volatility=target_vol),
            lambda: ef.max_sharpe(),
        ):
            if solved:
                break
            try:
                solve_call()
                solved = True
            except Exception:
                continue
        w = _safe_get_weights(ef, symbols)

    # --- Pre-constraint fallback & repair ---
    # If optimizer produced empty / invalid weights, fall back to equal-weight BEFORE constraints
    if (not w) or sum(v for v in w.values() if pd.notna(v)) <= 0:
        n = len(symbols)
        w = {symbols[i]: 1.0 / n for i in range(n)}
    else:
        # Coerce negatives / non-finite to zero then renormalize
        w = {k: (float(v) if (v is not None and pd.notna(v) and float(v) > 0) else 0.0) for k,v in w.items()}
        totw = sum(w.values())
        if totw <= 0:
            n = len(symbols)
            w = {symbols[i]: 1.0 / n for i in range(n)}
        else:
            w = {k: v / totw for k, v in w.items()}

    # Build portfolio equity curve and metrics
    weights_vec = np.array([w.get(c, 0.0) for c in rets.columns])
    port_ret = rets.fillna(0).dot(weights_vec)
    curve = (1 + port_ret).cumprod()

    if len(curve) == 0:
        metrics = {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "N": 0}
        context = {
            "receipts": {
                "provider_map": getattr(returns, "_provider_map", {}),
                "backfill_pct": getattr(returns, "_backfill_pct", {}),
                "coverage": getattr(returns, "_coverage", {})
            }
        }
        return {"weights": w, "metrics": metrics, "curve": curve, "context": context}

    ann_ret = float(curve.iloc[-1] ** (252 / len(curve)) - 1.0)
    ann_vol = float(port_ret.std() * (252 ** 0.5))
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
    peak = curve.cummax()
    maxdd = float((curve / peak - 1.0).min())

    metrics = {
        "CAGR": round(ann_ret, 4),
        "Vol": round(ann_vol, 4),
        "Sharpe": round(sharpe, 2),
        "MaxDD": round(maxdd, 4),
        "N": int(len(curve)),
    }

    # --- Core/Satellite constraint enforcement (post-optimization) ---
    try:
        from core.portfolio.constraints import apply_weight_constraints
        core_classes = {"public_equity","public_equity_intl","public_equity_em","treasury_long","corporate_bond","high_yield","tax_eff_muni","tbills"}
        sat_classes = {"commodities","gold","reit","public_equity_sector","public_equity_style"}
        CORE = [s for s in symbols if _class_of_symbol(s, cat) in core_classes]
        SATS = [s for s in symbols if _class_of_symbol(s, cat) in sat_classes]
        w = apply_weight_constraints(
            w,
            core_symbols=CORE,
            satellite_symbols=SATS,
            core_min=0.65,
            satellite_max=0.35,
            single_max=0.07,
        )
        # Final renormalization – if all weights zero after constraints, revert to equal-weight
        total_w = sum(max(0.0, float(v)) for v in w.values())
        if total_w <= 0:
            n = len(symbols)
            w = {symbols[i]: 1.0 / n for i in range(n)}
        else:
            w = {k: float(v) / total_w for k, v in w.items()}
    except Exception:
        pass
    
    context = {
        "receipts": {
            "provider_map": getattr(returns, "_provider_map", {}),
            "backfill_pct": getattr(returns, "_backfill_pct", {}),
            "coverage": getattr(returns, "_coverage", {})
        }
    }
    
    return {"weights": w, "metrics": metrics, "curve": curve, "context": context}

# ---- Per-ticker statistics used by UI tables ----
def compute_asset_stats(prices_df):
    """Input: tidy DF with columns [date, ticker, price]. Returns per-ticker stats DF."""
    import numpy as np, pandas as pd
    if prices_df is None or prices_df.empty:
        return pd.DataFrame()
    df = prices_df.copy().sort_values(["ticker", "date"]).reset_index(drop=True)
    df["ret"] = df.groupby("ticker")["price"].pct_change()

    last_price = df.groupby("ticker")["price"].last().rename("last_price")

    def total_ret(x, days):
        if len(x) <= days:
            return np.nan
        return (x.iloc[-1] / x.iloc[-days - 1]) - 1.0

    rows = []
    for tkr, sub in df.groupby("ticker", sort=False):
        pr = sub["price"].reset_index(drop=True)
        r1m = total_ret(pr, 21)
        r1y = total_ret(pr, 252)
        rets = sub["ret"].dropna()
        ann_vol = float(rets.std() * (252 ** 0.5)) if not rets.empty else np.nan
        curve = (1 + rets).cumprod()
        peak = curve.cummax()
        max_dd = float(((curve / peak) - 1.0).min()) if not curve.empty else np.nan
        rows.append([tkr, float(last_price.get(tkr, np.nan)), r1m, r1y, ann_vol, max_dd])

    out = pd.DataFrame(rows, columns=["ticker", "last_price", "1m_ret", "1y_ret", "ann_vol", "max_dd"]).set_index("ticker")
    return out


def _clean_returns_df(rets, winsorize_pct=0.0):
    """Return a numeric, finite daily-returns DataFrame.
    - Casts to float
    - Drops columns with zero variance
    - Drops any rows with non-finite values
    - Optional symmetric winsorization at winsorize_pct (e.g. 0.005 = 0.5%)"""
    import numpy as np
    import pandas as pd
    if rets is None or len(rets) == 0:
        return rets
    df = rets.copy()
    # keep only numeric columns
    df = df.select_dtypes(include=["number"]).astype("float64")
    # drop dead series
    df = df.loc[:, df.std(skipna=True) > 0]
    # winsorize if requested
    if winsorize_pct and winsorize_pct > 0:
        lo = df.quantile(winsorize_pct)
        hi = df.quantile(1 - winsorize_pct)
        df = df.clip(lower=lo, upper=hi, axis=1)
    # drop non-finite rows
    df = df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df


def anova_mean_returns(port_ret, groups):
    """One-way ANOVA on mean returns across groups.
    port_ret: pd.Series of daily returns (float), indexed by date
    groups:   pd.Series (same index) of categorical labels (e.g., 'hike','cut','hold')
    Returns dict with F-statistic and p-value.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats

    df = pd.DataFrame({"r": port_ret}).join(groups.rename("g"))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty or df["g"].nunique() < 2:
        return {"F": np.nan, "p": np.nan, "k": int(df["g"].nunique())}

    samples = [grp["r"].values for _, grp in df.groupby("g")]
    F, p = stats.f_oneway(*samples)
    return {"F": float(F), "p": float(p), "k": int(df["g"].nunique())}

__all__ = ["UserProfile", "recommend", "recommend_legacy", "compute_asset_stats", "target_vol_for"]
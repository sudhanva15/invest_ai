"""
Multi-factor asset and portfolio filtering engine.

This module implements the asset-first pipeline:
1. Evaluate per-asset metrics (Sharpe, vol, max DD, history)
2. Filter assets based on quality thresholds and risk profile
3. Compute portfolio-level metrics (CAGR, vol, Sharpe, risk contributions)
4. Filter portfolios using multi-factor checks

All filtering uses RiskProfileResult as the single source of truth.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np


# ============================================================================
# Asset-Level Metrics & Filters
# ============================================================================

def evaluate_asset_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.015,
) -> dict:
    """
    Compute comprehensive metrics for a single asset.
    
    Args:
        returns: Daily returns series for the asset
        risk_free_rate: Annual risk-free rate for Sharpe calculation (default 1.5%)
    
    Returns:
        Dict with keys:
            - years_history: Years of data available
            - cagr: Compound annual growth rate
            - volatility: Annualized volatility
            - sharpe: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown (negative value)
            - total_return: Cumulative return
            - valid: Whether metrics could be computed
    
    Uses existing formulas from audit Section 13.2:
        CAGR = (V_T / V_0)^(252/T) - 1
        σ_annual = σ_daily * sqrt(252)
        Sharpe = (CAGR - r_f) / σ_annual
    """
    try:
        if returns is None or len(returns) < 2:
            return {"valid": False, "reason": "insufficient data"}
        
        rets = returns.dropna()
        if len(rets) < 2:
            return {"valid": False, "reason": "insufficient data after dropna"}
        
        # Years of history
        years_history = len(rets) / 252.0
        
        # Equity curve
        equity = (1 + rets).cumprod()
        
        # CAGR
        total_return = equity.iloc[-1] - 1.0
        if equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
            return {"valid": False, "reason": "invalid equity curve"}
        
        cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (252.0 / len(rets)) - 1.0)
        
        # Volatility (annualized)
        volatility = float(rets.std() * np.sqrt(252))
        
        # Sharpe ratio
        sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Max drawdown
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = float(drawdown.min())
        
        return {
            "valid": True,
            "years_history": float(years_history),
            "cagr": float(cagr),
            "volatility": float(volatility),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_return": float(total_return),
        }
    
    except Exception as e:
        return {"valid": False, "reason": str(e)}


def asset_passes_filters(
    symbol: str,
    metrics: dict,
    asset_class: str,
    core_or_satellite: str,
    cfg: dict,
    risk_profile,  # RiskProfileResult
) -> Tuple[bool, str | None]:
    """
    Determine if an asset passes quality filters.
    
    Filters applied:
    1. Minimum history (core_min_years or sat_min_years from config)
    2. Minimum Sharpe ratio (config-driven)
    3. Maximum volatility (must not exceed 2x risk profile band_max_vol)
    4. Maximum drawdown threshold (config-driven)
    
    Args:
        symbol: Asset ticker
        metrics: Output from evaluate_asset_metrics()
        asset_class: Asset class from catalog ("equity_us", "bond_treasury", etc.)
        core_or_satellite: "core" or "satellite" from catalog
        cfg: Config dict with universe and multifactor settings
        risk_profile: RiskProfileResult with volatility bands
    
    Returns:
        (passes, reason_if_failed)
        - passes: True if asset passes all filters
        - reason_if_failed: String explaining why it failed, or None if passed
    
    Config keys used:
        universe.core_min_years: Min years for core assets (default 5.0)
        universe.sat_min_years: Min years for satellite assets (default 3.0)
        multifactor.min_asset_sharpe: Min Sharpe for any asset (default -0.5)
        multifactor.max_asset_vol_multiplier: Max vol as multiple of risk band (default 2.0)
        multifactor.max_asset_drawdown: Max acceptable drawdown (default -0.60)
    """
    if not metrics.get("valid"):
        return False, f"invalid metrics: {metrics.get('reason', 'unknown')}"
    
    # 1. History requirement
    universe_cfg = cfg.get("universe", {})
    core_min = float(universe_cfg.get("core_min_years", 5.0))
    sat_min = float(universe_cfg.get("sat_min_years", 3.0))
    
    min_years = core_min if core_or_satellite == "core" else sat_min
    if metrics["years_history"] < min_years:
        return False, f"insufficient history: {metrics['years_history']:.1f}y < {min_years:.1f}y required"
    
    # 2. Minimum Sharpe
    mf_cfg = cfg.get("multifactor", {})
    min_sharpe = float(mf_cfg.get("min_asset_sharpe", -0.5))
    if metrics["sharpe"] < min_sharpe:
        return False, f"low Sharpe: {metrics['sharpe']:.2f} < {min_sharpe:.2f}"
    
    # 3. Maximum volatility (relative to risk profile)
    max_vol_mult = float(mf_cfg.get("max_asset_vol_multiplier", 2.0))
    max_vol_threshold = risk_profile.band_max_vol * max_vol_mult
    if metrics["volatility"] > max_vol_threshold:
        return False, f"excessive volatility: {metrics['volatility']:.2%} > {max_vol_threshold:.2%}"
    
    # 4. Maximum drawdown
    max_dd = float(mf_cfg.get("max_asset_drawdown", -0.60))
    if metrics["max_drawdown"] < max_dd:
        return False, f"excessive drawdown: {metrics['max_drawdown']:.1%} < {max_dd:.1%}"
    
    return True, None


def build_filtered_universe(
    universe_symbols: list[str],
    returns: pd.DataFrame,
    catalog: dict,
    cfg: dict,
    risk_profile,  # RiskProfileResult
    objective_config=None,
) -> Tuple[pd.Index, pd.DataFrame]:
    """
    Filter asset universe using multi-factor quality checks.
    
    Process:
    1. For each symbol in universe:
        a. Compute asset metrics
        b. Apply asset_passes_filters
        c. Record pass/fail and reason
    2. Return filtered symbols + full receipts DataFrame
    
    Args:
        universe_symbols: List of symbols to evaluate
        returns: DataFrame of daily returns (columns = symbols)
        catalog: Assets catalog dict (symbol → metadata)
        cfg: Config dict
    risk_profile: RiskProfileResult
    objective_config: Optional ObjectiveConfig (reserved for future constraint wiring)
    
    Returns:
        (filtered_symbols, receipts_df)
        - filtered_symbols: pd.Index of symbols that passed
        - receipts_df: DataFrame with columns:
            [symbol, asset_class, core_satellite, years, cagr, vol, sharpe, 
             max_dd, passed, fail_reason]
    
    Usage:
        from core.risk_profile import compute_risk_profile
        from core.multifactor import build_filtered_universe
        
        profile = compute_risk_profile(...)
        filtered_syms, receipts = build_filtered_universe(
            universe_symbols=universe,
            returns=returns_df,
            catalog=catalog,
            cfg=config,
            risk_profile=profile,
            objective_config=my_objective_config,
        )
        
        # Use filtered_syms for optimization
        # Use receipts for diagnostics/debug page
    """
    receipts = []

    # Build symbol metadata map (catalog may be list-style, so normalize)
    symbol_meta = {}
    try:
        assets = catalog.get("assets", []) if isinstance(catalog, dict) else []
        for a in assets:
            try:
                s = str(a.get("symbol", "")).upper()
                if not s:
                    continue
                asset_cls = a.get("asset_class") or a.get("class") or "unknown"
                core_sat = a.get("core_or_satellite", "satellite") or "satellite"
                symbol_meta[s] = {"asset_class": str(asset_cls), "core_or_satellite": str(core_sat)}
            except Exception:
                continue
    except Exception:
        pass
    # Heuristic fallback for symbols not in catalog
    def _heuristic_ac(sym: str) -> str:
        s = sym.upper()
        if s in {"GLD","IAU","GLDM"}: return "commodity"
        if s in {"DBC","PDBC","GSG","GCC"}: return "commodity"
        if s in {"VNQ","SCHH","IYR"}: return "reit"
        if s in {"BIL","SHY"}: return "cash"
        if s in {"AGG","BND","LQD","HYG","MUB","TLT","IEF","IEI","TIP","SCHP","BSV","BIV","VGIT","VGLT","VGSH","EDV","ZROZ","GOVT","IGSB","IGIB","JNK","BNDX","EMB"}: return "bond"
        if s in {"SPY","VOO","IVV","VTI","ITOT","SCHB","SPTM","RSP","SPLG","SCHX","QQQ","QQQM","IWM","IJR","IJH","IWB","VUG","VTV","VBK","VBR","AVUV","MTUM","QUAL","USMV","SPLV","SCHD","VIG","VYM","DGRO","SPYG","SPYV","IWF","IWD","VLUE","SIZE","VEA","EFA","VXUS","VEU","ACWI","ACWX","IEFA","IEMG","VWO","EEM"}: return "equity"
        return "unknown"

    for symbol in universe_symbols:
        sym_u = symbol.upper()
        meta = symbol_meta.get(sym_u, {})
        asset_class = meta.get("asset_class") or _heuristic_ac(sym_u)
        core_sat = meta.get("core_or_satellite") or "satellite"
        
        # Check if symbol exists in returns
        if symbol not in returns.columns:
            receipts.append({
                "symbol": symbol,
                "asset_class": asset_class,
                "core_satellite": core_sat,
                "years": 0.0,
                "cagr": None,
                "vol": None,
                "sharpe": None,
                "max_dd": None,
                "passed": False,
                "fail_reason": "not in returns data"
            })
            continue
        
        # Compute metrics
        asset_returns = returns[symbol].dropna()
        metrics = evaluate_asset_metrics(asset_returns, cfg.get("optimization", {}).get("risk_free_rate", 0.015))
        
        # Apply filters
        passed, reason = asset_passes_filters(
            symbol=symbol,
            metrics=metrics,
            asset_class=asset_class,
            core_or_satellite=core_sat,
            cfg=cfg,
            risk_profile=risk_profile
        )
        
        receipts.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "core_satellite": core_sat,
            "years": metrics.get("years_history", 0.0),
            "cagr": metrics.get("cagr"),
            "vol": metrics.get("volatility"),
            "sharpe": metrics.get("sharpe"),
            "max_dd": metrics.get("max_drawdown"),
            "passed": passed,
            "fail_reason": reason
        })
    
    receipts_df = pd.DataFrame(receipts)
    filtered_symbols = receipts_df[receipts_df["passed"]]["symbol"].values
    
    return pd.Index(filtered_symbols), receipts_df


# ============================================================================
# Portfolio-Level Metrics & Filters
# ============================================================================

def portfolio_metrics(
    weights: pd.Series,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.015,
) -> dict:
    """
    Compute comprehensive metrics for a portfolio.
    
    Args:
        weights: Series of weights (index = symbols, values = weights)
        returns: DataFrame of daily returns (columns = symbols)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dict with keys:
            - cagr: Portfolio CAGR
            - volatility: Annualized portfolio volatility
            - sharpe: Portfolio Sharpe ratio
            - max_drawdown: Portfolio max drawdown
            - total_return: Cumulative return
            - diversification_ratio: Sum of weighted vols / portfolio vol
            - num_holdings: Number of non-zero positions
            - valid: Whether metrics computed successfully
    
    Formula (audit Section 13.1.3):
        r_p,t = w^T * r_t (portfolio return)
        Equity_t = prod(1 + r_p,k) for k=1..t
    """
    try:
        # Align weights and returns
        common_symbols = weights.index.intersection(returns.columns)
        if len(common_symbols) == 0:
            return {"valid": False, "reason": "no common symbols"}
        
        w = weights[common_symbols]
        w = w / w.sum()  # Normalize
        
        rets = returns[common_symbols].dropna()
        if len(rets) < 2:
            return {"valid": False, "reason": "insufficient returns data"}
        
        # Portfolio returns
        portfolio_returns = (rets * w).sum(axis=1)
        
        # Use evaluate_asset_metrics on portfolio returns
        port_metrics = evaluate_asset_metrics(portfolio_returns, risk_free_rate)
        if not port_metrics.get("valid"):
            return port_metrics
        
        # Diversification ratio
        # DR = (sum of weighted individual vols) / portfolio vol
        individual_vols = rets.std() * np.sqrt(252)
        weighted_vol_sum = (w * individual_vols).sum()
        div_ratio = weighted_vol_sum / port_metrics["volatility"] if port_metrics["volatility"] > 0 else 1.0
        
        # Number of holdings
        num_holdings = int((w > 0.0001).sum())
        
        return {
            "valid": True,
            "cagr": port_metrics["cagr"],
            "volatility": port_metrics["volatility"],
            "sharpe": port_metrics["sharpe"],
            "max_drawdown": port_metrics["max_drawdown"],
            "total_return": port_metrics["total_return"],
            "diversification_ratio": float(div_ratio),
            "num_holdings": num_holdings,
        }
    
    except Exception as e:
        return {"valid": False, "reason": str(e)}


def compute_risk_contributions(
    weights: pd.Series,
    cov: pd.DataFrame,
) -> pd.Series:
    """
    Compute risk contribution for each asset in portfolio.
    
    Formula (audit Section 13.5.4):
        RC_i = w_i * (Σw)_i / σ_p
        
        where:
        - Σw = covariance matrix times weight vector
        - σ_p = portfolio volatility
    
    Args:
        weights: Series of weights (index = symbols)
        cov: Covariance matrix (daily, not annualized)
    
    Returns:
        Series of risk contributions (index = symbols, values = RC_i)
        Sum of risk contributions equals portfolio volatility.
    
    Usage:
        risk_contrib = compute_risk_contributions(weights, cov_matrix)
        max_rc = risk_contrib.max()  # Check if any asset dominates risk
    """
    try:
        # Align weights and covariance matrix
        common = weights.index.intersection(cov.index).intersection(cov.columns)
        if len(common) == 0:
            return pd.Series(dtype=float)
        
        w = weights[common].values
        S = cov.loc[common, common].values
        
        # Portfolio variance
        port_var = w.T @ S @ w
        port_vol = np.sqrt(port_var)
        
        if port_vol <= 0:
            return pd.Series(0.0, index=common)
        
        # Marginal risk contributions: (Σw)_i / σ_p
        marginal_rc = (S @ w) / port_vol
        
        # Risk contributions: w_i * marginal_rc_i
        risk_contrib = w * marginal_rc
        
        return pd.Series(risk_contrib, index=common)
    
    except Exception:
        return pd.Series(dtype=float)


def derive_portfolio_thresholds(risk_profile, cfg: dict) -> dict:
    """
    Derive risk-adaptive portfolio filtering thresholds from risk profile.
    
    Instead of uniform static thresholds, this function maps the user's
    TRUE_RISK score → dynamic min_cagr, min_sharpe, max_drawdown, vol bands.
    
    Key mapping philosophy:
        - Conservative profiles (risk ~20): Accept lower CAGR/Sharpe, stricter drawdown
        - Moderate profiles (risk ~50): Middle ground
        - Aggressive profiles (risk ~80): Demand higher CAGR/Sharpe, tolerate larger drawdown
    
    Args:
        risk_profile: RiskProfileResult with cagr_min, cagr_target, vol bands, true_risk
        cfg: Config dict with multifactor section (provides baseline thresholds)
    
    Returns:
        Dict with:
            min_cagr: Minimum acceptable CAGR (from risk_profile.cagr_min)
            min_sharpe: Risk-scaled Sharpe threshold
            max_drawdown: Risk-scaled max drawdown (more negative = tolerate worse DD)
            vol_lower: Soft lower volatility bound (from risk_profile + soft factor)
            vol_upper: Upper volatility bound (from risk_profile)
            vol_band_min: Hard band minimum (risk_profile.band_min_vol)
            vol_band_max: Hard band maximum (risk_profile.band_max_vol)
    
    Example:
        Conservative (risk=20): min_cagr=0.05, min_sharpe=0.25, max_drawdown=-0.35
        Moderate (risk=50): min_cagr=0.08, min_sharpe=0.35, max_drawdown=-0.50
        Aggressive (risk=80): min_cagr=0.10, min_sharpe=0.40, max_drawdown=-0.65
    
    Phase 3 Context:
        This function enables the "relaxed filter" stage (Stage 2) in the fallback ladder.
        When strict filters produce no candidates, Stage 2 calls this to get looser
        thresholds that align with user risk tolerance rather than uniform cutoffs.
    """
    mf_cfg = cfg.get("multifactor", {})
    
    # CAGR: Use risk-derived band directly, then relax slightly for fallback stage
    min_cagr = risk_profile.cagr_min  # Already computed by map_true_risk_to_cagr_band()
    
    # Sharpe: Scale from baseline (0.3) based on risk
    # Conservative → lower Sharpe acceptable (0.25), Aggressive → higher demanded (0.45)
    baseline_sharpe = float(mf_cfg.get("min_portfolio_sharpe", 0.3))
    risk_norm = risk_profile.true_risk / 100.0  # [0, 1]
    # Linear scaling: 0.25 @ risk=0, 0.45 @ risk=100
    # Formula: sharpe = baseline + (risk_norm - 0.5) * sharpe_range
    sharpe_range = 0.20  # ±0.10 around 0.35 midpoint
    min_sharpe = baseline_sharpe + (risk_norm - 0.5) * sharpe_range
    min_sharpe = max(0.15, min(min_sharpe, 0.50))  # Clamp [0.15, 0.50]
    
    # Max Drawdown: Scale tolerance with risk
    # Conservative → stricter (-0.30), Moderate → -0.50, Aggressive → tolerate (-0.70)
    baseline_dd = float(mf_cfg.get("max_portfolio_drawdown", -0.50))
    # More risk → more negative (tolerate worse drawdown)
    dd_range = 0.40  # Range from -0.30 to -0.70
    max_drawdown = baseline_dd - (risk_norm - 0.5) * dd_range
    max_drawdown = max(-0.80, min(max_drawdown, -0.25))  # Clamp [-0.80, -0.25]
    
    # Volatility: Use risk_profile bands with soft lower bound
    soft_factor = float(mf_cfg.get("vol_soft_lower_factor", 0.6))
    soft_factor = max(0.3, min(soft_factor, 0.9))  # Safety clamp
    vol_lower = risk_profile.band_min_vol * soft_factor
    vol_upper = risk_profile.band_max_vol

    # Stage 2 adaptive relaxation: allow moderate loosening based on risk centrality
    # (mid-risk users benefit most; extremes still keep tighter guardrails)
    risk_centrality = max(0.0, 1.0 - abs(risk_norm - 0.5) * 2.0)  # 1 at mid-risk, 0 at extremes
    relax_cfg = mf_cfg.get("stage2_relaxation", {})
    min_cagr_floor = float(relax_cfg.get("min_cagr_floor", 0.045))
    base_cagr_mult = float(relax_cfg.get("cagr_multiplier_base", 0.82))
    cagr_bonus = float(relax_cfg.get("cagr_multiplier_bonus", 0.10))
    cagr_multiplier = base_cagr_mult + cagr_bonus * risk_centrality
    min_cagr = max(min_cagr_floor, min_cagr * cagr_multiplier)

    base_sharpe_mult = float(relax_cfg.get("sharpe_multiplier_base", 0.88))
    sharpe_bonus = float(relax_cfg.get("sharpe_multiplier_bonus", 0.07))
    min_sharpe = max(0.15, min_sharpe * (base_sharpe_mult + sharpe_bonus * risk_centrality))

    drawdown_shift_base = float(relax_cfg.get("drawdown_shift_base", 0.03))
    drawdown_shift_bonus = float(relax_cfg.get("drawdown_shift_bonus", 0.04))
    max_drawdown = max(
        -0.90,
        min(max_drawdown - (drawdown_shift_base + drawdown_shift_bonus * risk_centrality), -0.20),
    )

    vol_lower_multiplier = float(relax_cfg.get("vol_lower_multiplier", 0.65))
    # Encourage low/medium risk users to still see options by relaxing more aggressively near center
    vol_lower_bonus = float(relax_cfg.get("vol_lower_bonus", 0.15))
    relaxed_vol_lower = vol_lower * (vol_lower_multiplier + vol_lower_bonus * risk_centrality)
    vol_lower = max(float(relax_cfg.get("vol_lower_floor", 0.02)), relaxed_vol_lower)
    
    return {
        "min_cagr": min_cagr,
        "min_sharpe": min_sharpe,
        "max_drawdown": max_drawdown,
        "vol_lower": vol_lower,
        "vol_upper": vol_upper,
        "vol_band_min": risk_profile.band_min_vol,
        "vol_band_max": risk_profile.band_max_vol,
    }


def portfolio_passes_filters(
    portfolio_stats: dict,
    risk_contrib: pd.Series,
    cfg: dict,
    risk_profile,  # RiskProfileResult
    dynamic_thresholds: dict | None = None,  # NEW: Override thresholds for adaptive filtering
) -> Tuple[bool, str | None]:
    """
    Determine if portfolio passes multi-factor quality checks.
    
    Filters applied:
    1. Portfolio Sharpe >= min threshold
    2. Volatility within risk profile band [band_min_vol, band_max_vol]
    3. Max drawdown >= min threshold (not worse than threshold)
    4. CAGR >= min threshold (NEW in Phase 3: risk-adaptive growth filter)
    5. Max risk contribution <= threshold (risk parity check)
    6. Diversification ratio >= min threshold
    7. Minimum number of holdings
    
    Args:
        portfolio_stats: Output from portfolio_metrics()
        risk_contrib: Output from compute_risk_contributions()
        cfg: Config dict with multifactor settings
        risk_profile: RiskProfileResult with volatility bands
        dynamic_thresholds: Optional dict from derive_portfolio_thresholds() to override
            static config values. Used for adaptive filtering (Stage 2 fallback).
            Expected keys: min_cagr, min_sharpe, max_drawdown, vol_lower, vol_upper
    
    Returns:
        (passes, reason_if_failed)
    
    Config keys used (all under multifactor section):
        min_portfolio_sharpe: default 0.3
        max_portfolio_drawdown: default -0.50
        max_risk_contribution: default 0.40 (40% of total risk)
        min_diversification_ratio: default 1.2
        min_holdings: default 3
    
    Phase 3 Context:
        When dynamic_thresholds is provided, this function uses risk-adaptive
        cutoffs instead of uniform config values. This enables "relaxed filtering"
        for Stage 2 fallback while maintaining quality control.
    """
    if not portfolio_stats.get("valid"):
        return False, f"invalid portfolio metrics: {portfolio_stats.get('reason', 'unknown')}"
    
    mf_cfg = cfg.get("multifactor", {})
    
    # Decide thresholds: use dynamic if provided, else fall back to config
    if dynamic_thresholds:
        min_sharpe = dynamic_thresholds["min_sharpe"]
        max_dd_threshold = dynamic_thresholds["max_drawdown"]
        vol_lower = dynamic_thresholds["vol_lower"]
        vol_upper = dynamic_thresholds["vol_upper"]
        min_cagr = dynamic_thresholds["min_cagr"]
    else:
        min_sharpe = float(mf_cfg.get("min_portfolio_sharpe", 0.3))
        max_dd_threshold = float(mf_cfg.get("max_portfolio_drawdown", -0.50))
        soft_factor = float(mf_cfg.get("vol_soft_lower_factor", 0.6))
        if soft_factor <= 0 or soft_factor >= 1:
            soft_factor = 0.6
        vol_lower = risk_profile.band_min_vol * soft_factor
        vol_upper = risk_profile.band_max_vol
        min_cagr = float(mf_cfg.get("min_portfolio_cagr", 0.0))
    
    # 1. Minimum Sharpe
    if portfolio_stats["sharpe"] < min_sharpe:
        return False, f"low Sharpe: {portfolio_stats['sharpe']:.2f} < {min_sharpe:.2f}"
    
    # 2. Volatility band (soft lower band logic)
    vol = portfolio_stats["volatility"]
    # Upper bound hard fail
    if vol > vol_upper:
        return False, f"volatility too high: {vol:.2%} > {vol_upper:.2%}"
    # Hard fail if below soft_lower
    if vol < vol_lower:
        return False, f"volatility too low: {vol:.2%} < {vol_lower:.2%} (soft_lower)"
    # Soft pass zone if between soft_lower and band_min_vol
    if vol < risk_profile.band_min_vol:
        portfolio_stats["soft_band"] = True  # annotate for downstream scoring penalty
    else:
        portfolio_stats["soft_band"] = False
    
    # 3. Max drawdown
    if portfolio_stats["max_drawdown"] < max_dd_threshold:
        return False, f"excessive drawdown: {portfolio_stats['max_drawdown']:.1%} < {max_dd_threshold:.1%}"
    
    # 4. NEW: CAGR filter (risk-adaptive growth expectation)
    if "cagr" in portfolio_stats and portfolio_stats["cagr"] < min_cagr:
        return False, f"low CAGR: {portfolio_stats['cagr']:.2%} < {min_cagr:.2%}"
    
    # 4. Risk contribution (risk parity check)
    if len(risk_contrib) > 0:
        # Risk contributions should sum to portfolio volatility (same time scale)
        # Express each RC as fraction of total portfolio risk
        port_vol = portfolio_stats["volatility"]
        max_rc_threshold = float(mf_cfg.get("max_risk_contribution", 0.40))
        
        # Check if any single asset contributes too much risk as fraction of total
        if port_vol > 0:
            rc_fractions = risk_contrib / port_vol
            max_rc_fraction = rc_fractions.max()
            
            if max_rc_fraction > max_rc_threshold:
                return False, f"risk concentration: max asset RC {max_rc_fraction:.1%} > {max_rc_threshold:.1%} threshold"
    
    # 5. Diversification ratio
    min_div = float(mf_cfg.get("min_diversification_ratio", 1.2))
    if portfolio_stats["diversification_ratio"] < min_div:
        return False, f"low diversification: {portfolio_stats['diversification_ratio']:.2f} < {min_div:.2f}"
    
    # 6. Minimum holdings
    min_hold = int(mf_cfg.get("min_holdings", 3))
    if portfolio_stats["num_holdings"] < min_hold:
        return False, f"too few holdings: {portfolio_stats['num_holdings']} < {min_hold}"
    
    return True, None


# ============================================================================
# Composite Scoring
# ============================================================================

def compute_composite_score(
    sharpe: float,
    max_drawdown: float,
    lambda_penalty: float = 0.2,
    *,
    volatility: float | None = None,
    risk_profile: Any | None = None,
    soft_band: bool = False,
    soft_vol_penalty_lambda: float = 0.5,
) -> float:
    """Compute composite score for ranking portfolios with optional soft-band penalty.

    Base formula (audit Section 13.7):
        base_score = Sharpe - λ_drawdown * |MDD|

    Soft volatility band adjustment:
        If portfolio volatility is in the soft zone (between soft_lower and band_min_vol),
        apply penalty proportional to the gap to band_min_vol:

            penalty = soft_vol_penalty_lambda * (band_min_vol - volatility)
            score = base_score - penalty

        Where soft_lower = band_min_vol * vol_soft_lower_factor (factor provided via cfg; applied earlier).

    Args:
        sharpe: Portfolio Sharpe ratio.
        max_drawdown: Portfolio max drawdown (negative value).
        lambda_penalty: Drawdown penalty coefficient.
        volatility: Annualized portfolio volatility (required for soft band logic).
        risk_profile: RiskProfileResult providing band_min_vol.
        soft_band: True if portfolio volatility below band_min_vol but above soft_lower.
        soft_vol_penalty_lambda: Penalty coefficient for soft band gap (default 0.5).

    Returns:
        Composite score (higher is better).
    """
    base = sharpe - lambda_penalty * abs(max_drawdown)
    if soft_band and risk_profile is not None and isinstance(volatility, (int, float)) and volatility is not None:
        try:
            band_min = float(getattr(risk_profile, "band_min_vol", None) or getattr(risk_profile, "vol_min", None) or 0.0)
            if band_min > 0 and volatility < band_min:
                gap = band_min - volatility
                penalty = float(soft_vol_penalty_lambda) * gap
                return base - penalty
        except Exception:
            return base
    return base


def check_distinctness(
    weights_a: pd.Series,
    weights_b: pd.Series,
    threshold: float = 0.995,
) -> bool:
    """
    Check if two portfolios are distinct using cosine similarity.
    
    Formula (audit Section 13.7):
        similarity = (w_a · w_b) / (||w_a|| * ||w_b||)
        
    Portfolios are considered duplicates if similarity > threshold.
    
    Args:
        weights_a: First portfolio weights
        weights_b: Second portfolio weights
        threshold: Max similarity before considering duplicate (default 0.995)
    
    Returns:
        True if portfolios are distinct (similarity <= threshold)
        False if portfolios are too similar (similarity > threshold)
    """
    try:
        # Get all unique symbols from both portfolios
        all_symbols = weights_a.index.union(weights_b.index)
        
        # Align both weight series to same index, filling missing with 0
        wa = weights_a.reindex(all_symbols, fill_value=0.0).values
        wb = weights_b.reindex(all_symbols, fill_value=0.0).values
        
        # Cosine similarity
        dot_product = np.dot(wa, wb)
        norm_a = np.linalg.norm(wa)
        norm_b = np.linalg.norm(wb)
        
        if norm_a == 0 or norm_b == 0:
            return True
        
        similarity = dot_product / (norm_a * norm_b)
        
        # Distinct if similarity <= threshold
        # Too similar (duplicate) if similarity > threshold
        return bool(similarity <= threshold)
    
    except Exception:
        return True  # On error, assume distinct

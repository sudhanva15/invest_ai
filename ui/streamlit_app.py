import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from core.utils.env_tools import load_env_once
from core.data_ingestion import get_prices, get_prices_with_provenance

# Config helpers
def load_config(_path: str = "config.yaml") -> dict:
    import yaml, pathlib
    p = pathlib.Path(_path)
    if not p.exists():
        return {}
    with p.open("r") as f:
        return yaml.safe_load(f) or {}

def load_json(_path: str) -> dict:
    import json, pathlib
    p = pathlib.Path(_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())

load_env_once('.env')

# Page config
st.set_page_config(
    page_title="Invest AI - Portfolio Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session helpers
def reset_app_state():
    """Explicit session reset per UX spec.
    Only clears targeted analytical / portfolio keys and returns user to Dashboard.
    Questionnaire answers remain so user can re-save easily if desired.
    """
    keys_to_clear = [
        "risk_score", "risk_answers",
        "chosen_portfolio", "last_candidates", "candidate_curves",
        "run_simulation", "asset_pool_text", "risk_slider_value",
        "prices_loaded", "prov_loaded", "last_run_settings", "last_run_at",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    st.session_state["page"] = "Dashboard"
    st.rerun()

# Build version
try:
    from core.utils.version import get_build_version
    version = get_build_version().replace("+", ".")
    st.caption(f"Build: {version}")
except Exception:
    pass

# Load config
try:
    CFG = load_config(str(ROOT / "config/config.yaml"))
    CAT = load_json(str(ROOT / "config/assets_catalog.json"))
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

from core.preprocessing import compute_returns

# Presets
PRESETS = {
    "grow": ["SPY", "QQQ", "TLT"],
    "grow_income": ["SPY", "LQD", "TLT"],
    "income": ["LQD", "HYG", "MUB"],
    "tax_efficiency": ["SPY", "MUB"],
    "preserve": ["BND", "SHY", "TIP"],
}

def dedupe_keep_order(items):
    seen, out = set(), []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

# Small helper used by tests to ensure unique symbol indexing
def _ensure_unique_symbol_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by unique 'symbol' values.
    - Upper-cases symbols
    - Drops duplicate symbols keeping the first occurrence
    - Ensures string index type
    """
    if df is None or df.empty:
        return pd.DataFrame().set_index(pd.Index([], dtype=str))
    if "symbol" not in df.columns:
        return df
    _df = df.copy()
    try:
        _df["symbol"] = _df["symbol"].astype(str).str.upper()
        _df = _df.drop_duplicates(subset=["symbol"], keep="first").set_index("symbol")
        _df.index = _df.index.astype(str)
        return _df
    except Exception:
        return df

# Initialize session state - ONLY set defaults if keys don't exist
# DO NOT overwrite existing values (fixes state reset bug)
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

if "prices_loaded" not in st.session_state:
    st.session_state["prices_loaded"] = None

if "prov_loaded" not in st.session_state:
    st.session_state["prov_loaded"] = {}

# Sidebar
st.sidebar.title("Invest AI")

# Navigation
nav_pages = ["Dashboard", "Profile", "Portfolios", "Macro", "Diagnostics"]
nav_selection = st.sidebar.radio(
    "Navigation", 
    nav_pages,
    index=nav_pages.index(st.session_state["page"]) if st.session_state["page"] in nav_pages else 0,
    key="nav_radio"
)

if nav_selection != st.session_state["page"]:
    st.session_state["page"] = nav_selection
    st.rerun()

# Sidebar buttons
if st.sidebar.button("Run simulation", key="run_sim_btn"):
    st.session_state["run_simulation"] = True

if st.sidebar.button("Reset session", key="reset_btn"):
    reset_app_state()

st.title("📊 Invest AI - Portfolio Recommender")

current_page = st.session_state["page"]

# ====================== DASHBOARD ======================
if current_page == "Dashboard":
    st.header("Dashboard")
    
    # Hero summary
    st.markdown("""
    Welcome to **Invest AI** – a portfolio recommendation engine that helps you build 
    diversified ETF portfolios matched to your risk profile.
    
    **How it works:**
    1. Complete your risk profile (8 questions)
    2. Generate portfolio candidates based on your objectives
    3. Select a recommended portfolio within your risk band
    4. Review macro indicators and system diagnostics
    """)
    
    st.markdown("---")
    
    # Universe stats from snapshot
    st.subheader("Universe Summary")
    try:
        from core.universe_validate import load_valid_universe
        valid_symbols, records, metrics = load_valid_universe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valid ETFs", len(valid_symbols))
        with col2:
            if metrics:
                hist_dist = metrics.get("history_years_distribution")
                if hist_dist and hist_dist.get("median"):
                    st.metric("History (median)", f"{hist_dist['median']:.1f} yrs")
                else:
                    st.metric("Avg History", f"{metrics.get('avg_hist_years', 0):.1f} yrs")
        with col3:
            if metrics:
                st.metric("Avg Coverage", f"{metrics.get('avg_coverage_pct', 0):.0f}%")
        with col4:
            dropped = metrics.get("dropped_count", 0) if metrics else 0
            st.metric("Dropped", dropped)
    except Exception as e:
        st.warning(f"Universe stats unavailable: {e}")
    
    st.markdown("---")
    
    # Selected portfolio (if any) - now stored as name string per spec
    chosen_name = st.session_state.get("chosen_portfolio")
    if isinstance(chosen_name, str):
        st.subheader("✨ Your Selected Portfolio")
        candidates = st.session_state.get("last_candidates", []) or []
        candidate_curves = st.session_state.get("candidate_curves", {}) or {}
        cand_obj = next((c for c in candidates if c.get("name") == chosen_name), None)
        weights = cand_obj.get("weights", {}) if isinstance(cand_obj, dict) else {}
        curve = candidate_curves.get(chosen_name)
        st.markdown(f"**{chosen_name}**")
        if weights:
            w_df = pd.Series(weights).sort_values(ascending=False).to_frame("Weight (%)")
            w_df["Weight (%)"] = (w_df["Weight (%)"] * 100).round(2)
            st.dataframe(w_df, use_container_width=True)
        if curve is not None:
            st.markdown("**Equity Curve (Full History)**")
            try:
                curve_plot = curve.resample("W").last() if hasattr(curve, "resample") else curve
                st.line_chart(curve_plot)
            except Exception:
                st.caption("Chart unavailable")
    else:
        st.info("No portfolio selected yet. Go to **Portfolios** to generate recommendations.")
    
    st.markdown("---")
    
    # CTA button
    if st.button("Go to Profile →", key="go_to_profile_from_dashboard"):
        st.session_state["page"] = "Profile"
        st.rerun()
    
    st.stop()

# PROFILE
elif current_page == "Profile":
    st.header("Your risk profile")
    
    try:
        from core.risk_profile import (
            compute_risk_score,
            map_time_horizon_choice,
            map_loss_tolerance_choice,
            map_reaction_20_drop_choice,
            map_income_stability_choice,
            map_dependence_on_money_choice,
            map_investing_experience_choice,
            map_safety_net_choice,
            map_goal_type_choice,
        )
    except Exception:
        compute_risk_score = None
        map_time_horizon_choice = map_loss_tolerance_choice = map_reaction_20_drop_choice = None
        map_income_stability_choice = map_dependence_on_money_choice = None
        map_investing_experience_choice = map_safety_net_choice = map_goal_type_choice = None
        st.warning("Risk profiling unavailable")

    saved_risk_score = st.session_state.get("risk_score")
    if saved_risk_score is not None:
        st.success(f"✓ Saved risk score: **{saved_risk_score:.1f}/100**")
    
    st.markdown("---")
    
    # Initialize question defaults ONLY if missing (do not overwrite user answers)
    defaults = {
        "risk_q1_time_horizon": "7–15 years",
        "risk_q2_loss_tolerance": "Medium",
        "risk_q3_reaction20": "Hold and wait",
        "risk_q4_income_stability": "Moderate",
        "risk_q5_dependence": "Important but not critical",
        "risk_q6_experience": "Some experience (< 3 years)",
        "risk_q7_safety_net": "Moderate safety net (3-6 months)",
        "risk_q8_goal_type": "Balanced growth (moderate risk)",
    }
    for _k, _v in defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    q1 = st.radio(
        "**Q1.** How long until you'll need most of this money?",
        ["0–3 years", "3–7 years", "7–15 years", "15+ years"],
        key="risk_q1_time_horizon",
        horizontal=True
    )
    
    q2 = st.radio(
        "**Q2.** How comfortable are you with temporary losses?",
        ["Very low", "Low", "Medium", "High", "Very high"],
        key="risk_q2_loss_tolerance",
        horizontal=True
    )
    
    q3 = st.radio(
        "**Q3.** If your portfolio dropped 20% soon after investing?",
        [
            "Sell everything immediately",
            "Sell some to reduce risk",
            "Hold and wait",
            "Hold and might buy more",
            "Definitely buy more (opportunity)",
        ],
        key="risk_q3_reaction20"
    )
    
    q4 = st.radio(
        "**Q4.** How stable is your income?",
        ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
        key="risk_q4_income_stability",
        horizontal=True
    )
    
    q5 = st.radio(
        "**Q5.** How dependent are you on this money?",
        [
            "Critical for living expenses",
            "Important but not critical",
            "Helpful but have other income",
            "Nice-to-have growth money",
        ],
        key="risk_q5_dependence"
    )
    
    q6 = st.radio(
        "**Q6.** Investing experience?",
        [
            "Beginner (first time)",
            "Some experience (< 3 years)",
            "Experienced (3-10 years)",
            "Advanced (10+ years)",
        ],
        key="risk_q6_experience"
    )
    
    q7 = st.radio(
        "**Q7.** Emergency fund and insurance?",
        [
            "No emergency fund or insurance",
            "Small emergency fund (< 3 months)",
            "Moderate safety net (3-6 months)",
            "Strong safety net (6+ months)",
        ],
        key="risk_q7_safety_net"
    )
    
    q8 = st.radio(
        "**Q8.** Main goal for this money?",
        [
            "Capital preservation (safety first)",
            "Income generation (steady returns)",
            "Balanced growth (moderate risk)",
            "Aggressive growth (max returns)",
        ],
        key="risk_q8_goal_type"
    )

    st.markdown("---")
    
    if st.button("💾 Save profile"):
        # Ensure all mapping functions are available
        mapping_funcs = [
            map_time_horizon_choice, map_loss_tolerance_choice, map_reaction_20_drop_choice,
            map_income_stability_choice, map_dependence_on_money_choice, map_investing_experience_choice,
            map_safety_net_choice, map_goal_type_choice
        ]
        if compute_risk_score is None or any(f is None for f in mapping_funcs):
            st.warning("Risk scoring unavailable")
        else:
            try:
                answers = {
                    "q1_time_horizon": map_time_horizon_choice(q1),
                    "q2_loss_tolerance": map_loss_tolerance_choice(q2),
                    "q3_reaction_20_drop": map_reaction_20_drop_choice(q3),
                    "q4_income_stability": map_income_stability_choice(q4),
                    "q5_dependence_on_money": map_dependence_on_money_choice(q5),
                    "q6_investing_experience": map_investing_experience_choice(q6),
                    "q7_safety_net": map_safety_net_choice(q7),
                    "q8_goal_type": map_goal_type_choice(q8),
                }
                rscore = compute_risk_score(answers)
            except Exception as e:
                st.error(f"Risk scoring error: {e}")
            else:
                st.session_state["risk_score"] = rscore
                st.session_state["risk_answers"] = answers
                st.success(f"✅ Profile saved. Risk score: **{rscore:.1f}/100**")

# PORTFOLIOS  
elif current_page == "Portfolios":
    st.header("Portfolio Recommendations")
    
    risk_score = st.session_state.get("risk_score")
    if risk_score is None:
        st.warning("⚠️ Please complete Profile first")
        if st.button("Go to Profile"):
            st.session_state["page"] = "Profile"
            st.rerun()
        st.stop()
    
    st.info(f"Using risk score: **{risk_score:.1f}/100**")
    
    st.markdown("### Portfolio Settings")
    
    colc1, colc2 = st.columns([1,1])
    with colc1:
        objective = st.selectbox(
            "Investment Objective",
            options=[
                ("Balanced", "grow"),
                ("Growth", "grow"),
                ("Growth + Income", "grow_income"),
                ("Income", "income"),
                ("Tax Efficiency", "tax_efficiency"),
                ("Preservation", "preserve")
            ],
            index=0,
            format_func=lambda x: x[0]
        )
    with colc2:
        n_candidates = st.slider(
            "Portfolios to Generate", 
            5, 10, 8, 1
        )
    
    st.markdown("### Asset Pool")

    # Full universe checkbox
    use_full_universe = st.checkbox(
        "Use full ETF universe (recommended for beginners)",
        value=st.session_state.get("use_full_universe", True),
        key="use_full_universe",
        help="Automatically use all validated ETFs from the universe snapshot. "
             "Uncheck to specify custom tickers or apply presets."
    )

    # Always render manual input controls, but disable when using full universe
    default_pool_text = ",".join(CFG.get("data",{}).get("default_universe", ["SPY","QQQ","TLT","BND"]))
    pool_val = st.session_state.get("asset_pool_text", default_pool_text)

    colp1, colp2 = st.columns([1,3])
    with colp1:
        if st.button("Apply preset", disabled=use_full_universe):
            syms = PRESETS.get(objective[1], [])
            pool_val = ",".join(dedupe_keep_order([s.strip().upper() for s in syms]))
            st.session_state["asset_pool_text"] = pool_val
            st.rerun()

    pool = st.text_input(
        "Ticker Symbols (comma-separated)", 
        value=pool_val, 
        key="asset_pool_text",
        disabled=use_full_universe,
        help="Enter tickers like SPY, QQQ, TLT. Disabled when using full universe."
    )

    if use_full_universe:
        # Load universe and show info
        try:
            from core.universe_validate import load_valid_universe
            valid_symbols, _, _ = load_valid_universe()
            st.info(f"✓ Using **{len(valid_symbols)} validated ETFs** from universe snapshot")
            symbols_to_use = valid_symbols
        except Exception as e:
            st.error(f"Failed to load universe: {e}")
            st.info("Falling back to default pool")
            default_pool = CFG.get("data",{}).get("default_universe", ["SPY","QQQ","TLT","BND"])
            symbols_to_use = default_pool
    else:
        symbols_to_use = [s.strip().upper() for s in pool.split(",") if s.strip()]
    
    st.markdown("---")
    
    # Check if we have previous results
    run_flag = st.session_state.get("run_simulation", False)
    last_candidates = st.session_state.get("last_candidates")
    
    if not run_flag and not last_candidates:
        st.info("👆 Click **Run simulation** in sidebar to generate portfolios")
        st.stop()
    
    # Handle both fresh simulation and display of previous results
    if run_flag:
        # RUN SIMULATION
        with st.spinner("Generating portfolios..."):
            symbols = symbols_to_use
            if len(symbols) < 3:
                st.error("❌ Need at least 3 symbols")
                st.session_state["run_simulation"] = False
                st.stop()

            if st.session_state.get("prices_loaded") is None:
                try:
                    _px, _prov = get_prices_with_provenance(symbols, start="1900-01-01")
                    st.session_state["prices_loaded"] = _px
                    st.session_state["prov_loaded"] = _prov
                except Exception as e:
                    st.error(f"Data load failed: {e}")
                    st.session_state["prices_loaded"] = pd.DataFrame()
                    st.session_state["prov_loaded"] = {}

            prices = get_prices(symbols, start="1900-01-01")
            if prices.empty:
                st.error("❌ No price data")
                st.session_state["run_simulation"] = False
                st.stop()
                
            rets = compute_returns(prices)

            from core.recommendation_engine import DEFAULT_OBJECTIVES, ObjectiveConfig, generate_candidates
            obj_key = objective[1]
            obj_cfg = DEFAULT_OBJECTIVES.get(obj_key)
            if obj_cfg is None:
                obj_cfg = ObjectiveConfig(name=obj_key)
            elif isinstance(obj_cfg, dict):
                obj_cfg = ObjectiveConfig(**obj_cfg)
            if not hasattr(obj_cfg, "bounds") or obj_cfg.bounds is None:
                obj_cfg.bounds = {"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07}

            try:
                cands = generate_candidates(
                    returns=rets,
                    objective_cfg=obj_cfg,
                    catalog=CAT,
                    n_candidates=n_candidates,
                )
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
                st.session_state["run_simulation"] = False
                st.stop()

            if not cands:
                st.warning("⚠️ No candidates generated")
                st.session_state["run_simulation"] = False
                st.stop()

            st.session_state["last_candidates"] = cands
            
            from core.utils.metrics import annualized_metrics, beta_vs_benchmark, var_95
            last_1y = rets.tail(252)
            try:
                spy_w = get_prices(["SPY"], start=str(last_1y.index.min().date()))
                spy_r = compute_returns(spy_w)["SPY"].reindex(last_1y.index).dropna()
            except Exception:
                spy_r = pd.Series(dtype=float)

            rows = []
            curves = {}
            for cand in cands:
                w = pd.Series(cand["weights"]).reindex(rets.columns).fillna(0.0)
                port_ret = (rets * w).sum(axis=1)
                port_1y = (last_1y * w).sum(axis=1)
                
                m = annualized_metrics(port_1y)
                beta = beta_vs_benchmark(port_1y, spy_r) if not spy_r.empty else float("nan")
                var95 = var_95(port_1y)
                
                curve = (1 + port_ret).cumprod().rename(cand["name"]).dropna()
                curves[cand["name"]] = curve
                
                n_holdings = sum(1 for v in cand["weights"].values() if v > 0.001)
                
                rows.append({
                    "Portfolio": cand["name"],
                    "Holdings": n_holdings,
                    "CAGR_1Y": round(m.get("CAGR", float("nan")) * 100, 2) if m.get("CAGR") is not None else float("nan"),
                    "Vol_1Y": round(m.get("Volatility", float("nan")) * 100, 2) if m.get("Volatility") is not None else float("nan"),
                    "Sharpe_1Y": round(m.get("Sharpe", float("nan")), 3) if m.get("Sharpe") is not None else float("nan"),
                    "Beta": round(beta, 3) if pd.notna(beta) else float("nan"),
                    "VaR95": round(var95, 4) if pd.notna(var95) else float("nan"),
                    "weights": cand["weights"],
                    "curve": curve,
                })
            
            st.session_state["candidate_curves"] = curves
            st.session_state["run_simulation"] = False

            # Store run metadata for traceability
            try:
                from datetime import datetime as _dt
                st.session_state["last_run_settings"] = {
                    "objective": obj_key,
                    "n_candidates": int(n_candidates),
                    "use_full_universe": bool(use_full_universe),
                    "asset_pool": symbols if not use_full_universe else "FULL_UNIVERSE",
                    "risk_score": float(risk_score),
                }
                st.session_state["last_run_at"] = _dt.now().isoformat(timespec="seconds")
            except Exception:
                pass
            
            st.success(f"✅ Generated {len(cands)} portfolios")
    else:
        # Display previous results
        last_run_at = st.session_state.get("last_run_at")
        last_run_settings = st.session_state.get("last_run_settings", {}) or {}
        if last_run_at:
            universe_desc = (
                "Full universe"
                if last_run_settings.get("use_full_universe")
                else f"Manual pool ({len(last_run_settings.get('asset_pool', []))} tickers)"
            )
            st.info(
                f"📊 Showing last simulation run from {last_run_at} — "
                f"Objective: {last_run_settings.get('objective', '?')}, "
                f"Universe: {universe_desc}. "
                f"Click **Run simulation** in sidebar to regenerate."
            )
        else:
            st.info("📊 Showing previous simulation results. Click **Run simulation** in sidebar to regenerate.")
        cands = last_candidates or []
        curves = st.session_state.get("candidate_curves", {})
        # Reconstruct rows from candidates for display
        rows = []
        for cand in cands:
            rows.append({
                "Portfolio": cand.get("name"),
                "Holdings": sum(1 for v in cand.get("weights", {}).values() if v > 0.001),
                "CAGR_1Y": round(cand.get("metrics", {}).get("CAGR", 0.0) * 100, 2),
                "Vol_1Y": round(cand.get("metrics", {}).get("Vol", cand.get("metrics", {}).get("Volatility", 0.0)) * 100, 2),
                "Sharpe_1Y": round(cand.get("metrics", {}).get("Sharpe", 0.0), 3),
                "Beta": round(cand.get("metrics", {}).get("Beta", float("nan")), 3),
                "VaR95": round(cand.get("metrics", {}).get("VaR95", float("nan")), 4),
                "weights": cand.get("weights", {}),
                "curve": curves.get(cand.get("name")),
            })
    
    df_cands = pd.DataFrame(rows)
    
    st.subheader("Portfolio Comparison")
    st.dataframe(
        df_cands[["Portfolio", "Holdings", "CAGR_1Y", "Vol_1Y", "Sharpe_1Y", "Beta", "VaR95"]],
        use_container_width=True,
        hide_index=True
    )
    
    st.markdown("### Top 4 Portfolios (by CAGR)")
    
    top4 = df_cands.nlargest(4, "CAGR_1Y")
    
    if len(top4) >= 4:
        row1_cols = st.columns(2)
        row2_cols = st.columns(2)
        
        for idx, (_, row) in enumerate(top4.iterrows()):
            if idx < 2:
                col = row1_cols[idx]
            else:
                col = row2_cols[idx - 2]
            
            with col:
                st.markdown(f"**{row['Portfolio']}**")
                st.caption(f"CAGR: {row['CAGR_1Y']:.1f}% | Sharpe: {row['Sharpe_1Y']:.2f}")
                try:
                    curve = row['curve']
                    if hasattr(curve, 'resample'):
                        curve_plot = curve.resample("W").last()
                    else:
                        curve_plot = curve
                    st.line_chart(curve_plot, height=200)
                except Exception:
                    st.caption("(Chart unavailable)")
    
    st.markdown("---")
    
    st.subheader("🎯 Risk Match")
    
    try:
        from core.recommendation_engine import select_candidates_for_risk_score, pick_portfolio_from_slider
    except Exception:
        select_candidates_for_risk_score = pick_portfolio_from_slider = None
    
    if select_candidates_for_risk_score and pick_portfolio_from_slider:
        # Slider first to compute TRUE_RISK used for filtering
        slider_val = st.slider(
            "Choose position within risk zone:",
            0.0, 1.0, 0.5, 0.01,
            key="risk_slider_value"
        )
        slider_score = slider_val * 100.0
        TRUE_RISK = 0.7 * risk_score + 0.3 * slider_score
        st.caption(f"TRUE_RISK: **{TRUE_RISK:.1f}/100** (questionnaire: {risk_score:.1f}, slider: {slider_score:.1f})")

        # Filter using TRUE_RISK instead of raw risk score
        from typing import Any, cast
        filtered = select_candidates_for_risk_score(cast(Any, cands), float(TRUE_RISK))

        if not filtered:
            # Fallback: pick closest portfolio by volatility based on TRUE_RISK
            st.warning("⚠️ No exact matches in your risk band. Showing closest portfolio:")
            sigma_min, sigma_max = 0.1271, 0.2202
            target_vol = sigma_min + (sigma_max - sigma_min) * (TRUE_RISK / 100.0)

            def _get_vol(c):
                return float(c.get("metrics", {}).get("Vol") or c.get("metrics", {}).get("Volatility", 0.0))

            closest = min(cands, key=lambda c: abs(_get_vol(c) - target_vol))
            filtered = [closest]

            st.caption(
                f"**TRUE_RISK:** {TRUE_RISK:.0f}/100 → **Target volatility:** {target_vol:.1%}\n\n"
                f"Selected portfolio volatility: {_get_vol(closest):.1%}"
            )
        else:
            st.success(f"Found **{len(filtered)}** matching portfolios")

        picked = pick_portfolio_from_slider(cast(Any, filtered), slider_val)

        if picked:
            st.markdown(f"### ✨ Recommended: **{picked.get('name')}**")

            pw = pd.Series(picked.get("weights", {})).sort_values(ascending=False)
            st.dataframe(
                pw.to_frame("Weight (%)").mul(100).round(2),
                use_container_width=True
            )

            # Store chosen portfolio name only (per spec)
            # Dashboard will derive details from last_candidates and candidate_curves
            st.session_state["chosen_portfolio"] = picked.get("name")

            # Show equity curve
            try:
                pcurve = curves.get(picked.get("name"))
                if pcurve is not None:
                    if hasattr(pcurve, 'resample'):
                        pcurve_plot = pcurve.resample("W").last()
                    else:
                        pcurve_plot = pcurve
                    st.line_chart(pcurve_plot)
            except Exception:
                pass

# ====================== MACRO ======================
elif current_page == "Macro":
    st.header("Macroeconomic Indicators")
    
    st.markdown("""
    These indicators help explain the economic environment but don't directly change your portfolio.
    They provide context for why certain asset mixes may perform better in different conditions.
    """)
    
    st.markdown("---")
    
    try:
        from core.data_sources.fred import load_series
        import datetime as _dt
        
        macro_data = {
            "CPIAUCSL": {
                "name": "CPI (Inflation)",
                "series": load_series("CPIAUCSL"),
                "explanation": (
                    "**Consumer Price Index (CPI)** measures inflation – the rate at which prices rise. "
                    "Higher inflation can hurt bonds and change how central banks set interest rates, "
                    "which affects stock valuations."
                )
            },
            "FEDFUNDS": {
                "name": "Fed Funds Rate",
                "series": load_series("FEDFUNDS"),
                "explanation": (
                    "This is the **short-term interest rate** set by the Federal Reserve. "
                    "When it's high, cash and short-term bonds pay more, making riskier assets less attractive. "
                    "When it's low, investors often seek higher returns in stocks."
                )
            },
            "DGS10": {
                "name": "10-Year Treasury Yield",
                "series": load_series("DGS10"),
                "explanation": (
                    "This **long-term interest rate** is a key benchmark. "
                    "It affects mortgage rates and is often used as the 'risk-free' rate in portfolio models. "
                    "Rising yields can pressure stock prices."
                )
            },
            "UNRATE": {
                "name": "Unemployment Rate",
                "series": load_series("UNRATE"),
                "explanation": (
                    "This shows how tight or weak the **job market** is. "
                    "Very low unemployment usually means strong growth (good for stocks) but also inflation risk. "
                    "Very high unemployment can signal recessions."
                )
            },
        }
        
        for series_id, info in macro_data.items():
            st.subheader(info["name"])
            
            s = info["series"]
            if s is not None and not s.dropna().empty:
                try:
                    val = float(s.dropna().iloc[-1])
                    last_date = s.dropna().index[-1]
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric("Latest Value", f"{val:.2f}")
                        st.caption(f"As of {last_date.strftime('%Y-%m-%d')}")
                    with col2:
                        st.line_chart(s.tail(365), height=200)
                    
                    st.markdown(info["explanation"])
                except Exception as e:
                    st.caption(f"Error displaying {info['name']}: {e}")
            else:
                st.caption("No data available")
            st.markdown("---")
    
    except Exception as e:
        st.error(f"Failed to load macro data: {e}")

# ====================== DIAGNOSTICS ======================
elif current_page == "Diagnostics":
    st.header("System Diagnostics")
    
    st.markdown("""
    This page shows the health of the data universe and provider coverage.
    The system uses a **snapshot-based** approach at runtime, so live provider issues 
    don't shrink the universe unexpectedly.
    """)
    
    st.markdown("---")
    
    st.subheader("Universe Summary")
    try:
        from core.universe_validate import load_valid_universe
        valid_symbols, records, metrics = load_valid_universe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Valid ETFs", len(valid_symbols))
        with col2:
            if metrics:
                hist_dist = metrics.get("history_years_distribution")
                if hist_dist and hist_dist.get("median"):
                    st.metric("History (median)", f"{hist_dist['median']:.1f} yrs")
                    with st.expander("History distribution"):
                        st.write(f"Min: {hist_dist.get('min', 0):.1f} years")
                        st.write(f"Median: {hist_dist.get('median', 0):.1f} years")
                        st.write(f"Max: {hist_dist.get('max', 0):.1f} years")
                else:
                    st.metric("Avg History", f"{metrics.get('avg_hist_years', 0):.1f} yrs")
        with col3:
            if metrics:
                st.metric("Avg Coverage", f"{metrics.get('avg_coverage_pct', 0):.0f}%")
        with col4:
            if metrics:
                dropped = metrics.get("dropped_count", 0)
                st.metric("Dropped", dropped)
        
        # Provider breakdown
        st.markdown("### Provider Breakdown")
        from collections import Counter
        
        # Handle different record types safely
        provider_counts = Counter()
        for sym in valid_symbols:
            rec = records.get(sym)
            if rec:
                # Check if it's a dict or object with provider attribute
                if isinstance(rec, dict):
                    prov = rec.get("provider")
                elif hasattr(rec, "provider"):
                    prov = rec.provider
                else:
                    prov = None
                
                if prov:
                    provider_counts[prov] += 1
        
        prov_df = pd.DataFrame({
            "Provider": ["Tiingo", "Stooq", "yfinance"],
            "Valid ETFs": [
                provider_counts.get("tiingo", 0),
                provider_counts.get("stooq", 0),
                provider_counts.get("yfinance", 0)
            ]
        })
        st.dataframe(prov_df, use_container_width=True, hide_index=True)
        
        st.caption(
            "✅ Using cached snapshot data. Live providers are best-effort and "
            "do not affect universe size at runtime."
        )
    
    except Exception as e:
        st.warning(f"Universe snapshot unavailable: {e}")
    
    st.markdown("---")
    
    st.subheader("Rolling Metrics & Robustness")
    with st.expander("See explanation", expanded=False):
        st.markdown("""
        **Rolling metrics** help evaluate how stable a portfolio strategy is over time.
        
        Examples include:
        - **Rolling 1Y CAGR**: Shows if returns are consistent or highly variable
        - **Rolling Sharpe Ratio**: Measures risk-adjusted performance over different periods
        - **Rolling Drawdowns**: Tracks the largest peak-to-trough declines
        
        These are used internally to evaluate robustness but are not yet visualized in this UI.
        Future versions will show interactive charts of these metrics.
        """)
    
    st.markdown("---")
    
    st.subheader("Provider Receipts")
    st.caption("Data sources used in the last simulation run")
    
    try:
        prov = st.session_state.get("prov_loaded", {}) or {}
        if prov:
            rows = []
            for sym, pinfo in prov.items():
                if isinstance(pinfo, dict):
                    rows.append({
                        "Ticker": sym,
                        "Provider": pinfo.get("provider", "unknown"),
                        "First Date": pinfo.get("first_date", ""),
                        "Last Date": pinfo.get("last_date", ""),
                        "History (years)": round(pinfo.get("hist_years", 0), 1) if pinfo.get("hist_years") else None,
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No receipts available from last simulation.")
        else:
            st.info("Run a simulation once on the **Portfolios** page to see provider receipts.")
    except Exception as e:
        st.error(f"Failed to load receipts: {e}")

else:
    st.error(f"Unknown page: {current_page}")
    if st.button("Go to Home"):
        st.session_state["page"] = "Home"
        st.rerun()

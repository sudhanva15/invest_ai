import sys
from pathlib import Path

# Add repo root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_ingestion import get_prices, get_prices_with_provenance

import pandas as pd
import streamlit as st
from core.utils.env_tools import load_env_once
from core.utils.version import get_build_version, get_build_diagnostics
from core.utils.metrics import annualized_metrics, beta_vs_benchmark, var_95
from core.utils.stats import anova_bootstrap
from core.risk.profile import risk_profile_to_constraints
from core.recommendation_engine import (
    DEFAULT_OBJECTIVES,
    ObjectiveConfig,
    generate_candidates,
)

# Robust imports for config/json utilities (fallbacks if not re-exported)
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
# --- Cached helpers (speed) ---
@st.cache_data(ttl=3600)
def _cached_prices(sym_list, start="1900-01-01"):
    try:
        return get_prices(sym_list, start=start)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def _cached_prices_with_prov(sym_list, start="1900-01-01"):
    try:
        return get_prices_with_provenance(sym_list, start=start)
    except Exception:
        return pd.DataFrame(), {}

@st.cache_data(ttl=3600)
def _cached_returns(prices_df: pd.DataFrame):
    try:
        return compute_returns(prices_df)
    except Exception:
        return pd.DataFrame()


# --- Page config (must be first Streamlit command)
st.set_page_config(
    page_title="Invest AI - Portfolio Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------- Session helpers ----------------------
def reset_app_state():
    keys_to_clear = [
        # Profile / risk
        "risk_score",
        "risk_answers",
        # Portfolios
        "chosen_portfolio",
        "last_candidates",
        "run_simulation",
        "prices_loaded",
        "prov_loaded",
        # UI
        "current_page",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

# Display build version
try:
    version = get_build_version()
    # Convert to dot notation: V3.0.1+25 → V3.0.1.25
    version_dot = version.replace("+", ".")
    st.caption(f"Build: {version_dot}")
    
    # Build & Data Diagnostics expander
    with st.expander("Build & Data Diagnostics", expanded=False):
        from datetime import datetime
        diag = get_build_diagnostics()
        st.write({
            "version": version,
            "version_dot": version_dot,
            "commit_hash": diag.get("commit_hash"),
            "git_describe": diag.get("git_describe"),
            "commits_since_v3.0.1": diag.get("commits_since_tag"),
            "tracked_dirty": diag.get("tracked_dirty"),
            "untracked_count": diag.get("untracked_count"),
            "current_time_iso8601": datetime.now().isoformat(),
        })
except Exception:
    pass

# --- Load config and catalog
try:
    CFG = load_config(str(ROOT / "config/config.yaml"))
    CAT = load_json(str(ROOT / "config/assets_catalog.json"))
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

# Import core functions
from core.investor_profiles import (
    classify_investor,
    safe_investment_budget,
    TIER_ORDER,
)
from core.recommendation_engine import recommend, UserProfile

# Minimum risk per objective (default)
MIN_RISK_OBJ = {
    "grow": 45,
    "grow_income": 40,
    "income": 30,
    "tax_efficiency": 35,
    "preserve": 20,
}
from core.preprocessing import compute_returns
from core.trust import (
    data_coverage,
    effective_n_assets,
    bootstrap_interval,
    credibility_score,
)
from core.recommendation_engine import anova_mean_returns
from core.simulation_runner import simulate_dca_calendar_series, compute_xirr

# Helper: simple lump+DCA projection (MVP)
def project_lump_and_dca(curve: pd.Series, lump: float = 0.0, monthly: float = 0.0) -> float:
    """
    Simple projection: lump sum at start + fixed monthly buys, return final value.
    This is a naive MVP; for accurate calendar-aware DCA use simulate_dca_calendar_series.
    """
    if curve.empty:
        return 0.0
    start_val = curve.iloc[0]
    end_val = curve.iloc[-1]
    # Lump shares
    shares = lump / start_val if start_val != 0 else 0.0
    # Approximate monthly buys (assuming linear growth for simplicity)
    n_months = len(curve) // 21  # rough monthly count
    avg_price = curve.mean()
    shares += (monthly * n_months) / avg_price if avg_price != 0 else 0.0
    return shares * end_val

# --- Presets and helpers
PRESETS = {
    "grow": ["SPY", "QQQ", "TLT"],
    "grow_income": ["SPY", "LQD", "TLT"],
    "income": ["LQD", "HYG", "MUB"],
    "tax_efficiency": ["SPY", "MUB"],
    "preserve": ["BND", "SHY", "TIP"],
}
SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Staples": "XLP",
}
EXTRAS = {
    "International (dev)": ["EFA"],
    "Emerging markets": ["EEM"],
    "Commodities": ["DBC", "GSG"],
    "Gold": ["GLD"],
    "REITs": ["VNQ"],
    "T-Bills (cash)": ["BIL"],
}

def dedupe_keep_order(items):
    """Dedup while preserving order."""
    seen, out = set(), []
    for s in items:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

def _ensure_unique_symbol_index(df_cat_in):
    """
    Ensure df_cat has unique symbol index. Drop duplicates by 'symbol' (keep first),
    then set index='symbol', converting to string. Attaches _dropped_dupes_count as attr.
    """
    before = len(df_cat_in)
    df_dedup = df_cat_in.drop_duplicates(subset="symbol", keep="first").copy()
    after = len(df_dedup)
    df_dedup.index = df_dedup["symbol"].astype(str)
    df_dedup.index.name = "symbol"
    df_dedup._dropped_dupes_count = before - after
    return df_dedup

# --- Initialize session_state (lazy loading - data loaded on demand)
if "prices_loaded" not in st.session_state:
    st.session_state["prices_loaded"] = None  # Loaded on first simulation run
    st.session_state["prov_loaded"] = {}

if False:
    # OLD SIDEBAR UI – DISABLED
    with st.sidebar:
        st.header("Your Profile")
        try:
            version = get_build_version()
            version_dot = version.replace("+", ".")
            st.caption(f"Build: {version_dot}")
        except Exception:
            pass
        st.write("(Old sidebar controls hidden)")

# ---------------------- Minimal Sidebar ----------------------
st.sidebar.title("Invest AI")
page = st.sidebar.radio("Go to", ["Home", "Profile", "Portfolios", "Macro", "Diagnostics"], key="nav_radio")

if st.sidebar.button("Run simulation"):
    st.session_state["run_simulation"] = True

if st.sidebar.button("Reset session"):
    reset_app_state()

st.title("📊 Invest AI - Portfolio Recommender")

# ---------------------- Routing ----------------------
current_page = st.session_state.get("current_page", page)
st.session_state["current_page"] = current_page

# ---------------------- HOME ----------------------
if current_page == "Home":
    st.title("Invest AI – Portfolio Recommender")
    st.markdown(
        """
        Welcome! This app has 4 main sections:

        1. **Profile** – answer a short questionnaire to estimate your risk score  
        2. **Portfolios** – see ETF portfolios that match your risk profile  
        3. **Macro** – explore macroeconomic trends (FRED data)  
        4. **Diagnostics** – see the ETF universe and data provider health  

        Start with **Profile**, then move to **Portfolios**.
        """
    )
    if st.button("Go to Profile →"):
        st.session_state["current_page"] = "Profile"
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
    st.stop()

# ---------------------- PROFILE ----------------------
if current_page == "Profile":
    st.header("Your risk profile")
    st.markdown("Answer a few questions to estimate your risk score. Save when done. Only this page asks about you.")
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
        def _id(x):
            return x
        map_time_horizon_choice = map_loss_tolerance_choice = map_reaction_20_drop_choice = _id
        map_income_stability_choice = map_dependence_on_money_choice = map_investing_experience_choice = _id
        map_safety_net_choice = map_goal_type_choice = _id

    q1 = st.radio(
        "Q1. How long until you'll need most of this money?",
        ["0–3 years", "3–7 years", "7–15 years", "15+ years"],
        index=2,
        key="risk_q1_time_horizon_choice",
        horizontal=True,
    )
    q2 = st.radio(
        "Q2. How comfortable are you with temporary losses?",
        ["Very low", "Low", "Medium", "High", "Very high"],
        index=2,
        key="risk_q2_loss_tolerance_choice",
        horizontal=True,
    )
    q3 = st.radio(
        "Q3. If your portfolio dropped 20% soon after investing, what would you do?",
        [
            "Sell everything immediately",
            "Sell some to reduce risk",
            "Hold and wait",
            "Hold and might buy more",
            "Definitely buy more (opportunity)",
        ],
        index=2,
        key="risk_q3_reaction20_choice",
    )
    q4 = st.radio(
        "Q4. How stable is your income over the next 3–5 years?",
        ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
        index=2,
        key="risk_q4_income_stability_choice",
        horizontal=True,
    )
    q5 = st.radio(
        "Q5. How dependent are you on this money for living expenses?",
        [
            "Critical for living expenses",
            "Important but not critical",
            "Helpful but have other income",
            "Nice-to-have growth money",
        ],
        index=2,
        key="risk_q5_dependence_choice",
    )
    q6 = st.radio(
        "Q6. How experienced are you with investing and markets?",
        [
            "Beginner (first time)",
            "Some experience (< 3 years)",
            "Experienced (3-10 years)",
            "Advanced (10+ years)",
        ],
        index=1,
        key="risk_q6_experience_choice",
    )
    q7 = st.radio(
        "Q7. Do you have an emergency fund and basic insurance?",
        [
            "No emergency fund or insurance",
            "Small emergency fund (< 3 months)",
            "Moderate safety net (3-6 months)",
            "Strong safety net (6+ months)",
        ],
        index=2,
        key="risk_q7_safety_net_choice",
    )
    q8 = st.radio(
        "Q8. What's the main goal for this money?",
        [
            "Capital preservation (safety first)",
            "Income generation (steady returns)",
            "Balanced growth (moderate risk)",
            "Aggressive growth (max returns)",
        ],
        index=2,
        key="risk_q8_goal_type_choice",
    )

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
    if st.button("Save profile"):
        if compute_risk_score is None:
            st.warning("Risk scoring not available in this build.")
        else:
            rscore = compute_risk_score(answers)
            st.session_state["risk_score"] = rscore
            st.session_state["risk_answers"] = answers
            st.success(f"Risk profile saved: {rscore:.1f}/100")

# ---------------------- PORTFOLIOS ----------------------
if current_page == "Portfolios":
    st.header("Portfolios")
    risk_score = st.session_state.get("risk_score")
    if risk_score is None:
        st.warning("No risk profile found. Please fill out the Profile page first.")
        st.stop()

    # Page-local controls for portfolio generation
    colc1, colc2, colc3 = st.columns([1,1,1])
    with colc1:
        objective = st.selectbox(
            "Objective",
            options=[("Grow capital","grow"),
                     ("Grow + Income","grow_income"),
                     ("Income","income"),
                     ("Tax efficiency","tax_efficiency"),
                     ("Capital preservation","preserve")],
            index=0,
            format_func=lambda x: x[0],
        )
    with colc2:
        n_candidates = st.slider("# Candidates", min_value=5, max_value=10, value=8, step=1)
    with colc3:
        use_regime_nudge = st.checkbox("Use regime nudge", value=True)

    # Optional pool presets
    st.markdown("### Asset Pool")
    preset_key = objective[1]
    default_pool = ",".join(CFG.get("data",{}).get("default_universe", ["SPY","QQQ","TLT"]))
    pool_val = st.session_state.get("asset_pool_text", default_pool)
    colp1, colp2, colp3 = st.columns([1,1,2])
    with colp1:
        apply_preset = st.button("Apply preset")
    with colp2:
        clear_pool = st.button("Clear pool")
    if apply_preset:
        syms = PRESETS.get(preset_key, [])
        pool_val = ",".join(dedupe_keep_order([s.strip().upper() for s in syms]))
    if clear_pool:
        pool_val = ""
    pool = st.text_input("Symbols (comma-separated)", value=pool_val, key="asset_pool_text")

    # Triggered recompute only; no hidden resets
    run_flag = st.session_state.get("run_simulation")
    if not run_flag:
        st.info("Use the sidebar 'Run simulation' button to compute candidates with current settings.")
        st.stop()

    # Data prep
    symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
    if len(symbols) < 3:
        st.error("Please provide at least 3 tickers in the Asset Pool.")
        st.stop()

    # Lazy load prices (first run)
    if st.session_state.get("prices_loaded") is None:
        with st.spinner("Loading market data (first run)..."):
            try:
                _px, _prov = get_prices_with_provenance(symbols, start="1900-01-01")
                st.session_state["prices_loaded"] = _px
                st.session_state["prov_loaded"] = _prov
            except Exception as e:
                st.error(f"Failed to load market data: {e}")
                st.session_state["prices_loaded"] = pd.DataFrame()
                st.session_state["prov_loaded"] = {}

    prices = get_prices(symbols, start="1900-01-01")
    if prices.empty:
        st.error("No price data returned (source/network). Try fewer/different symbols.")
        st.stop()
    rets = compute_returns(prices)

    # Objective config
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
        st.error(f"Candidates failed: {e}")
        cands = []

    if not cands:
        st.warning("No candidates generated for current pool/objective.")
        st.stop()

    # Summaries and curves
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
        port_1y = (last_1y * w).sum(axis=1)
        m = annualized_metrics(port_1y)
        beta = beta_vs_benchmark(port_1y, spy_r) if not spy_r.empty else float("nan")
        var95 = var_95(port_1y)
        curve = (1 + (rets * w).sum(axis=1)).cumprod().rename(cand["name"]).dropna()
        curves[cand["name"]] = curve
        rows.append({
            "Name": cand["name"],
            "Sharpe_1Y": round(m.get("Sharpe", float("nan")), 3) if m.get("Sharpe") is not None else float("nan"),
            "CAGR_1Y": round(m.get("CAGR", float("nan")), 3) if m.get("CAGR") is not None else float("nan"),
            "Vol_1Y": round(m.get("Volatility", float("nan")), 3) if m.get("Volatility") is not None else float("nan"),
            "Beta": round(beta, 3) if pd.notna(beta) else float("nan"),
            "VaR95": round(var95, 4) if pd.notna(var95) else float("nan"),
            "weights_dict": cand["weights"],
        })
    df_rows = pd.DataFrame(rows)
    st.subheader("Candidate portfolios")
    st.dataframe(df_rows[["Name","Sharpe_1Y","CAGR_1Y","Vol_1Y","Beta","VaR95"]], use_container_width=True)

    # Risk match (consumer of saved risk_score)
    try:
        from core.recommendation_engine import select_candidates_for_risk_score, pick_portfolio_from_slider
    except Exception:
        select_candidates_for_risk_score = pick_portfolio_from_slider = None

    if select_candidates_for_risk_score and pick_portfolio_from_slider:
        st.subheader("Risk match")
        st.caption(f"Using saved risk score: {risk_score:.1f}/100")
        filtered = select_candidates_for_risk_score(cands, float(risk_score))
        if not filtered:
            st.warning("No candidates matched your risk band. Add more assets or adjust objective.")
        else:
            slider_val = st.slider(
                "Where to sit within your band (safer ↔ more growth)?",
                0.0, 1.0,
                value=st.session_state.get("risk_slider_value", 0.5),
                step=0.01,
                key="risk_slider_value",
            )
            picked = pick_portfolio_from_slider(filtered, slider_val)
            if picked:
                st.success(f"Selected: {picked.get('name')}")
                pw = pd.Series(picked.get("weights", {})).sort_values(ascending=False)
                st.dataframe(pw.to_frame("weight"))
                try:
                    pcurve = curves.get(picked.get("name"))
                    st.line_chart(pcurve.resample("W").last() if hasattr(pcurve, "resample") else pcurve)
                except Exception:
                    pass

    # Clear the run flag after completing compute to avoid auto re-runs
    st.session_state["run_simulation"] = False

if False:
    # OLD tabs/UI – DISABLED
    tabs = st.tabs(["Portfolio", "Macro", "Debug"])

    # End Portfolios page

    # Old Candidates tab removed in favor of Portfolios page

# ==================== Macro Page ====================
if current_page == "Macro":
    st.subheader("Macro")
    try:
        from core.data_sources.fred import load_series
        import datetime as _dt
        macro_series = {
            "DGS10": load_series("DGS10"),
            "T10Y2Y": load_series("T10Y2Y"),
            "CPIAUCSL": load_series("CPIAUCSL"),
            "UNRATE": load_series("UNRATE"),
        }
        now = _dt.datetime.utcnow()
        stale_msgs = []
        cols = st.columns(len(macro_series))
        for (name, s), col in zip(macro_series.items(), cols):
            try:
                last_ts = s.dropna().index.max()
                val = float(s.dropna().iloc[-1]) if not s.dropna().empty else float("nan")
                age_days = (now - last_ts.to_pydatetime()).days if last_ts is not None else 9999
                col.metric(name, f"{val:.2f}", help=f"Last: {last_ts.date().isoformat() if last_ts is not None else 'n/a'}")
                limit = 60 if name in {"DGS10","T10Y2Y"} else 90
                if age_days > limit:
                    stale_msgs.append(f"{name} stale ({age_days}d>")
                col.line_chart(s.tail(365))
            except Exception:
                pass
        if stale_msgs:
            st.warning("; ".join(stale_msgs))
        # Keep this page independent of Portfolios controls/state
    except Exception as e:
        st.error(f"Macro load failed: {e}")

# ==================== Diagnostics Page ====================
if current_page == "Diagnostics":
    st.subheader("Diagnostics")
    # Universe snapshot and provider breakdown
    try:
        from core.universe_validate import load_valid_universe
        valid_symbols, records, metrics = load_valid_universe()
        st.write({"universe_size": len(valid_symbols), **(metrics or {})})
        from collections import Counter
        providers = Counter(records[sym].get("provider") for sym in valid_symbols if isinstance(records.get(sym), dict))
        st.subheader("Provider breakdown (valid ETFs)")
        st.write({
            "Tiingo": providers.get("tiingo", 0),
            "Stooq": providers.get("stooq", 0),
            "yfinance": providers.get("yfinance", 0),
        })
        st.caption("Using cached data only. Live providers are best-effort and do not affect the universe size at runtime.")
    except Exception as e:
        st.caption(f"(Universe snapshot unavailable: {e})")

    # Receipts from last run (optional)
    try:
        prov = st.session_state.get("prov_loaded", {}) or {}
        if prov:
            rows = []
            for sym, pinfo in prov.items():
                first = pinfo.get("first_date") if isinstance(pinfo, dict) else None
                last = pinfo.get("last_date") if isinstance(pinfo, dict) else None
                rows.append({
                    "ticker": sym,
                    "provider": pinfo.get("provider") if isinstance(pinfo, dict) else str(pinfo),
                    "first": first,
                    "last": last,
                    "hist_years": pinfo.get("hist_years", None) if isinstance(pinfo, dict) else None,
                })
            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption("No provider receipts yet — run a simulation once.")
    except Exception as e:
        st.error(f"Receipts build failed: {e}")

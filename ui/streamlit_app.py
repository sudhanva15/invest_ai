import os
import sys
from pathlib import Path

# Add repo root to path before importing project modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from core.utils.env_tools import load_env_once, is_demo_mode

st.set_page_config(
    page_title="Invest AI - Portfolio Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from core.objective_mapper import (
        load_objectives_config,
        classify_objective_fit,
    )
except Exception as _obj_err:  # pragma: no cover
    load_objectives_config = None
    classify_objective_fit = None
    st.warning(f"Objective mapper unavailable: {_obj_err}")

try:
    from core.data_ingestion import get_prices, get_prices_with_provenance
except Exception as _e:
    # Degraded mode: log import failure and provide minimal stubs so UI can render help text
    get_prices = get_prices_with_provenance = None
    st.warning(f"Data ingestion modules unavailable: {_e}")

try:
    # Phase 3 UI components (multi-factor engine views)
    from ui.components.portfolio_display import (
        display_selected_portfolio,
        display_receipts,
    )
except Exception:
    display_selected_portfolio = None
    display_receipts = None
    st.info("Portfolio display components not available; running in minimal mode.")

try:
    import importlib
    import core.risk_profile as _rp
    try:
        importlib.invalidate_caches()
        _rp = importlib.reload(_rp)
    except Exception:
        pass
    compute_risk_profile = getattr(_rp, "compute_risk_profile", None)
    RiskProfileResult = getattr(_rp, "RiskProfileResult", None)
    if compute_risk_profile is None or RiskProfileResult is None:
        st.warning("Risk profile module partially loaded; attempting a second reload…")
        try:
            _rp = importlib.reload(_rp)
            compute_risk_profile = getattr(_rp, "compute_risk_profile", None)
            RiskProfileResult = getattr(_rp, "RiskProfileResult", None)
        except Exception:
            pass
        if compute_risk_profile is None or RiskProfileResult is None:
            st.warning("Risk profile module still incomplete; engine features will degrade until reload completes.")
except Exception as _e:  # pragma: no cover
    st.error(f"Phase 3 engine unavailable (risk profile module): {_e}")
    compute_risk_profile = None
    RiskProfileResult = None

# --- Debug panel helper ---
def _risk_profile_debug_info():
    try:
        import core.risk_profile as _rpd
        return {
            "module_file": getattr(_rpd, "__file__", "?"),
            "has_compute_risk_profile": hasattr(_rpd, "compute_risk_profile"),
            "has_RiskProfileResult": hasattr(_rpd, "RiskProfileResult"),
            "exported_names_subset": [n for n in dir(_rpd) if n.startswith("compute_") or n.endswith("Result")][:12],
        }
    except Exception as e:
        return {"error": str(e)}

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

IS_PROD = os.getenv("INVEST_AI_ENV", "").lower() == "production"
DEMO_MODE = is_demo_mode()

st.markdown(
    """
    <style>
    .stApp {background-color:#080c18; color:#f5f5f5;}
    .stMarkdown, .stText, .stCaption, .stMetric {color:#f5f5f5 !important;}
    .stButton>button, .stRadio>div>label, .stCheckbox>label {
        background-color:#1c2333;
        color:#f5f5f5;
        border-radius:8px;
        border:1px solid #2f3648;
    }
    .css-1dp5vir, .stDataFrame, .stTable {background-color:#111827 !important; color:#f5f5f5 !important;}
    .stMetric {background-color:#111827; border-radius:10px; padding:0.75rem; border:1px solid #1f2433;}
    input, textarea, select,
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stSelectbox div[data-baseweb="select"] *,
    .stMultiselect div[data-baseweb="select"] *,
    .stSlider > div[data-baseweb="slider"] *,
    .stRadio label,
    .stDateInput input {
        color:#ffffff !important;
    }
    .stTextInput input, .stNumberInput input, .stTextArea textarea,
    .stDateInput input,
    .stSelectbox div[data-baseweb="select"],
    .stMultiselect div[data-baseweb="select"] {
        background-color:#1c2333 !important;
        border:1px solid #2f3648 !important;
    }
    .stTextInput label,
    .stNumberInput label,
    .stTextArea label,
    .stSelectbox label,
    .stMultiselect label,
    .stSlider label,
    .stDateInput label,
    .stRadio label,
    [data-testid="stWidgetLabel"] p {
        color:#f8fafc !important;
    }
    input:disabled, textarea:disabled, select:disabled,
    .stTextInput input:disabled, .stNumberInput input:disabled,
    .stTextArea textarea:disabled {
        color:#d1d5db !important;
        background-color:#2a3144 !important;
    }
    ::placeholder {color:#cccccc !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar debug expander for engine status (dev only)
if not IS_PROD:
    with st.sidebar.expander("⚙️ Engine Diagnostics", expanded=False):
        dbg = _risk_profile_debug_info()
        st.markdown("**Risk Profile Module**")
        for k,v in dbg.items():
            st.write(f"{k}: {v}")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reload risk module"):
                try:
                    import importlib, core.risk_profile as _rp_reload
                    importlib.invalidate_caches()
                    importlib.reload(_rp_reload)
                    st.success("Risk profile module reloaded. Rerunning app…")
                    st.rerun()
                except Exception as e:
                    st.error(f"Reload failed: {e}")
        with col_b:
            st.toggle("debug_mode", value=st.session_state.get("debug_mode", False), key="debug_mode", help="Show module paths and extra info")
        st.caption("If has_compute_risk_profile=False, try the reload button, restarting from repo root, or clearing __pycache__.")

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

OBJECTIVES_CONFIG = {}
OBJECTIVE_OPTIONS: list[str] = []
if load_objectives_config is not None:
    try:
        OBJECTIVES_CONFIG = load_objectives_config() or {}
        OBJECTIVE_OPTIONS = list(OBJECTIVES_CONFIG.keys())
    except Exception as _obj_load_err:  # pragma: no cover
        st.warning(f"Unable to load objectives configuration: {_obj_load_err}")

OBJECTIVE_PRESET_MAP = {
    "CONSERVATIVE": "preserve",
    "BALANCED": "grow_income",
    "GROWTH_PLUS_INCOME": "grow_income",
    "GROWTH": "grow",
    "AGGRESSIVE": "grow",
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
    st.session_state["page"] = "Landing"

# Keep legacy "current_page" in sync for older session states
st.session_state["current_page"] = st.session_state.get("page", "Landing")

if "beginner_mode" not in st.session_state:
    st.session_state["beginner_mode"] = True

if "prices_loaded" not in st.session_state:
    st.session_state["prices_loaded"] = None

if "prov_loaded" not in st.session_state:
    st.session_state["prov_loaded"] = {}

# Sidebar
st.sidebar.title("Invest AI")

# Navigation
PROD_NAV_PAGES = ["Landing", "Profile", "Portfolios", "Dashboard", "Macro"]
DEV_NAV_PAGES = [
    "Landing",
    "Dashboard",
    "Profile",
    "Portfolios",
    "Macro",
    "Diagnostics",
    "Settings",
]

nav_pages = PROD_NAV_PAGES if IS_PROD else DEV_NAV_PAGES

session_page = st.session_state.get("page", "Landing")
legacy_page = st.session_state.get("current_page")
if legacy_page and legacy_page != session_page:
    session_page = legacy_page

if session_page not in nav_pages:
    session_page = "Landing"

st.session_state["page"] = session_page
st.session_state["current_page"] = session_page

nav_selection = st.sidebar.radio(
    "Navigation",
    nav_pages,
    index=nav_pages.index(session_page),
    key="nav_radio",
)

if nav_selection != session_page:
    st.session_state["page"] = nav_selection
    st.session_state["current_page"] = nav_selection
    st.rerun()

# Beginner mode indicator
beginner_mode = st.session_state.get("beginner_mode", True)
if beginner_mode:
    st.sidebar.caption("🎓 Beginner mode: ON")
else:
    st.sidebar.caption("⚙️ Beginner mode: OFF")

# Sidebar buttons
if st.sidebar.button("Run simulation", key="run_sim_btn"):
    st.session_state["run_simulation"] = True

if st.sidebar.button("Reset session", key="reset_btn"):
    reset_app_state()

st.title("📊 Invest AI - Portfolio Recommender")

# Helper functions for income-based risk scoring (used by Profile page)
def compute_risk_score_facts(income_profile: dict) -> float:
    """Compute objective risk capacity from financial facts.
    
    Returns 0-100 score based on:
    - Income stability (0-25 pts)
    - Emergency fund coverage (0-25 pts)
    - Investable surplus (0-25 pts)
    - Debt burden (0-25 pts)
    """
    score = 0.0
    
    # Income stability (0-25)
    stability = income_profile.get("income_stability", "Moderate")
    stability_map = {
        "Very unstable": 5, "Unstable": 10, "Moderate": 15, "Stable": 20, "Very stable": 25
    }
    score += stability_map.get(stability, 15)
    
    # Emergency fund (0-25)
    efund = income_profile.get("emergency_fund_months", 3)
    if efund >= 6:
        score += 25
    elif efund >= 3:
        score += 15
    elif efund >= 1:
        score += 8
    else:
        score += 0
    
    # Investable surplus (0-25) - based on ratio to monthly expenses
    try:
        investable = float(income_profile.get("investable_amount", 0))
        monthly_exp = float(income_profile.get("monthly_expenses", 1))
        if monthly_exp > 0:
            surplus_ratio = investable / (monthly_exp * 12)  # as fraction of annual expenses
            if surplus_ratio >= 2.0:
                score += 25
            elif surplus_ratio >= 1.0:
                score += 20
            elif surplus_ratio >= 0.5:
                score += 12
            elif surplus_ratio >= 0.2:
                score += 8
            else:
                score += 3
        else:
            score += 10  # neutral if expenses unknown
    except Exception:
        score += 10
    
    # Debt burden (0-25) - inverse score
    try:
        annual_income = float(income_profile.get("annual_income", 0))
        debt = float(income_profile.get("outstanding_debt", 0))
        if annual_income > 0:
            debt_ratio = debt / annual_income
            if debt_ratio >= 3.0:
                score += 0  # very high debt
            elif debt_ratio >= 1.5:
                score += 8
            elif debt_ratio >= 0.5:
                score += 15
            elif debt_ratio >= 0.1:
                score += 20
            else:
                score += 25  # minimal debt
        else:
            score += 15  # neutral
    except Exception:
        score += 15
    
    return min(100, max(0, score))

def compute_risk_score_combined(risk_score_questionnaire: float, risk_score_facts: float) -> float:
    """Combine feelings (questionnaire) and facts (income) into unified score.
    
    50/50 weighting: both psychology and capacity matter equally.
    """
    return (risk_score_questionnaire * 0.5) + (risk_score_facts * 0.5)

def risk_label(score: float) -> str:
    """Convert numeric risk score to qualitative label."""
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


def _format_currency(value: float) -> str:
    """Format numbers as simple dollar strings."""
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "$0"


def _future_value(lump_sum: float, monthly_contrib: float, years: float, annual_rate: float) -> float:
    """Compute future value with monthly compounding for lump sum plus contributions."""
    months = max(0, int(round(years * 12)))
    if months == 0:
        return float(lump_sum)
    monthly_rate = annual_rate / 12.0
    if monthly_rate == 0:
        fv_lump = float(lump_sum)
        fv_contrib = monthly_contrib * months
    else:
        fv_lump = float(lump_sum) * ((1 + monthly_rate) ** months)
        fv_contrib = monthly_contrib * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    return fv_lump + fv_contrib


SCENARIO_RISK_TEXT = {
    "Conservative": "Low-volatility buffer: aims to limit drawdowns (<5% typical yearly drop).",
    "Base": "Moderate case: what we expect if markets follow long-term trends.",
    "Optimistic": "Higher variance: deeper drawdowns possible, best for higher-risk investors.",
}


def _extract_base_cagr(metrics: dict, fallback: float = 0.06) -> float:
    for key in ("cagr", "CAGR", "CAGR_1Y", "CAGR_5Y"):
        val = metrics.get(key)
        if val is None:
            continue
        if isinstance(val, (int, float)) and abs(val) > 1.5:
            val = val / 100.0
        return float(val)
    return fallback


def _lookup_asset_class(symbol: str, universe_records: dict, catalog_assets: dict) -> str:
    sym = str(symbol or "").upper()
    record = universe_records.get(sym)
    if record:
        try:
            ac = record.get("asset_class") if isinstance(record, dict) else getattr(record, "asset_class", None)
            if ac:
                return str(ac)
        except Exception:
            pass
    catalog_entry = catalog_assets.get(sym)
    if catalog_entry:
        if catalog_entry.get("asset_class"):
            return str(catalog_entry["asset_class"])
        cls = str(catalog_entry.get("class", ""))
        if cls.startswith("equity"):
            return "equity"
        if cls.startswith("bond") or cls in {"high_yield", "munis"}:
            return "bond"
        if cls in {"cash", "tbills"}:
            return "cash"
        if cls in {"commodities", "gold"}:
            return "commodity"
        if cls in {"reit"}:
            return "reit"
    return "Unknown"


def _quality_label(portfolio: dict) -> str:
    if not portfolio:
        return "Unknown"
    if portfolio.get("hard_fallback"):
        return "Emergency fallback portfolio"
    if portfolio.get("fallback"):
        if portfolio.get("fallback_level") == 2:
            return "Relaxed filters portfolio (still in your risk band)"
        return "Relaxed filters portfolio (soft guardrails)"
    return "Strict-quality portfolio"

current_page = st.session_state["page"]

# ====================== LANDING ======================
if current_page == "Landing":
    st.header("Welcome to Invest AI")

    if DEMO_MODE:
        st.info("Demo mode: using cached snapshots only — live market data is disabled in this deployment.")
    
    st.markdown("""
    ### Portfolio recommendations built for learning
    
    Invest AI is an **educational tool** that demonstrates how quantitative portfolio construction works. 
    It combines historical data, risk profiling, and algorithmic optimization to generate diversified 
    ETF portfolios matched to different risk profiles.
    
    #### How it works
    
    1. **Profile** → Answer questions about your financial situation and risk preferences
    2. **Generate** → The system creates portfolio candidates using historical data and optimization
    3. **Explore** → Compare portfolios, review allocations, see how they would have performed historically
    4. **Learn** → Understand macro indicators, data quality, and portfolio construction principles
    
    #### Important disclaimers
    
    ⚠️ **This is NOT financial advice.** Invest AI is an educational demonstration tool. It:
    
    - Uses historical data which may not predict future performance
    - Does not consider your complete financial situation, tax status, or personal circumstances
    - Cannot replace consultation with a qualified financial advisor
    - Is provided "as is" with no guarantees of accuracy or suitability
    
    **Use at your own risk.** By proceeding, you acknowledge that any investment decisions you make 
    are your sole responsibility.
    
    ---
    
    Ready to explore how portfolio construction works?
    """)
    
    if st.button("Start with Profile →", key="landing_to_profile"):
        st.session_state["page"] = "Profile"
        st.session_state["current_page"] = "Profile"
        st.rerun()
    
    st.stop()

# ====================== DASHBOARD ======================
elif current_page == "Dashboard":
    st.header("Portfolio Dashboard")

    if DEMO_MODE:
        st.caption("Demo mode is active: all analytics use historical snapshots. Real markets move faster than these illustrations.")

    plan_summary = st.session_state.get("plan_summary")
    if plan_summary is None:
        plan_summary = {
            "lump_sum": float(st.session_state.get("ip_investable_amount", 10000) or 10000),
            "monthly": float(st.session_state.get("plan_monthly_contrib", 500.0) or 500.0),
            "years": int(st.session_state.get("plan_years", 10) or 10),
            "total": 0.0,
        }
        plan_summary["total"] = plan_summary["lump_sum"] + plan_summary["monthly"] * 12 * plan_summary["years"]

    chosen_name = st.session_state.get("chosen_portfolio")
    candidates = st.session_state.get("last_candidates") or st.session_state.get("mf_recommended") or []
    candidate_curves = st.session_state.get("candidate_curves", {}) or {}

    if not chosen_name or not candidates:
        st.info("No portfolio selected yet. Visit the **Portfolios** page to generate and choose one.")
        st.stop()

    selected = next((c for c in candidates if c.get("name") == chosen_name), None)
    if selected is None:
        st.info("Your previously selected portfolio isn't available. Re-run recommendations on the Portfolios page.")
        st.stop()

    metrics = selected.get("metrics", {}) or {}
    base_cagr = _extract_base_cagr(metrics)
    cagr_band = f"{max(0.0, base_cagr - 0.01):.1%} – {max(0.0, base_cagr + 0.01):.1%}"
    risk_score_val = st.session_state.get("risk_score_combined") or st.session_state.get("risk_score")
    risk_badge = risk_label(risk_score_val) if risk_score_val is not None else "Not rated"

    header_cols = st.columns([2, 1])
    with header_cols[0]:
        st.markdown(f"### {chosen_name}")
        st.caption(_quality_label(selected))
        st.metric("Expected CAGR band", cagr_band)
    with header_cols[1]:
        st.metric("Risk label", risk_badge)
        if risk_score_val is not None:
            st.metric("True risk score", f"{risk_score_val:.0f}/100")
        else:
            st.metric("True risk score", "n/a")

    weights_series = pd.Series(selected.get("weights", {})).sort_values(ascending=False)
    universe_records = st.session_state.get("universe_records") or {}
    catalog_assets = {
        str(a.get("symbol", "")).upper(): a for a in (CAT or {}).get("assets", [])
    }

    asset_class_weights = {}
    asset_class_counts = {}
    for sym, wt in weights_series.items():
        ac = _lookup_asset_class(sym, universe_records, catalog_assets)
        asset_class_weights[ac] = asset_class_weights.get(ac, 0.0) + float(wt)
        asset_class_counts[ac] = asset_class_counts.get(ac, 0) + 1

    st.markdown("### Allocation & Composition")
    alloc_cols = st.columns(2)

    with alloc_cols[0]:
        st.caption("Asset allocation pie chart")
        if not weights_series.empty:
            pie_df = weights_series.reset_index()
            pie_df.columns = ["symbol", "weight"]
            pie_df["weight_pct"] = pie_df["weight"] * 100
            pie_df["asset_class"] = pie_df["symbol"].apply(
                lambda s: _lookup_asset_class(s, universe_records, catalog_assets)
            )
            pie_chart = (
                alt.Chart(pie_df)
                .mark_arc(innerRadius=45, stroke="#0d0d12", strokeWidth=1)
                .encode(
                    theta=alt.Theta(field="weight", type="quantitative"),
                    color=alt.Color("symbol:N", title="Ticker", scale=alt.Scale(scheme="tableau10")),
                    tooltip=[
                        alt.Tooltip("symbol:N", title="Ticker"),
                        alt.Tooltip("asset_class:N", title="Asset class"),
                        alt.Tooltip("weight_pct:Q", title="Weight (%)", format=",.1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("Weights unavailable. Re-run the engine to refresh.")

    with alloc_cols[1]:
        st.caption("Asset class breakdown (combined allocation)")
        class_df = pd.DataFrame(
            {"asset_class": list(asset_class_weights.keys()), "allocation": [v * 100 for v in asset_class_weights.values()]}
        )
        if not class_df.empty:
            class_chart = (
                alt.Chart(class_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X("allocation:Q", title="Allocation (%)", axis=alt.Axis(format=",.0f")),
                    y=alt.Y("asset_class:N", sort="-x", title="Asset class"),
                    color=alt.Color("asset_class:N", legend=None, scale=alt.Scale(scheme="dark2")),
                    tooltip=[
                        alt.Tooltip("asset_class:N", title="Class"),
                        alt.Tooltip("allocation:Q", title="Weight (%)", format=",.1f"),
                    ],
                )
                .properties(height=260)
            )
            st.altair_chart(class_chart, use_container_width=True)
        else:
            st.info("No asset class data available yet.")

    if asset_class_counts:
        st.caption("Category overlap (simplified Venn view)")
        st.markdown(
            """
            <style>
            .venn-chip {
                background: #1f1f28;
                border-radius: 999px;
                padding: 0.75rem 1.25rem;
                text-align: center;
                border: 1px solid #2f2f3f;
                margin-bottom: 0.5rem;
            }
            .venn-chip span {display:block; font-size:0.8rem; color:#c5c5d1;}
            .venn-chip strong {font-size:1.4rem; color:#ffffff;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        venn_cols = st.columns(min(4, len(asset_class_counts)))
        sorted_counts = sorted(asset_class_counts.items(), key=lambda x: -x[1])[:4]
        for col, (cls, count) in zip(venn_cols, sorted_counts):
            col.markdown(
                f"<div class='venn-chip'><span>{cls.title()}</span><strong>{count}</strong> holdings</div>",
                unsafe_allow_html=True,
            )

    st.markdown("### Performance & Projections")
    perf_cols = st.columns(2)

    def _build_history_chart():
        series_bundle = []
        curve = candidate_curves.get(chosen_name)
        if curve is not None and not getattr(curve, "empty", True):
            curve = curve.dropna()
            if not curve.empty:
                curve.index = pd.to_datetime(curve.index)
                start_cut = curve.index.max() - pd.DateOffset(years=20)
                trimmed = curve[curve.index >= start_cut]
                if not trimmed.empty:
                    normalized = (trimmed / trimmed.iloc[0]) * 100
                    series_bundle.append(("Your portfolio", normalized))
        benchmark_prices = None
        if get_prices is not None:
            try:
                benchmark_prices = get_prices(["SPY", "BIL"], start="2004-01-01")
            except Exception:
                benchmark_prices = None
        if benchmark_prices is not None and not benchmark_prices.empty:
            for sym, label in [("SPY", "SPY benchmark"), ("BIL", "Cash proxy (BIL)")]:
                if sym in benchmark_prices.columns:
                    series = benchmark_prices[sym].dropna()
                    if not series.empty:
                        series.index = pd.to_datetime(series.index)
                        start_cut = series.index.max() - pd.DateOffset(years=20)
                        trimmed = series[series.index >= start_cut]
                        if not trimmed.empty:
                            normalized = (trimmed / trimmed.iloc[0]) * 100
                            series_bundle.append((label, normalized))
        if not series_bundle:
            return None
        combined = pd.concat([s.rename(label) for label, s in series_bundle], axis=1).dropna(how="all")
        combined = combined.reset_index(names="Date").melt("Date", var_name="Series", value_name="Growth of $100")
        chart = (
            alt.Chart(combined)
            .mark_line(point=False)
            .encode(
                x=alt.X("Date:T", title="Date"),
                y=alt.Y("Growth of $100:Q", title="Value", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Series:N", title=""),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Growth of $100:Q", title="Value", format="$,.0f"),
                ],
            )
            .properties(height=320)
        )
        return chart

    with perf_cols[0]:
        st.caption("20-year historical replay (growth of $100)")
        history_chart = _build_history_chart()
        if history_chart is not None:
            st.altair_chart(history_chart, use_container_width=True)
        else:
            st.info("Not enough historical data to draw this chart.")

    with perf_cols[1]:
        st.caption("Forward projection scenarios")
        scenario_points = []
        horizon_years = max(1, int(plan_summary.get("years", 10)))
        scenario_configs = [
            ("Conservative", max(0.01, base_cagr - 0.02)),
            ("Base", base_cagr),
            ("Optimistic", base_cagr + 0.02),
        ]
        for label, rate in scenario_configs:
            for year in range(0, horizon_years + 1):
                scenario_points.append(
                    {
                        "Year": year,
                        "Value": _future_value(
                            plan_summary["lump_sum"], plan_summary["monthly"], year, rate
                        ),
                        "Scenario": label,
                    }
                )
        scenario_df = pd.DataFrame(scenario_points)
        scenario_chart = (
            alt.Chart(scenario_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("Year:Q", title="Years from today"),
                y=alt.Y("Value:Q", title="Projected value", axis=alt.Axis(format="$,.0f")),
                color=alt.Color("Scenario:N", title="Scenario"),
                tooltip=[
                    alt.Tooltip("Year:Q", title="Year"),
                    alt.Tooltip("Scenario:N", title="Scenario"),
                    alt.Tooltip("Value:Q", title="Value", format="$,.0f"),
                ],
            )
            .properties(height=320)
        )
        st.altair_chart(scenario_chart, use_container_width=True)
        st.caption(
            "Conservative/Base/Optimistic follow the same assumptions shown on the Portfolios page, "
            "now drawn as full growth paths."
        )

    st.markdown("### What this means for you")
    dd_val = metrics.get("max_drawdown") or metrics.get("MaxDD")
    if isinstance(dd_val, (int, float)) and abs(dd_val) > 1.5:
        dd_val = dd_val / 100.0
    if not isinstance(dd_val, (int, float)):
        dd_val = 0.15
    bad_year_pct = abs(dd_val)
    vol_disp = metrics.get("volatility") or metrics.get("Vol")
    if isinstance(vol_disp, (int, float)) and vol_disp > 1.5:
        vol_disp = vol_disp / 100.0
    if not isinstance(vol_disp, (int, float)):
        vol_disp = 0.12
    good_year_gain = max(base_cagr, 0.07)
    base_projection = _future_value(
        plan_summary["lump_sum"], plan_summary["monthly"], plan_summary["years"], base_cagr
    )
    plan_lump_fmt = _format_currency(plan_summary["lump_sum"])
    plan_monthly_fmt = _format_currency(plan_summary["monthly"])
    plan_total_fmt = _format_currency(plan_summary["total"])
    summary_lines = [f"- **Risk level:** {risk_badge}"]
    if risk_score_val is not None:
        summary_lines[-1] += f" ({risk_score_val:.0f}/100)"
    summary_lines.append(f"- **Return band:** {cagr_band}")
    summary_lines.append(
        f"- **Your contribution plan:** {plan_lump_fmt} now + {plan_monthly_fmt}/month for {int(plan_summary['years'])} years (total {plan_total_fmt})"
    )
    st.markdown("\n".join(summary_lines))
    st.write(
        f"With **{chosen_name}**, keeping that plan for {int(plan_summary['years'])} years puts the base case near "
        f"{_format_currency(base_projection)}."
    )
    st.write(
        f"Typical swings are about {vol_disp:.1%} per year. "
        f"A rough 'bad year' could mean around {bad_year_pct:.1%} drawdown, "
        f"while a 'good year' might add {good_year_gain:.1%} or more."
    )
    st.markdown(
        f"- **Good year example:** +{good_year_gain:.1%} → adds about {_format_currency(plan_summary['total'] * good_year_gain)} in gains.\n"
        f"- **Tough year example:** −{bad_year_pct:.1%} → could temporarily reduce value by {_format_currency(plan_summary['total'] * bad_year_pct)}."
    )

    st.markdown("### Compare with another portfolio")
    other_names = [c.get("name") for c in candidates if c.get("name") and c.get("name") != chosen_name]
    compare_choice = None
    if other_names:
        compare_choice = st.selectbox(
            "Compare with another portfolio",
            options=["None"] + other_names,
            index=0,
        )
    else:
        st.caption("Generate more portfolios to unlock comparisons.")

    if compare_choice and compare_choice != "None":
        other = next((c for c in candidates if c.get("name") == compare_choice), None)
        if other:
            comp_cols = st.columns(2)
            for col, (label, port) in zip(comp_cols, [("Selected", selected), ("Comparison", other)]):
                pm = port.get("metrics", {}) or {}
                col.subheader(f"{label}: {port.get('name')}")
                this_cagr = _extract_base_cagr(pm)
                this_vol = pm.get("volatility") or pm.get("Vol")
                if isinstance(this_vol, (int, float)) and this_vol > 1.5:
                    this_vol = this_vol / 100.0
                this_dd = pm.get("max_drawdown") or pm.get("MaxDD")
                if isinstance(this_dd, (int, float)) and abs(this_dd) > 1.5:
                    this_dd = this_dd / 100.0
                col.metric("CAGR", f"{this_cagr:.2%}")
                col.metric("Volatility", f"{this_vol:.2%}" if isinstance(this_vol, (int, float)) else "n/a")
                col.metric("Max drawdown", f"{this_dd:.2%}" if isinstance(this_dd, (int, float)) else "n/a")
                col.caption(_quality_label(port))

            sel_weights = selected.get("weights", {})
            other_weights = other.get("weights", {})
            symbols = sorted(set(sel_weights.keys()) | set(other_weights.keys()))
            diff_df = pd.DataFrame({"Symbol": symbols})
            diff_df[chosen_name] = diff_df["Symbol"].apply(lambda s: round(sel_weights.get(s, 0.0) * 100, 1))
            diff_df[compare_choice] = diff_df["Symbol"].apply(lambda s: round(other_weights.get(s, 0.0) * 100, 1))
            diff_df["Diff (pp)"] = (diff_df[chosen_name] - diff_df[compare_choice]).round(1)
            st.dataframe(diff_df.set_index("Symbol"), use_container_width=True)

    if not IS_PROD:
        st.markdown("### Advanced")
        show_debug = st.toggle("Show diagnostics", value=False)
        if show_debug:
            st.write("**Raw weights**")
            st.dataframe(pd.Series(selected.get("weights", {})).to_frame("Weight").mul(100).round(2))
            rets_cached = st.session_state.get("rets_cached")
            if isinstance(rets_cached, pd.DataFrame):
                st.write("**Recent return snapshot**")
                st.dataframe(rets_cached.tail().style.format("{:.4f}"))
            st.write("**Phase 3 stats**", st.session_state.get("phase3_stats"))
            if display_receipts is not None:
                display_receipts(
                    asset_receipts=st.session_state.get("mf_asset_receipts"),
                    portfolio_receipts=st.session_state.get("mf_portfolio_receipts"),
                    beginner_mode=st.session_state.get("beginner_mode", True),
                )

    st.stop()

# ====================== PROFILE ======================
elif current_page == "Profile":
    st.header("Your Risk Profile")
    
    # Import questionnaire mapping functions
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
        st.warning("Risk profiling module unavailable")
    
    st.markdown("""
    Building a portfolio starts with understanding **who you are financially** and **how you feel about risk**.
    Both matter. Complete both panels below to get your comprehensive risk profile.
    """)
    
    # Display saved scores if they exist
    saved_combined = st.session_state.get("risk_score_combined")
    saved_questionnaire = st.session_state.get("risk_score_questionnaire")
    saved_facts = st.session_state.get("risk_score_facts")
    
    if saved_combined is not None:
        st.success(f"✓ **Combined Risk Score: {saved_combined:.1f}/100** ({risk_label(saved_combined)})")
        col1, col2 = st.columns(2)
        with col1:
            if saved_questionnaire is not None:
                st.caption(f"Feelings: {saved_questionnaire:.1f} ({risk_label(saved_questionnaire)})")
        with col2:
            if saved_facts is not None:
                st.caption(f"Facts: {saved_facts:.1f} ({risk_label(saved_facts)})")
    
    st.markdown("---")
    
    # Two-panel layout
    col_facts, col_feelings = st.columns(2)
    
    # ========== LEFT PANEL: INCOME & BALANCE SHEET (FACTS) ==========
    with col_facts:
        st.subheader("📊 Income & Balance Sheet")
        st.caption("Objective financial capacity")
        
        # Initialize income profile defaults
        income_defaults = {
            "annual_income": 75000,
            "income_stability": "Moderate",
            "monthly_expenses": 3000,
            "outstanding_debt": 0,
            "investable_amount": 10000,
            "emergency_fund_months": 3,
        }
        if "income_profile" not in st.session_state:
            st.session_state["income_profile"] = income_defaults.copy()
        
        ip = st.session_state["income_profile"]
        
        st.number_input(
            "Annual income ($)",
            min_value=0,
            value=int(ip.get("annual_income", 75000)),
            step=5000,
            key="ip_annual_income",
            help="Total annual income before taxes"
        )
        
        st.selectbox(
            "Income stability",
            ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
            index=["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"].index(
                ip.get("income_stability", "Moderate")
            ),
            key="ip_income_stability",
            help="How predictable and secure is your income stream?"
        )
        
        st.number_input(
            "Monthly expenses ($)",
            min_value=0,
            value=int(ip.get("monthly_expenses", 3000)),
            step=500,
            key="ip_monthly_expenses",
            help="Average monthly living expenses"
        )
        
        st.number_input(
            "Outstanding debt ($)",
            min_value=0,
            value=int(ip.get("outstanding_debt", 0)),
            step=5000,
            key="ip_outstanding_debt",
            help="Total debt (credit cards, loans, mortgage balance, etc.)"
        )
        
        st.number_input(
            "Investable amount ($)",
            min_value=0,
            value=int(ip.get("investable_amount", 10000)),
            step=1000,
            key="ip_investable_amount",
            help="Amount you're planning to invest now"
        )
        
        st.number_input(
            "Emergency fund (months)",
            min_value=0.0,
            max_value=24.0,
            value=float(ip.get("emergency_fund_months", 3)),
            step=0.5,
            key="ip_emergency_fund_months",
            help="How many months of expenses can you cover with savings?"
        )
        
        if st.session_state.get("beginner_mode", True):
            with st.expander("ℹ️ Why these questions?"):
                st.markdown("""
                **Financial facts matter** because they measure your actual capacity to take risk:
                
                - **Stable income + emergency fund** → more room for portfolio volatility
                - **High debt or unstable income** → suggests prioritizing safety and liquidity
                - **Large investable surplus** → can afford longer time horizons
                
                This is objective: the math of your balance sheet determines how much risk you can handle.
                """)
    
    # ========== RIGHT PANEL: RISK QUESTIONNAIRE (FEELINGS) ==========
    with col_feelings:
        st.subheader("🧠 Risk Questionnaire")
        st.caption("Subjective risk tolerance & preferences")
        
        # Initialize questionnaire defaults
        q_defaults = {
            "risk_q1_time_horizon": "7–15 years",
            "risk_q2_loss_tolerance": "Medium",
            "risk_q3_reaction20": "Hold and wait",
            "risk_q4_income_stability": "Moderate",
            "risk_q5_dependence": "Important but not critical",
            "risk_q6_experience": "Some experience (< 3 years)",
            "risk_q7_safety_net": "Moderate safety net (3-6 months)",
            "risk_q8_goal_type": "Balanced growth (moderate risk)",
        }
        for _k, _v in q_defaults.items():
            if _k not in st.session_state:
                st.session_state[_k] = _v
        
        q1 = st.selectbox(
            "**Q1.** Time horizon?",
            ["0–3 years", "3–7 years", "7–15 years", "15+ years"],
            index=["0–3 years", "3–7 years", "7–15 years", "15+ years"].index(
                st.session_state.get("risk_q1_time_horizon", "7–15 years")
            ),
            key="risk_q1_time_horizon"
        )
        
        q2 = st.selectbox(
            "**Q2.** Loss tolerance?",
            ["Very low", "Low", "Medium", "High", "Very high"],
            index=["Very low", "Low", "Medium", "High", "Very high"].index(
                st.session_state.get("risk_q2_loss_tolerance", "Medium")
            ),
            key="risk_q2_loss_tolerance"
        )
        
        q3 = st.selectbox(
            "**Q3.** If portfolio drops 20%?",
            [
                "Sell everything immediately",
                "Sell some to reduce risk",
                "Hold and wait",
                "Hold and might buy more",
                "Definitely buy more (opportunity)",
            ],
            index=[
                "Sell everything immediately",
                "Sell some to reduce risk",
                "Hold and wait",
                "Hold and might buy more",
                "Definitely buy more (opportunity)",
            ].index(st.session_state.get("risk_q3_reaction20", "Hold and wait")),
            key="risk_q3_reaction20"
        )
        
        q4 = st.selectbox(
            "**Q4.** Income stability?",
            ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
            index=["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"].index(
                st.session_state.get("risk_q4_income_stability", "Moderate")
            ),
            key="risk_q4_income_stability"
        )
        
        q5 = st.selectbox(
            "**Q5.** Dependence on this money?",
            [
                "Critical for living expenses",
                "Important but not critical",
                "Helpful but have other income",
                "Nice-to-have growth money",
            ],
            index=[
                "Critical for living expenses",
                "Important but not critical",
                "Helpful but have other income",
                "Nice-to-have growth money",
            ].index(st.session_state.get("risk_q5_dependence", "Important but not critical")),
            key="risk_q5_dependence"
        )
        
        q6 = st.selectbox(
            "**Q6.** Investing experience?",
            [
                "Beginner (first time)",
                "Some experience (< 3 years)",
                "Experienced (3-10 years)",
                "Advanced (10+ years)",
            ],
            index=[
                "Beginner (first time)",
                "Some experience (< 3 years)",
                "Experienced (3-10 years)",
                "Advanced (10+ years)",
            ].index(st.session_state.get("risk_q6_experience", "Some experience (< 3 years)")),
            key="risk_q6_experience"
        )
        
        q7 = st.selectbox(
            "**Q7.** Emergency fund?",
            [
                "No emergency fund or insurance",
                "Small emergency fund (< 3 months)",
                "Moderate safety net (3-6 months)",
                "Strong safety net (6+ months)",
            ],
            index=[
                "No emergency fund or insurance",
                "Small emergency fund (< 3 months)",
                "Moderate safety net (3-6 months)",
                "Strong safety net (6+ months)",
            ].index(st.session_state.get("risk_q7_safety_net", "Moderate safety net (3-6 months)")),
            key="risk_q7_safety_net"
        )
        
        q8 = st.selectbox(
            "**Q8.** Main goal?",
            [
                "Capital preservation (safety first)",
                "Income generation (steady returns)",
                "Balanced growth (moderate risk)",
                "Aggressive growth (max returns)",
            ],
            index=[
                "Capital preservation (safety first)",
                "Income generation (steady returns)",
                "Balanced growth (moderate risk)",
                "Aggressive growth (max returns)",
            ].index(st.session_state.get("risk_q8_goal_type", "Balanced growth (moderate risk)")),
            key="risk_q8_goal_type"
        )
        
        if st.session_state.get("beginner_mode", True):
            with st.expander("ℹ️ Why these questions?"):
                st.markdown("""
                **Your feelings about risk matter** because portfolios only work if you can stick with them:
                
                - **Emotional comfort** with volatility is as important as capacity
                - **Experience** affects how you'll react to market swings
                - **Goals and time horizon** shape what "success" means for you
                
                This is subjective: how you *feel* about risk determines whether you'll stay the course.
                """)
    
    st.markdown("---")
    
    # Save profile button
    if st.button("💾 Save Complete Profile", key="save_profile_btn"):
        # Update income profile from inputs
        st.session_state["income_profile"] = {
            "annual_income": st.session_state.get("ip_annual_income", 75000),
            "income_stability": st.session_state.get("ip_income_stability", "Moderate"),
            "monthly_expenses": st.session_state.get("ip_monthly_expenses", 3000),
            "outstanding_debt": st.session_state.get("ip_outstanding_debt", 0),
            "investable_amount": st.session_state.get("ip_investable_amount", 10000),
            "emergency_fund_months": st.session_state.get("ip_emergency_fund_months", 3),
        }
        
        # Compute facts score
        score_facts = compute_risk_score_facts(st.session_state["income_profile"])
        st.session_state["risk_score_facts"] = score_facts
        
        # Compute questionnaire score
        if compute_risk_score is not None:
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
                score_questionnaire = compute_risk_score(answers)
                st.session_state["risk_score_questionnaire"] = score_questionnaire
                st.session_state["risk_answers"] = answers
                
                # Compute combined score
                score_combined = compute_risk_score_combined(score_questionnaire, score_facts)
                st.session_state["risk_score_combined"] = score_combined
                
                # Also store in risk_score for backward compatibility with Portfolios page
                st.session_state["risk_score"] = score_combined
                
                st.success(f"✅ Profile saved!")
                st.info(f"""
                **Risk Scores:**
                - Facts (financial capacity): {score_facts:.1f}/100 ({risk_label(score_facts)})
                - Feelings (questionnaire): {score_questionnaire:.1f}/100 ({risk_label(score_questionnaire)})
                - **Combined**: {score_combined:.1f}/100 ({risk_label(score_combined)})
                """)
                
                st.markdown("Go to **Portfolios** to generate recommendations matched to your profile.")
                
            except Exception as e:
                st.error(f"Error computing questionnaire score: {e}")
        else:
            st.warning("Questionnaire scoring unavailable. Only facts score computed.")
            st.session_state["risk_score_facts"] = score_facts
            st.session_state["risk_score"] = score_facts

# PORTFOLIOS  
elif current_page == "Portfolios":
    st.header("Portfolio Recommendations")
    
    risk_score = st.session_state.get("risk_score")
    if risk_score is None:
        st.warning("⚠️ Please complete Profile first")
        if st.button("Go to Profile", key="portfolios_go_profile"):
            st.session_state["page"] = "Profile"
            st.session_state["current_page"] = "Profile"
            st.rerun()
        st.stop()
    
    st.info(f"Using risk score: **{risk_score:.1f}/100**")
    
    st.markdown("### Portfolio Settings")

    def _ensure_risk_profile_cached():
        rp = st.session_state.get("risk_profile")
        answers = st.session_state.get("risk_answers") or {}
        income_profile = st.session_state.get("income_profile") or {}
        slider_seed = float(st.session_state.get("risk_slider_value_pct", risk_score))
        if rp is None and compute_risk_profile is not None and answers and income_profile:
            try:
                rp = compute_risk_profile(
                    questionnaire_answers=answers,
                    income_profile=income_profile,
                    slider_score=slider_seed,
                )
                st.session_state["risk_profile"] = rp
            except Exception:
                rp = None
        return rp

    def _render_objective_fit_banner(risk_profile_obj, objective_name):
        if (
            not use_phase3
            or classify_objective_fit is None
            or risk_profile_obj is None
            or not objective_name
            or not OBJECTIVES_CONFIG
        ):
            return
        fit_type, fit_explanation, fit_suggested = classify_objective_fit(
            getattr(risk_profile_obj, "true_risk", 0.0),
            objective_name,
            OBJECTIVES_CONFIG,
        )
        suggested_label = None
        if fit_suggested:
            suggested_cfg = OBJECTIVES_CONFIG.get(fit_suggested)
            suggested_label = suggested_cfg.label if suggested_cfg else fit_suggested.title()

        message = fit_explanation
        if fit_type == "match":
            st.success(f"✅ Objective fit: {message}")
        elif fit_type == "stretch":
            st.warning(f"⚠️ Objective stretch: {message}")
        else:
            if suggested_label:
                message = f"{message} Consider **{suggested_label}** instead."
            st.error(f"❗ Objective mismatch: {message}")

    active_risk_profile = _ensure_risk_profile_cached()

    # Phase 3 toggle: Use new multi-factor engine
    use_phase3 = st.checkbox(
        "Use multi-factor engine (Phase 3)",
        value=st.session_state.get("use_phase3_engine", True),
        help="Enable the new asset-first multi-factor filtering and recommendation engine",
        key="use_phase3_engine"
    )
    
    colc1, colc2 = st.columns([1,1])
    with colc1:
        selected_objective_cfg = None

        if OBJECTIVE_OPTIONS:
            def _objective_label(name: str) -> str:
                cfg = OBJECTIVES_CONFIG.get(name)
                return cfg.label if cfg else name.title()
            default_name = st.session_state.get("selected_objective_name")
            if default_name not in OBJECTIVE_OPTIONS:
                default_name = "BALANCED" if "BALANCED" in OBJECTIVE_OPTIONS else OBJECTIVE_OPTIONS[0]
            selected_objective_name = st.selectbox(
                "Investment Objective",
                options=OBJECTIVE_OPTIONS,
                index=OBJECTIVE_OPTIONS.index(default_name),
                format_func=_objective_label,
            )
            selected_objective_cfg = OBJECTIVES_CONFIG.get(selected_objective_name)
            st.session_state["selected_objective_name"] = selected_objective_name
            if selected_objective_cfg is not None:
                st.session_state["selected_objective"] = selected_objective_cfg
            objective_label = _objective_label(selected_objective_name)
            legacy_slug = OBJECTIVE_PRESET_MAP.get(selected_objective_name, selected_objective_name.lower())
            objective = (objective_label, legacy_slug)
        else:
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
            st.session_state.pop("selected_objective", None)
            st.session_state.pop("selected_objective_name", None)
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
    
    st.markdown("### Your Investment Plan")
    default_lump = float(st.session_state.get("plan_lump_sum", st.session_state.get("ip_investable_amount", 10000) or 10000))
    default_monthly = float(st.session_state.get("plan_monthly_contrib", 500.0))
    default_years = int(st.session_state.get("plan_years", 10))
    col_plan1, col_plan2, col_plan3 = st.columns(3)
    with col_plan1:
        plan_lump_sum = col_plan1.number_input(
            "Lump sum today ($)",
            min_value=0.0,
            value=default_lump,
            step=1000.0,
            key="plan_lump_sum",
            help="Money you plan to invest right now"
        )
    with col_plan2:
        plan_monthly = col_plan2.number_input(
            "Monthly contribution ($)",
            min_value=0.0,
            value=default_monthly,
            step=100.0,
            key="plan_monthly_contrib",
            help="Amount you expect to add each month"
        )
    with col_plan3:
        plan_years = col_plan3.slider(
            "Years invested",
            min_value=1,
            max_value=40,
            value=min(max(default_years, 1), 40),
            key="plan_years",
            help="How long you plan to let this money grow"
        )
    plan_total = plan_lump_sum + (plan_monthly * 12 * plan_years)
    plan_summary = {
        "lump_sum": plan_lump_sum,
        "monthly": plan_monthly,
        "years": plan_years,
        "total": plan_total,
    }
    st.session_state["plan_summary"] = plan_summary
    st.caption(
        f"You are planning to invest {_format_currency(plan_lump_sum)} right away plus "
        f"{_format_currency(plan_monthly)} each month for {plan_years} years. "
        f"Total invested = {_format_currency(plan_total)}."
    )

    st.markdown("---")
    
    # Check if we have previous results
    run_flag = st.session_state.get("run_simulation", False)
    
    # DEBUG: Phase 3 session state inspection
    DEBUG_PHASE3 = st.session_state.get("debug_mode", False)
    if DEBUG_PHASE3 and not IS_PROD:
        with st.expander("🔍 DEBUG: Phase 3 Session State (BEFORE simulation)", expanded=False):
            st.write("**Session keys:**", sorted([k for k in st.session_state.keys() if not k.startswith("FormSubmitter")]))
            st.write("**last_candidates:**", st.session_state.get("last_candidates"))
            st.write("**mf_recommended:**", st.session_state.get("mf_recommended"))
            st.write("**run_flag:**", run_flag)
    
    last_candidates = st.session_state.get("last_candidates")
    
    if not run_flag and not last_candidates:
        st.info("👆 Click **Run simulation** in sidebar to generate portfolios")
        st.stop()
    
    # Handle both fresh simulation and display of previous results
    if run_flag:
        selected_objective_cfg_state = st.session_state.get("selected_objective")
        selected_objective_name_state = st.session_state.get("selected_objective_name")
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
            # Initialize rows and cands to ensure they're always available for df_cands creation
            rows = []
            cands = []
            if use_phase3:
                # Phase 3: Multi-factor engine (use pre-imported compute_risk_profile)
                # Robust import pattern for recommendation engine to avoid circular partial import
                try:
                    import importlib
                    import core.recommendation_engine as _re_mod
                    # Force a reload to avoid partial import state from earlier errors/hot-reload
                    try:
                        _re_mod = importlib.reload(_re_mod)
                    except Exception:
                        pass
                    build_recommendations = getattr(_re_mod, "build_recommendations", None)
                    ObjectiveConfig = getattr(_re_mod, "ObjectiveConfig", None)
                    # Diagnostics: show module path if debugging is enabled
                    if st.session_state.get("debug_mode"):
                        st.info(f"rec_engine: {_re_mod.__file__}")
                except Exception as e:
                    st.error(f"Phase 3 engine components unavailable (module load): {e}")
                    st.session_state["run_simulation"] = False
                    st.stop()
                if build_recommendations is None or ObjectiveConfig is None:
                    st.error("Phase 3 engine components unavailable: build_recommendations or ObjectiveConfig missing (partial import).")
                    st.session_state["run_simulation"] = False
                    st.stop()
                if compute_risk_profile is None:
                    st.error("Risk profile engine not loaded; see diagnostics sidebar.")
                    st.session_state["run_simulation"] = False
                    st.stop()

                # Risk fine-tune slider
                slider_val = st.slider(
                    "Fine-tune risk (slider)",
                    0.0, 100.0, float(st.session_state.get("risk_slider_value_pct", 50.0)), 1.0,
                    help="Adjust within your band: lower = more conservative, higher = more aggressive",
                    key="risk_slider_value_pct"
                )

                # Build risk profile from saved answers & income
                answers = st.session_state.get("risk_answers") or {}
                income_profile = st.session_state.get("income_profile") or {}
                objective_name = (objective[0] or "Balanced").lower()
                risk_profile = compute_risk_profile(
                    questionnaire_answers=answers,
                    income_profile=income_profile,
                    slider_score=float(slider_val),
                )
                st.session_state["risk_profile"] = risk_profile  # persist canonical profile

                # Objective config
                obj_cfg = ObjectiveConfig(
                    name=objective_name,
                    universe_filter=None,
                    bounds={"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07},
                    optimizer="hrp",
                )

                # DEBUG: Log Phase 3 parameters
                if st.session_state.get("debug_mode"):
                    st.info(f"🔍 Phase 3 Debug:\n"
                           f"- Universe size: {len(symbols)}\n"
                           f"- Risk band: vol_min={risk_profile.vol_min:.2%}, "
                           f"vol_target={risk_profile.vol_target:.2%}, vol_max={risk_profile.vol_max:.2%}\n"
                           f"- True risk: {risk_profile.true_risk:.1f}/100")

                # Build recommendations
                result = build_recommendations(
                    returns=rets,
                    catalog=CAT,
                    cfg=CFG,
                    risk_profile=risk_profile,
                    objective_cfg=obj_cfg,
                    n_candidates=int(n_candidates),
                    objective_config=selected_objective_cfg_state,
                )

                recommended = result.get("recommended", [])
                asset_receipts = result.get("asset_receipts")
                portfolio_receipts = result.get("portfolio_receipts")
                all_candidates = result.get("all_candidates", [])
                phase3_stats = result.get("stats") or {}
                
                # DEBUG: Log generation results
                if st.session_state.get("debug_mode"):
                    st.info(f"🔍 Generation results:\n"
                           f"- Total candidates: {len(all_candidates)}\n"
                           f"- Passed filters: {sum(1 for c in all_candidates if c.get('passed_filters'))}\n"
                           f"- Recommended: {len(recommended)}")

                # Check for fallback usage and show appropriate message
                has_fallback = any(c.get("fallback", False) for c in recommended)
                has_hard_fallback = any(c.get("hard_fallback", False) for c in recommended)
                
                if has_hard_fallback:
                    if IS_PROD:
                        st.warning("We couldn't find enough high-quality data for your filters, so you're seeing a conservative backup portfolio.")
                    else:
                        st.warning("⚠️ **Hard Fallback Active**: No suitable portfolios could be generated. "
                                  "Showing a safe equal-weight portfolio. Consider adjusting your asset pool or risk settings.")
                elif has_fallback:
                    if IS_PROD:
                        st.info("Some recommendations use broader filters so you still have options. Adjust your settings for stricter matches.")
                    else:
                        st.info("ℹ️ **Relaxed Filters**: Some portfolios shown use relaxed quality thresholds to ensure recommendations are available.")

                # Compute curves for display and dashboard compatibility
                curves = {}
                for cand in recommended:
                    w = pd.Series(cand.get("weights", {})).reindex(rets.columns).fillna(0.0)
                    port_ret = (rets * w).sum(axis=1)
                    curve = (1 + port_ret).cumprod().rename(cand.get("name")).dropna()
                    curves[cand.get("name")] = curve

                # Persist in session
                st.session_state["mf_recommended"] = recommended
                st.session_state["mf_asset_receipts"] = asset_receipts
                st.session_state["mf_portfolio_receipts"] = portfolio_receipts
                st.session_state["candidate_curves"] = curves  # for charts
                st.session_state["last_candidates"] = recommended  # unify with dashboard
                # Cache returns for later display (e.g., when viewing details without rerun)
                try:
                    st.session_state["rets_cached"] = rets
                except Exception:
                    pass
                st.session_state["phase3_stats"] = phase3_stats
                st.session_state["phase3_objective_debug"] = selected_objective_cfg_state
                st.session_state["run_simulation"] = False

                # Store metadata
                try:
                    from datetime import datetime as _dt
                    st.session_state["last_run_settings"] = {
                        "objective": objective_name,
                        "n_candidates": int(n_candidates),
                        "use_full_universe": bool(use_full_universe),
                        "asset_pool": symbols if not use_full_universe else "FULL_UNIVERSE",
                        "risk_score": float(risk_score),
                        "engine": "phase3",
                    }
                    st.session_state["last_run_at"] = _dt.now().isoformat(timespec="seconds")
                except Exception:
                    pass

                st.success(f"✅ Generated {len(recommended)} recommended portfolios (Phase 3)")
                if phase3_stats and not IS_PROD:
                    st.caption(
                        "Phase 3 engine: "
                        f"{phase3_stats.get('total_candidates', 0)} candidates → "
                        f"{phase3_stats.get('strict_passes', 0)} strict passes → "
                        f"{phase3_stats.get('recommended', 0)} recommended "
                        f"(fallback={phase3_stats.get('fallback_count', 0)}, "
                        f"hard={phase3_stats.get('hard_fallback_count', 0)})"
                    )
                elif phase3_stats:
                    st.caption("Portfolios generated successfully.")
                elif not IS_PROD:
                    st.caption("Phase 3 engine stats unavailable (first run?)")
                
                # DEBUG: Verify session state was updated
                if DEBUG_PHASE3 and not IS_PROD:
                    with st.expander("🔍 DEBUG: Phase 3 Session State (AFTER simulation)", expanded=False):
                        st.write("**last_candidates count:**", len(st.session_state.get("last_candidates", [])))
                        st.write("**mf_recommended count:**", len(st.session_state.get("mf_recommended", [])))
                        st.write("**candidate_curves keys:**", list(st.session_state.get("candidate_curves", {}).keys()))
            else:
                # Legacy pipeline
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
                        "engine": "legacy",
                    }
                    st.session_state["last_run_at"] = _dt.now().isoformat(timespec="seconds")
                except Exception:
                    pass
                
                st.success(f"✅ Generated {len(cands)} portfolios")

    # CRITICAL FIX: Re-read last_candidates from session state AFTER simulation
    # This ensures we have the freshly generated portfolios for display sections below
    last_candidates = st.session_state.get("last_candidates")
    
    if not run_flag:
        # Display previous results (only show this message if NOT just ran simulation)
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
    
    # Now set cands from the (possibly refreshed) last_candidates
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

    if use_phase3:
        _render_objective_fit_banner(active_risk_profile, st.session_state.get("selected_objective_name"))

    # Phase 3: Show recommendations table and selected portfolio view
    if use_phase3:
        if not IS_PROD:
            with st.expander("🛠️ Debug (Phase 3 Engine)", expanded=False):
                rp = active_risk_profile or st.session_state.get("risk_profile")
                stats_snapshot = st.session_state.get("phase3_stats") or {}
                obj_cfg_debug = (
                    st.session_state.get("phase3_objective_debug")
                    or st.session_state.get("selected_objective")
                )

                if rp is not None:
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("True risk", f"{getattr(rp, 'true_risk', 0):.1f}/100")
                        st.metric("Volatility band", f"{rp.vol_min:.2%} – {rp.vol_max:.2%}")
                    with col_r2:
                        st.metric("Target vol", f"{rp.vol_target:.2%}")
                        st.metric("Target CAGR", f"{rp.cagr_min:.2%} – {rp.cagr_target:.2%}")
                else:
                    st.warning("Risk profile not available yet. Save your profile or run a simulation.")

                if obj_cfg_debug is not None:
                    st.markdown(
                        f"**Objective:** {getattr(obj_cfg_debug, 'label', getattr(obj_cfg_debug, 'name', 'Unknown'))}"
                    )
                    st.caption(
                        f"Return band: {getattr(obj_cfg_debug, 'target_return_min', 0.0):.1%} – "
                        f"{getattr(obj_cfg_debug, 'target_return_max', 0.0):.1%} | "
                        f"Vol band: {getattr(obj_cfg_debug, 'target_vol_min', 0.0):.1%} – "
                        f"{getattr(obj_cfg_debug, 'target_vol_max', 0.0):.1%}"
                    )
                else:
                    st.info("Objective metadata unavailable; select an objective to view target bands.")

                if stats_snapshot:
                    cols = st.columns(3)
                    cols[0].metric("Total candidates", stats_snapshot.get("total_candidates", 0))
                    cols[1].metric("Strict passes", stats_snapshot.get("strict_passes", 0))
                    cols[2].metric("Recommended", stats_snapshot.get("recommended", 0))
                    st.caption(
                        f"Fallback portfolios: {stats_snapshot.get('fallback_count', 0)} | "
                        f"Hard fallback: {stats_snapshot.get('hard_fallback_count', 0)} | "
                        f"Stage used: {stats_snapshot.get('stage_used', 0) or 'n/a'}"
                    )
                else:
                    st.info("Run the engine to see candidate/fallback counts.")

        st.subheader("Recommended Portfolios (Phase 3)")
        recommended_list = st.session_state.get("mf_recommended", [])
        plan_for_cards = st.session_state.get("plan_summary") or {
            "lump_sum": plan_lump_sum,
            "monthly": plan_monthly,
            "years": plan_years,
            "total": plan_total,
        }
        if not recommended_list:
            st.info("No recommendations to display. Click Run simulation in the sidebar.")
        else:
            chosen_name = st.session_state.get("chosen_portfolio")
            names = [c.get("name") for c in recommended_list if c.get("name")]
            if names and chosen_name not in names:
                st.session_state["chosen_portfolio"] = names[0]

            for idx, cand in enumerate(recommended_list, start=1):
                name = cand.get("name", f"Portfolio {idx}")
                metrics = cand.get("metrics", {}) or {}
                base_cagr = _extract_base_cagr(metrics)
                conservative_rate = max(0.01, base_cagr - 0.02)
                optimistic_rate = base_cagr + 0.02
                scenarios = [
                    ("Conservative", conservative_rate, SCENARIO_RISK_TEXT["Conservative"]),
                    ("Base", base_cagr, SCENARIO_RISK_TEXT["Base"]),
                    ("Optimistic", optimistic_rate, SCENARIO_RISK_TEXT["Optimistic"]),
                ]
                weights = cand.get("weights", {}) or {}
                with st.container():
                    st.markdown(f"#### {idx}. {name}")
                    st.caption(_quality_label(cand))
                    col_metrics = st.columns(4)
                    cagr_val = metrics.get("cagr")
                    if cagr_val is None:
                        cagr_val = metrics.get("CAGR")
                    if isinstance(cagr_val, (int, float)) and abs(cagr_val) > 1.5:
                        cagr_val = cagr_val / 100.0
                    col_metrics[0].metric("CAGR", f"{cagr_val:.2%}" if isinstance(cagr_val, (int, float)) else "n/a")

                    vol_val = metrics.get("volatility") or metrics.get("Vol") or metrics.get("Volatility")
                    if isinstance(vol_val, (int, float)) and vol_val > 1.5:
                        vol_val = vol_val / 100.0
                    col_metrics[1].metric("Volatility", f"{vol_val:.2%}" if isinstance(vol_val, (int, float)) else "n/a")

                    sharpe_val = metrics.get("sharpe") or metrics.get("Sharpe")
                    col_metrics[2].metric("Sharpe", f"{sharpe_val:.2f}" if isinstance(sharpe_val, (int,float)) else "n/a")

                    dd_val = metrics.get("max_drawdown") or metrics.get("MaxDD")
                    if isinstance(dd_val, (int, float)) and abs(dd_val) > 1.5:
                        dd_val = dd_val / 100.0
                    col_metrics[3].metric("Max drawdown", f"{dd_val:.2%}" if isinstance(dd_val, (int,float)) else "n/a")

                    st.markdown(
                        f"**You are investing:**  \n"
                        f"- {_format_currency(plan_for_cards['lump_sum'])} lump sum today  \n"
                        f"- {_format_currency(plan_for_cards['monthly'])} each month  \n"
                        f"- {int(plan_for_cards['years'])} years in the market  \n"
                        f"- **Total invested:** {_format_currency(plan_for_cards['total'])}"
                    )

                    st.markdown("**Projection scenarios (monthly compounding):**")
                    lump_fmt = _format_currency(plan_for_cards["lump_sum"])
                    monthly_fmt = _format_currency(plan_for_cards["monthly"])
                    total_fmt = _format_currency(plan_for_cards["total"])
                    horizon_years = int(plan_for_cards["years"])
                    for order, (label, rate, note) in enumerate(scenarios, start=1):
                        final_value = _future_value(
                            lump_sum=plan_for_cards["lump_sum"],
                            monthly_contrib=plan_for_cards["monthly"],
                            years=plan_for_cards["years"],
                            annual_rate=rate,
                        )
                        final_fmt = _format_currency(final_value)
                        st.markdown(
                            f"{order}. **{label} (~{rate * 100:.1f}%/yr)**  \n"
                            f"   You invest {lump_fmt} now and {monthly_fmt}/month for {horizon_years} years. "
                            f"You contribute {total_fmt} total and could end up with about {final_fmt} in this scenario.  \n"
                            f"   _Risk view: {note}_"
                        )

                    pie_series = pd.Series(weights).sort_values(ascending=False)
                    if not pie_series.empty:
                        pie_df = pie_series.reset_index()
                        pie_df.columns = ["symbol", "weight"]
                        pie_df["weight_pct"] = pie_df["weight"] * 100
                        pie_chart = (
                            alt.Chart(pie_df)
                            .mark_arc(innerRadius=45, stroke="#0d0d12", strokeWidth=1)
                            .encode(
                                theta=alt.Theta(field="weight", type="quantitative"),
                                color=alt.Color("symbol:N", legend=None, scale=alt.Scale(scheme="tableau10")),
                                tooltip=[
                                    alt.Tooltip("symbol:N", title="Ticker"),
                                    alt.Tooltip("weight_pct:Q", title="Weight (%)", format=",.1f"),
                                ],
                            )
                            .properties(height=200)
                        )
                        st.altair_chart(pie_chart, use_container_width=True)
                    else:
                        st.caption("No weight data available to plot.")

                    if st.button("Use this portfolio", key=f"choose_{idx}_{name}"):
                        st.session_state["chosen_portfolio"] = name
                        st.success("Saved! Open the Dashboard for the full breakdown.")

                    st.markdown("---")
    else:
        # Legacy comparison and top 4 charts
        st.subheader("Portfolio Comparison (Legacy)")
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

        # Unified volatility accessor (fixes prior 0.0% display bug)
        def _get_vol(c):
            try:
                m = c.get("metrics", {}) if isinstance(c, dict) else {}
                v = m.get("Vol") if m.get("Vol") is not None else m.get("volatility")
                if v is None:
                    return 0.0
                return float(v)
            except Exception:
                return 0.0

        # Filter using TRUE_RISK instead of raw risk score
        from typing import Any, cast
        
        # DEBUG: Check cands before Risk Match logic
        if DEBUG_PHASE3 and not IS_PROD:
            with st.expander("🔍 DEBUG: Risk Match Section", expanded=False):
                st.write("**cands count:**", len(cands) if cands else 0)
                st.write("**cands type:**", type(cands))
                if cands:
                    st.write("**First cand name:**", cands[0].get("name") if len(cands) > 0 else "N/A")
        
        # Check if we have any candidates at all
        if not cands:
            st.error("❌ No portfolios available. Please run a simulation first.")
        else:
            filtered = select_candidates_for_risk_score(cast(Any, cands), float(TRUE_RISK))

            if not filtered:
                # Fallback: pick closest portfolio by volatility based on TRUE_RISK
                st.warning("⚠️ No exact matches in your risk band. Showing closest portfolio:")
                sigma_min, sigma_max = 0.1271, 0.2202
                target_vol = sigma_min + (sigma_max - sigma_min) * (TRUE_RISK / 100.0)
                closest = min(cands, key=lambda c: abs(_get_vol(c) - target_vol))
                filtered = [closest]

                st.caption(
                    f"**TRUE_RISK:** {TRUE_RISK:.0f}/100 → **Target volatility:** {target_vol:.1%}\n\n"
                    f"Selected portfolio volatility: {_get_vol(closest):.1%}"
                )
            else:
                st.success(f"Found **{len(filtered)}** matching portfolios")
                # Show target volatility band and first candidate's volatility for transparency
                sigma_min, sigma_max = 0.1271, 0.2202
                target_vol = sigma_min + (sigma_max - sigma_min) * (TRUE_RISK / 100.0)
                band = 0.02
                low = target_vol - band
                high = target_vol + band
                first_vol = _get_vol(filtered[0]) if filtered else 0.0
                st.caption(f"Target vol: {target_vol:.2%} (band: {low:.2%}–{high:.2%}) | First match vol: {first_vol:.2%}")

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
                # Confirm non-zero volatility readout for picked portfolio
                st.caption(f"Portfolio volatility: {_get_vol(picked):.2%}")

# ====================== MACRO ======================
elif current_page == "Macro":
    st.header("Macroeconomic Indicators")
    
    beginner_mode = st.session_state.get("beginner_mode", True)
    
    if beginner_mode:
        st.markdown("""
        These indicators provide **context** for understanding the economic environment. They don't 
        directly change your portfolio, but they help explain why certain asset classes may perform 
        differently in different conditions.
        
        Use this page to learn what economic factors influence markets, not to time entries or exits.
        """)
    else:
        st.markdown("Macroeconomic context for portfolio performance.")
    
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
                ),
                "portfolio_impact": (
                    "📊 **What this means for your portfolio:**\n\n"
                    "- **Rising inflation** → bonds lose purchasing power, equities may struggle if Fed tightens\n"
                    "- **Falling inflation** → bonds perform better, may signal economic weakness\n"
                    "- **Stable moderate inflation** → generally favorable for balanced portfolios\n\n"
                    "💡 *Don't chase inflation reads.* Your portfolio should already balance these risks through diversification."
                )
            },
            "FEDFUNDS": {
                "name": "Fed Funds Rate",
                "series": load_series("FEDFUNDS"),
                "explanation": (
                    "This is the **short-term interest rate** set by the Federal Reserve. "
                    "When it's high, cash and short-term bonds pay more, making riskier assets less attractive. "
                    "When it's low, investors often seek higher returns in stocks."
                ),
                "portfolio_impact": (
                    "📊 **What this means for your portfolio:**\n\n"
                    "- **High rates** → cash earns more, stocks may underperform as borrowing costs rise\n"
                    "- **Low rates** → encourages risk-taking, can boost stock valuations\n"
                    "- **Rate cuts** → often bullish for bonds and stocks, but may signal economic worry\n\n"
                    "💡 *Fed policy works slowly.* Don't overreact to single rate changes."
                )
            },
            "DGS10": {
                "name": "10-Year Treasury Yield",
                "series": load_series("DGS10"),
                "explanation": (
                    "This **long-term interest rate** is a key benchmark. "
                    "It affects mortgage rates and is often used as the 'risk-free' rate in portfolio models. "
                    "Rising yields can pressure stock prices."
                ),
                "portfolio_impact": (
                    "📊 **What this means for your portfolio:**\n\n"
                    "- **Rising yields** → bond prices fall, stocks may be less attractive (higher discount rates)\n"
                    "- **Falling yields** → bond prices rise, stocks may rally (flight to quality or growth optimism)\n"
                    "- **Yield curve inversion** (when 10Y < 2Y) → historically predicts recessions\n\n"
                    "💡 *This is the 'price of long-term money'.* It affects everything from mortgages to stock valuations."
                )
            },
            "UNRATE": {
                "name": "Unemployment Rate",
                "series": load_series("UNRATE"),
                "explanation": (
                    "This shows how tight or weak the **job market** is. "
                    "Very low unemployment usually means strong growth (good for stocks) but also inflation risk. "
                    "Very high unemployment can signal recessions."
                ),
                "portfolio_impact": (
                    "📊 **What this means for your portfolio:**\n\n"
                    "- **Low unemployment** → strong economy, may support stocks, but risk of inflation\n"
                    "- **Rising unemployment** → can signal economic slowdown, defensive assets may outperform\n"
                    "- **Persistently high unemployment** → recession risk, lower earnings for companies\n\n"
                    "💡 *This is a lagging indicator.* By the time unemployment spikes, markets often already reacted."
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
                    
                    if beginner_mode:
                        st.markdown(info["explanation"])
                        with st.expander("💡 Portfolio implications"):
                            st.markdown(info["portfolio_impact"])
                    else:
                        st.caption(info["explanation"])
                        
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
    
    beginner_mode = st.session_state.get("beginner_mode", True)
    
    if beginner_mode:
        st.markdown("""
        ### What you're seeing here
        
        This page shows the **quality and coverage of the data** used to build portfolio recommendations.
        
        **Key concepts:**
        - **Universe**: The set of ETFs eligible for inclusion in portfolios
        - **Data history**: How many years of price data each ETF has (more is better for backtesting)
        - **Providers**: Where the data comes from (Tiingo, Stooq, yfinance)
        - **Snapshot-based**: The system uses pre-validated data, so temporary API issues don't affect you
        
        This is "under the hood" information. You don't need to optimize anything here – it's for transparency.
        """)
    else:
        st.markdown("""
        Data universe health and provider coverage. System uses snapshot-based approach 
        to ensure runtime stability.
        """)
    
    st.markdown("---")
    
    # Friendly summary for beginner mode
    if beginner_mode:
        st.info("""
        **🎯 Bottom line:** The system has data for dozens of ETFs spanning many years. 
        Quality is good, providers are redundant for reliability. You're seeing the building 
        blocks used to construct portfolio recommendations.
        
        **Technical details** below if you're curious about the data sources.
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
        if beginner_mode:
            with st.expander("🔍 Provider Breakdown (technical)", expanded=False):
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
        else:
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
    
    # Phase 3 / Phase 4: Multi-factor engine diagnostics
    st.subheader("🔬 Multi-Factor Engine Diagnostics (Phase 3/4)")
    
    engine_used = st.session_state.get("last_run_settings", {}).get("engine")
    if engine_used == "phase3":
        st.success("✅ Last simulation used Phase 3 multi-factor engine")
        
        # Asset filtering receipts
        asset_receipts = st.session_state.get("mf_asset_receipts")
        if asset_receipts is not None and not asset_receipts.empty:
            with st.expander("📦 Asset Filtering Receipts", expanded=True):
                passed = asset_receipts[asset_receipts["passed"]]
                failed = asset_receipts[~asset_receipts["passed"]]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Evaluated", len(asset_receipts))
                with col2:
                    st.metric("✅ Passed", len(passed), delta=None, delta_color="normal")
                with col3:
                    st.metric("❌ Failed", len(failed), delta=None, delta_color="inverse")
                
                st.markdown("**Assets that passed quality filters:**")
                if not passed.empty:
                    display_cols = ["symbol", "asset_class", "core_satellite", "years", "sharpe", "vol", "max_dd"]
                    available_cols = [c for c in display_cols if c in passed.columns]
                    st.dataframe(
                        passed[available_cols].style.format({
                            "years": "{:.1f}",
                            "sharpe": "{:.2f}",
                            "vol": "{:.2%}",
                            "max_dd": "{:.2%}",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True
                    )
                
                if not failed.empty and len(failed) <= 20:
                    st.markdown("**Assets that failed (reasons):**")
                    fail_display_cols = ["symbol", "fail_reason", "years", "sharpe"]
                    available_fail_cols = [c for c in fail_display_cols if c in failed.columns]
                    st.dataframe(
                        failed[available_fail_cols],
                        use_container_width=True,
                        hide_index=True
                    )
                elif not failed.empty:
                    st.caption(f"{len(failed)} assets failed (too many to display all)")
        
        # Portfolio filtering receipts
        portfolio_receipts = st.session_state.get("mf_portfolio_receipts")
        if portfolio_receipts is not None and not portfolio_receipts.empty:
            with st.expander("📊 Portfolio Filtering Receipts", expanded=True):
                passed = portfolio_receipts[portfolio_receipts["passed"]]
                failed = portfolio_receipts[~portfolio_receipts["passed"]]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Generated", len(portfolio_receipts))
                with col2:
                    st.metric("✅ Passed", len(passed))
                with col3:
                    st.metric("❌ Failed", len(failed))
                
                st.markdown("**Portfolios that passed all filters:**")
                if not passed.empty:
                    display_cols = ["name", "optimizer", "sharpe", "vol", "max_dd", "div_ratio", "composite_score"]
                    available_cols = [c for c in display_cols if c in passed.columns]
                    st.dataframe(
                        passed[available_cols].style.format({
                            "sharpe": "{:.2f}",
                            "vol": "{:.2%}",
                            "max_dd": "{:.2%}",
                            "div_ratio": "{:.2f}",
                            "composite_score": "{:.3f}",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True
                    )
                
                if not failed.empty:
                    st.markdown("**Portfolios that failed (reasons):**")
                    fail_display_cols = ["name", "fail_reason", "sharpe", "vol", "max_dd"]
                    available_fail_cols = [c for c in fail_display_cols if c in failed.columns]
                    st.dataframe(
                        failed[available_fail_cols].style.format({
                            "sharpe": "{:.2f}",
                            "vol": "{:.2%}",
                            "max_dd": "{:.2%}",
                        }, na_rep="—"),
                        use_container_width=True,
                        hide_index=True
                    )
        
        # Debug bundle download
        st.markdown("---")
        st.subheader("📥 Debug Bundle")
        st.caption("Download complete diagnostic data for analysis or sharing with support")
        
        if st.button("Generate Debug Bundle", key="gen_debug_bundle"):
            import json
            from datetime import datetime as _dt
            
            try:
                debug_bundle = {
                    "generated_at": _dt.now().isoformat(),
                    "version": "4.5.0-phase3",
                    "config": {
                        "multifactor": CFG.get("multifactor", {}),
                        "optimization": CFG.get("optimization", {}),
                        "data": {k: v for k, v in CFG.get("data", {}).items() if k != "api_keys"},
                    },
                    "last_run": st.session_state.get("last_run_settings", {}),
                    "risk_profile": {
                        "risk_score": st.session_state.get("risk_score"),
                        "risk_score_questionnaire": st.session_state.get("risk_score_questionnaire"),
                        "risk_score_facts": st.session_state.get("risk_score_facts"),
                        "risk_score_combined": st.session_state.get("risk_score_combined"),
                    },
                    "asset_receipts": asset_receipts.to_dict(orient="records") if asset_receipts is not None else [],
                    "portfolio_receipts": portfolio_receipts.to_dict(orient="records") if portfolio_receipts is not None else [],
                    "recommended_portfolios": [
                        {
                            "name": p.get("name"),
                            "optimizer": p.get("optimizer"),
                            "sat_cap": p.get("sat_cap"),
                            "metrics": p.get("metrics"),
                            "weights": p.get("weights"),
                            "composite_score": p.get("composite_score"),
                        }
                        for p in st.session_state.get("mf_recommended", [])
                    ],
                }
                
                bundle_json = json.dumps(debug_bundle, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download debug_bundle.json",
                    data=bundle_json,
                    file_name=f"invest_ai_debug_{_dt.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_debug_bundle"
                )
                
                st.success("✅ Debug bundle ready for download")
                
                # Show bundle size
                bundle_size_kb = len(bundle_json.encode('utf-8')) / 1024
                st.caption(f"Bundle size: {bundle_size_kb:.1f} KB")
                
            except Exception as e:
                st.error(f"Failed to generate debug bundle: {e}")
    
    elif engine_used == "legacy":
        st.info("ℹ️ Last simulation used legacy engine. Enable Phase 3 on Portfolios page for advanced diagnostics.")
    else:
        st.info("ℹ️ Run a simulation on the **Portfolios** page to see diagnostics.")
    
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
    
    st.stop()

# ====================== SETTINGS ======================
elif current_page == "Settings":
    st.header("Settings")
    
    st.markdown("""
    ### Application preferences
    
    Configure how Invest AI presents information and explanations.
    """)
    
    st.markdown("---")
    
    st.subheader("Display Mode")
    
    current_mode = st.session_state.get("beginner_mode", True)
    
    beginner_mode_toggle = st.checkbox(
        "🎓 Beginner mode",
        value=current_mode,
        help="Show explanations, context, and educational content throughout the app. "
             "Disable for a more compact, expert-focused interface.",
        key="beginner_mode_toggle"
    )
    
    if beginner_mode_toggle != current_mode:
        st.session_state["beginner_mode"] = beginner_mode_toggle
        st.success(f"Beginner mode {'enabled' if beginner_mode_toggle else 'disabled'}. "
                   "Navigate to other pages to see changes.")
    
    st.markdown("---")
    
    st.markdown("""
    #### What beginner mode does
    
    When **enabled**:
    - Shows explanations and context for metrics, charts, and concepts
    - Uses qualitative labels alongside numbers (e.g., "Conservative" vs just "30/100")
    - Provides educational notes about portfolio construction principles
    - Explains what macro indicators mean for portfolio selection
    
    When **disabled**:
    - Focuses on data and numbers with minimal explanatory text
    - Assumes familiarity with financial concepts and portfolio theory
    - More compact interface for experienced users
    """)
    
    st.stop()

else:
    st.error(f"Unknown page: {current_page}")
    if st.button("Go to Landing"):
        st.session_state["page"] = "Landing"
        st.rerun()

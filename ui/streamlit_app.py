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

# --- Main header ---
st.title("📊 Invest AI - Portfolio Recommender")
st.markdown("""
Build an optimized investment portfolio tailored to your income, net worth, and risk tolerance.

**Quick Start:**
1. Configure your profile in the sidebar (income, net worth, risk)
2. Review eligible assets below
3. Click **Run Simulation** to generate your personalized portfolio

*Note: Data loads on first simulation run - this may take a moment.*
""")

# --- Sidebar ---
with st.sidebar:
    st.header("Your Profile")
    # Build version in sidebar (dot notation)
    try:
        version = get_build_version()
        version_dot = version.replace("+", ".")
        st.caption(f"Build: {version_dot}")
    except Exception:
        pass
    income = st.number_input("Annual income (AGI, USD)", min_value=0, value=90000, step=1000,
                             help="Used to estimate a safe monthly investment budget (10/20/30% heuristics).")
    networth = st.number_input("Net worth (exclude primary home)", min_value=0, value=150000, step=5000,
                               help="Used for class eligibility and one-time (lump-sum) capacity.")
    risk_pct = st.slider("Risk level (%)", 0, 100, 50, 1,
                         help="Higher % = higher expected volatility. Objectives and some assets require minimum risk.")
    objective = st.selectbox("Objective",
                             options=[("Grow capital","grow"),
                                      ("Grow + Income","grow_income"),
                                      ("Income","income"),
                                      ("Tax efficiency","tax_efficiency"),
                                      ("Capital preservation","preserve")],
                             index=0, format_func=lambda x: x[0])
    horizon = st.slider("Time horizon (years)", 1, 30, 10)
    st.caption("Lookback: using **full history** for backtests (no truncation).")

    st.markdown("---")
    opt_map = {"HRP (diversified)": "hrp", "MVO (max Sharpe / target vol)": "mvo"}
    opt_label = st.radio(
        "Optimization method",
        list(opt_map.keys()),
        index=0,
        help="HRP is robust & diversification-first. MVO targets risk/return but may be unstable on small samples.",
        key="opt_label",
    )
    st.session_state["opt_method"] = opt_map[opt_label]

    # --- Candidates controls ---
    st.markdown("### Candidates")
    n_candidates = st.slider("# Candidates", min_value=5, max_value=10, value=8, step=1)
    use_regime_nudge = st.checkbox("Use regime nudge", value=True)

    # Optional risk questionnaire → constraint overrides
    use_q = st.checkbox("Use risk questionnaire (override bounds)", value=False)
    constraint_overrides = None
    if use_q:
        with st.expander("Risk questionnaire", expanded=False):
            q_risk = st.selectbox("Risk attitude", ["very_low","low","moderate","high","very_high"], index=2)
            q_dd = st.selectbox("Drawdown tolerance", ["low","medium","high"], index=1)
            q_hz = st.number_input("Horizon (years)", min_value=1, max_value=40, value=horizon)
            q_inc = st.selectbox("Income stability", ["low","medium","high"], index=1)
            constraint_overrides = risk_profile_to_constraints({
                "risk_attitude": q_risk,
                "drawdown_tolerance": q_dd,
                "horizon_years": q_hz,
                "income_stability": q_inc,
            })

    st.markdown("### Presets & Sectors")
    preset_key = objective[1]
    st.caption("Apply a curated Asset Pool for your objective. You can add sectors/extras, then edit manually.")
    col_p1, col_p2 = st.columns([1,1])
    with col_p1:
        apply_preset = st.button("Apply preset", help="Prefill Asset Pool with a vetted mix for your objective.")
    with col_p2:
        clear_pool = st.button("Clear", help="Empty the Asset Pool field.")

    chosen_sectors = st.multiselect("Add sector ETFs (optional)", list(SECTOR_ETFS.keys()), default=[],
                                    help="Fine-tune exposures with SPDR sector ETFs.")
    add_extras = st.multiselect("Add extras (optional)", list(EXTRAS.keys()), default=[],
                                help="Broaden or stabilize the mix with international, commodities, REITs, or T-Bills.")

    default_pool = ",".join(CFG["data"]["default_universe"])
    pool_val = st.session_state.get("asset_pool_text", default_pool)

    if apply_preset:
        syms = PRESETS.get(preset_key, [])
        syms += [SECTOR_ETFS[k] for k in chosen_sectors]
        for k in add_extras: syms += EXTRAS[k]
        pool_val = ",".join(dedupe_keep_order([s.strip().upper() for s in syms]))
    if clear_pool:
        pool_val = ""
    pool = st.text_input("Asset Pool (symbols, comma-separated)", value=pool_val,
                         help="Assets considered for your portfolio.", key="asset_pool_text")

    run_btn = st.button("Run Simulation")

# --- Derived profile ---
tier = classify_investor(income=income, net_worth=networth)
budget = safe_investment_budget(income)
min_required = MIN_RISK_OBJ[objective[1]]

st.subheader("Your Investor Profile")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Investor class", tier.replace("_"," ").title())
with c2:
    st.metric("Min risk by objective", f"{min_required}%")
with c3:
    st.metric("Your selected risk", f"{risk_pct}%")

st.info("**Risk slider:** Objectives set a **minimum risk** (e.g., Grow ≥ 45%). You can always choose higher. "
        "Some assets also require a minimum risk and eligibility tier.")

st.caption(
    f"Suggested monthly from AGI (heuristic): "
    f"**Baseline** ${budget['baseline_monthly']:,} • "
    f"**Ambitious** ${budget['ambitious_monthly']:,} • "
    f"**Aggressive** ${budget['aggressive_monthly']:,}"
)

lump_baseline = round(networth * 0.02, 2)
lump_ambitious = round(networth * 0.05, 2)
lump_aggressive = round(networth * 0.10, 2)

# --- Eligibility ---
df_cat = pd.DataFrame(CAT["assets"])
df_cat = _ensure_unique_symbol_index(df_cat)
try:
    dropped = int(getattr(df_cat, "_dropped_dupes_count", 0))
except Exception:
    dropped = 0
if dropped > 0:
    st.info(
        f"Note: {dropped} duplicate catalog rows by symbol were removed for a stable simulation. "
        "This can happen when multiple providers contribute entries for the same ticker."
    )

def tix(name: str) -> int: return TIER_ORDER.index(name)
df_cat["eligible_by_tier"] = df_cat["min_tier"].apply(lambda t: tix(tier) >= tix(t))
df_cat["eligible_by_risk"] = df_cat["min_risk_pct"].apply(lambda mr: risk_pct >= max(mr, min_required))
df_cat["eligible_now"] = df_cat["eligible_by_tier"] & df_cat["eligible_by_risk"]

st.subheader("Eligible Assets (based on your class & risk)")
with st.expander("What does 'eligible' mean?", expanded=False):
    st.write("""
    - **Eligible by tier**: your investor class (Retail, Accredited, etc.) can access the instrument.
    - **Eligible by risk**: your chosen risk level is above both the objective minimum and the asset's minimum.
    - The optimizer only allocates across assets that are **eligible now**.
    """)

# Attach history length (years) per symbol for transparency
try:
    _syms = list(df_cat.index)
    _px = get_prices(_syms, start="1900-01-01")
    hist_years = (_px.groupby("ticker")["date"].max() - _px.groupby("ticker")["date"].min()).dt.days / 365.25
    hist_years = hist_years.rename("hist_years")
    df_cat = df_cat.join(hist_years, how="left")
except Exception:
    df_cat["hist_years"] = None

st.dataframe(
    df_cat.assign(symbol=df_cat.index)[["name","symbol","class","min_tier","min_risk_pct","eligible_now","hist_years"]]
)

# --- Tabs for V3 UI ---
tabs = st.tabs(["Portfolio", "Macro", "Debug"])

# --- Price coverage stats (from Stooq cache) ---
from pathlib import Path as _Path
import pandas as _pd
_cache_dir = _Path(
    CFG.get("paths", {}).get("cache_dir")
    or CFG.get("data", {}).get("cache_dir")
    or "data/cache"
)
syms_in_pool = [s.strip().upper() for s in pool.split(",") if s.strip()]
stats_rows = []
for s in syms_in_pool:
    f = _cache_dir / f"{s}.csv"
    if f.exists():
        try:
            d = _pd.read_csv(f, parse_dates=["date"])
            if not d.empty and {"date","price"}.issubset(d.columns):
                stats_rows.append({
                    "symbol": s,
                    "first_date": d["date"].min().date().isoformat(),
                    "last_date": d["date"].max().date().isoformat(),
                    "rows": len(d)
                })
        except Exception:
            pass
if stats_rows:
    st.markdown("**Price coverage (from Stooq cache):**")
    st.dataframe(_pd.DataFrame(stats_rows))

# ---------- Run Simulation ----------
if run_btn:
    try:
        # Lazy load prices on first simulation run
        if st.session_state["prices_loaded"] is None:
            with st.spinner("Loading market data (first run)..."):
                try:
                    _syms = CFG.get("data", {}).get("default_universe", ["SPY"])
                    _px, _prov = get_prices_with_provenance(_syms, start="1900-01-01")
                    st.session_state["prices_loaded"] = _px
                    st.session_state["prov_loaded"] = _prov
                except Exception as e:
                    st.error(f"Failed to load market data: {e}")
                    st.session_state["prices_loaded"] = pd.DataFrame()
                    st.session_state["prov_loaded"] = {}
        
        symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
        eligible_mask = df_cat.index.isin(symbols) & df_cat["eligible_now"].astype(bool)
        eligible_symbols = list(df_cat.index[eligible_mask])
        if not eligible_symbols:
            st.error("No eligible tradable symbols in your Asset Pool. Adjust risk or pool.")
            st.stop()
        prices = get_prices(eligible_symbols, start="1900-01-01")
        if prices.empty:
            st.error("No price data returned (source/network). Try fewer/different symbols.")
            st.stop()
    # prices is already in wide format (columns=symbols, index=date)
        rets = compute_returns(prices)
        prof = UserProfile(monthly_contribution=budget["baseline_monthly"], horizon_years=horizon, risk_level="moderate")
        rec = recommend(rets, prof, objective=objective[1], risk_pct=risk_pct, catalog=CAT, method=st.session_state.get("opt_method", "hrp"))
        w = pd.Series(rec.get("weights", {})).sort_values(ascending=False)
        metrics = rec.get("metrics", {})
        curve = rec.get("curve", pd.Series(dtype=float))
        wvec = pd.Series(w).reindex(rets.columns).fillna(0.0)
        port_ret = rets.fillna(0).dot(wvec)
        cov_pct = data_coverage(rets)
        effN = effective_n_assets(w.to_dict())
        try:
            ci_lo, ci_hi = bootstrap_interval(port_ret, stat="cagr", n=300, alpha=0.10)
        except Exception:
            ci_lo, ci_hi = float("nan"), float("nan")
        try:
            rolling_vol = port_ret.rolling(21).std()
            thr = rolling_vol.quantile(0.8)
            groups = pd.Series(pd.NA, index=port_ret.index)
            groups.loc[rolling_vol >= thr] = "high_vol"
            groups.loc[rolling_vol < thr] = "normal"
            anova = anova_mean_returns(port_ret, groups)
        except Exception:
            anova = {"F": float("nan"), "p": float("nan"), "k": 2}
        cred = credibility_score(
            n_obs=int(len(port_ret)),
            cov=float(cov_pct),
            effN=float(effN),
            sharpe_is=float(metrics.get("Sharpe", 0.0)),
            sharpe_oos=None,
        )
        with st.expander("Methodology & diagnostics", expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Credibility score", f"{cred['score']}/100")
            c2.metric("Coverage", f"{cov_pct*100:.1f}%")
            c3.metric("Effective N", f"{effN:.2f}")
            st.write(f"Bootstrap 90% CI (1Y CAGR): {ci_lo*100:.2f}% → {ci_hi*100:.2f}%")
            st.write(f"ANOVA: F={anova['F']:.3f}, p={anova['p']:.3f}, groups={anova['k']}")
        a, b = st.columns(2)
        with a:
            st.subheader("Suggested Allocation (within eligible assets)")
            try:
                receipts = pd.DataFrame({
                    'Asset': w.index,
                    'Weight': w.values,
                    'History (years)': df_cat['hist_years'].reindex(w.index.astype(str)).round(1),
                    'Provider': pd.Series(rec.get('providers', {})),
                    'Eligible': df_cat['eligible_now'].reindex(w.index.astype(str))
                })
                st.dataframe(w.to_frame("weight"))
                csv = receipts.to_csv(index=False)
                st.download_button("Download receipts CSV", csv, "receipts.csv", "text/csv", key='download-receipts')
                st.bar_chart(w)
                st.info("**Allocation Policy:**  \nCore ≥65% of portfolio. Satellites ≤35% total and ≤7% each for diversification.")
            except Exception as e:
                st.error(f"Portfolio table build failed: {e}")
                import traceback as _tb
                st.code("".join(_tb.format_exc()))
                st.stop()
        with b:
            st.subheader("Backtest Metrics")
            with st.expander("What do these mean?", expanded=False):
                st.markdown("""- **CAGR** annualized growth.\n- **Vol** annualized volatility.\n- **Sharpe** return/vol.\n- **MaxDD** largest peak→trough drop.\n- **N** observations.""")
            try:
                _m = pd.Series(metrics)
                st.table(_m.to_frame("value"))
            except Exception:
                st.write(metrics)
            st.subheader("Cumulative Growth (normalized)")
            try:
                bench_df = get_prices(["SPY"])
                if not bench_df.empty:
                    # bench_df is already wide format
                    _r = compute_returns(bench_df)
                    bench_curve = (1 + _r["SPY"].reindex(curve.index).fillna(0)).cumprod()
                    to_plot = pd.concat({"Portfolio": curve, "SPY (normalized)": bench_curve}, axis=1).dropna()
                else:
                    to_plot = pd.DataFrame({"Portfolio": curve})
            except Exception:
                to_plot = pd.DataFrame({"Portfolio": curve})
            try:
                to_plot_ds = to_plot.resample("W").last() if hasattr(to_plot, "resample") else to_plot
            except Exception:
                to_plot_ds = to_plot
            st.line_chart(to_plot_ds)
            try:
                start_dt = curve.index[0].date()
                end_dt = curve.index[-1].date()
                growth_pct = (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1.0) * 100.0
                st.caption(f"Backtest window: {start_dt} → {end_dt} • Normalized growth: **{growth_pct:.2f}%**")
            except Exception:
                pass

        # --- Contribution scenarios ---
        st.subheader("Contribution Plans — Projected Outcome (MVP)")
        plans = [
            ("Baseline", lump_baseline, budget["baseline_monthly"]),
            ("Ambitious", lump_ambitious, budget["ambitious_monthly"]),
            ("Aggressive", lump_aggressive, budget["aggressive_monthly"]),
        ]
        rows = []
        for name, lump, monthly in plans:
            total = project_lump_and_dca(curve, lump=lump, monthly=monthly)
            rows.append({
                "Plan": name,
                "One-time (lump-sum $)": f"{lump:,.0f}",
                "Monthly ($)": f"{monthly:,.0f}",
                "Projected Final ($)": f"{total:,.0f}"
            })
        st.table(pd.DataFrame(rows))
        st.caption("Projections reuse the historical backtest curve (MVP) — not guarantees.")

        st.subheader("Contribution Paths (calendar-accurate DCA)")
        path_baseline = simulate_dca_calendar_series(
            curve, monthly=budget['baseline_monthly'], lump=lump_baseline
        )
        path_ambitious = simulate_dca_calendar_series(
            curve, monthly=budget['ambitious_monthly'], lump=lump_ambitious
        )
        path_aggressive = simulate_dca_calendar_series(
            curve, monthly=budget['aggressive_monthly'], lump=lump_aggressive
        )
        paths = pd.concat(
            {"Baseline": path_baseline, "Ambitious": path_ambitious, "Aggressive": path_aggressive},
            axis=1
        ).dropna(how="all")
        try:
            st.line_chart(paths.resample("M").last())
        except Exception:
            st.line_chart(paths)

        if not paths.empty:
            finals = paths.tail(1).T.reset_index()
            finals.columns = ["Plan", "Projected Final ($)"]
            finals["Projected Final ($)"] = finals["Projected Final ($)"].map(lambda x: f"{x:,.0f}")
            finals["Monthly ($)"] = [
                f"{budget['baseline_monthly']:,.0f}",
                f"{budget['ambitious_monthly']:,.0f}",
                f"{budget['aggressive_monthly']:,.0f}",
            ]
            finals["One-time (lump-sum $)"] = [
                f"{lump_baseline:,.0f}",
                f"{lump_ambitious:,.0f}",
                f"{lump_aggressive:,.0f}",
            ]
            finals = finals[["Plan", "One-time (lump-sum $)", "Monthly ($)", "Projected Final ($)"]]
            st.table(finals)

        st.caption("Calendar DCA buys on month-end (nearest trading day). Lump sum at start date. "
                   "This is a historical replay — not a forecast.")

        # Compute month count and warn if short window
        try:
            # Robust month count: count unique year-month pairs
            if not paths.empty:
                n_months = len(set((d.year, d.month) for d in paths.index))
            else:
                n_months = 0
            if n_months < 36:
                st.warning(f"Data window is short ({n_months} months). Results may be noisy or misleading.")
        except Exception:
            n_months = None

        # Helper: invested-so-far series (lump at t0, monthly on month-end)
        def _invested_series(index, monthly, lump):
            import pandas as _pd
            s = _pd.Series(0.0, index=index)
            if len(index) == 0:
                return s
            s.iloc[0] = float(lump or 0.0)
            if monthly and monthly > 0:
                month_end_idx = _pd.Series(1, index=index).resample("ME").last().index
                used = set()
                for d in month_end_idx:
                    pos = index.get_indexer([d], method="pad")
                    if pos[0] == -1:
                        continue
                    ts = index[pos[0]]
                    if ts in used:
                        continue
                    used.add(ts)
                    s.loc[ts] += monthly
            return s.cumsum().ffill()

        inv_baseline = _invested_series(paths.index, budget['baseline_monthly'], lump_baseline) if not paths.empty else None
        inv_ambitious = _invested_series(paths.index, budget['ambitious_monthly'], lump_ambitious) if not paths.empty else None
        inv_aggressive = _invested_series(paths.index, budget['aggressive_monthly'], lump_aggressive) if not paths.empty else None

        # Detailed receipt table
        try:
            import pandas as _pd
            finals_map = paths.tail(1).iloc[0].to_dict() if not paths.empty else {}
            plan_specs = [
                ("Baseline", lump_baseline, budget['baseline_monthly'], inv_baseline, "Baseline"),
                ("Ambitious", lump_ambitious, budget['ambitious_monthly'], inv_ambitious, "Ambitious"),
                ("Aggressive", lump_aggressive, budget['aggressive_monthly'], inv_aggressive, "Aggressive"),
            ]
            rows = []
            for name, lump, monthly, inv_ser, key in plan_specs:
                invested_total = float(inv_ser.iloc[-1]) if inv_ser is not None and len(inv_ser)>0 else float((lump or 0.0) + (monthly or 0.0) * (n_months or 0))
                final_val = float(finals_map.get(key, 0.0))
                profit = final_val - invested_total
                roi = (profit / invested_total) * 100.0 if invested_total > 0 else 0.0
                xirr_val = 0.0
                try:
                    if inv_ser is not None and len(inv_ser) > 0:
                        inc = inv_ser.diff().fillna(inv_ser.iloc[0])
                        cf = []
                        for dt, amt in inc.items():
                            if amt != 0:
                                # Robust conversion for pandas Timestamp or datetime
                                if hasattr(dt, 'to_pydatetime'):
                                    cf.append((dt.to_pydatetime(), -float(amt)))  # type: ignore
                                else:
                                    cf.append((dt, -float(amt)))
                        last_dt = paths.index[-1]
                        if hasattr(last_dt, 'to_pydatetime'):
                            cf.append((last_dt.to_pydatetime(), float(final_val)))  # type: ignore
                        else:
                            cf.append((last_dt, float(final_val)))
                        # compute_xirr expects a pandas Series: index=date, value=amount
                        import pandas as _pd
                        cf_ser = _pd.Series([amt for (dt, amt) in cf], index=[dt for (dt, amt) in cf])
                        xirr_val = compute_xirr(cf_ser) * 100.0
                except Exception:
                    xirr_val = 0.0
                rows.append({
                    "Plan": name,
                    "Lump ($)": f"{lump:,.0f}",
                    "Monthly ($)": f"{monthly:,.0f}",
                    "Months": n_months if n_months is not None else "",
                    "Total Invested ($)": f"{invested_total:,.0f}",
                    "Final Value ($)": f"{final_val:,.0f}",
                    "Profit/Loss ($)": f"{profit:,.0f}",
                    "ROI (%)": f"{roi:,.2f}",
                    "XIRR (cash-flow annualized, %)": f"{xirr_val:,.2f}",
                })
            st.subheader("Contribution Receipt (detailed)")
            st.table(_pd.DataFrame(rows))
        except Exception as e:
            st.caption(f"(Could not build detailed receipt: {e})")

        # Per-plan receipt expanders
        with st.expander("See per-plan calculation receipts", expanded=False):
            try:
                for lbl, lump, monthly, inv_ser, key in [
                    ("Baseline", lump_baseline, budget['baseline_monthly'], inv_baseline, "Baseline"),
                    ("Ambitious", lump_ambitious, budget['ambitious_monthly'], inv_ambitious, "Ambitious"),
                    ("Aggressive", lump_aggressive, budget['aggressive_monthly'], inv_aggressive, "Aggressive"),
                ]:
                    if inv_ser is None or paths.empty:
                        continue
                    final_val = float(paths[key].iloc[-1])
                    invested_total = float(inv_ser.iloc[-1])
                    profit = final_val - invested_total
                    st.markdown(
                        f"**{lbl}**  \n"
                        f"Lump: ${lump:,.0f} at start  \n"
                        f"Monthly: ${monthly:,.0f} on month-end × **{n_months}** months  \n"
                        f"Invested total: **${invested_total:,.0f}**  \n"
                        f"Final value: **${final_val:,.0f}**  \n"
                        f"Profit/Loss: **${profit:,.0f}**"
                    )
            except Exception:
                pass

        # Absolute value chart with invested-so-far (Altair)
        try:
            import pandas as _pd, altair as alt
            chart_df = _pd.DataFrame(index=paths.index)
            chart_df["Baseline (value)"] = paths["Baseline"]
            chart_df["Ambitious (value)"] = paths["Ambitious"]
            chart_df["Aggressive (value)"] = paths["Aggressive"]
            if inv_baseline is not None: chart_df["Baseline (invested)"] = inv_baseline
            if inv_ambitious is not None: chart_df["Ambitious (invested)"] = inv_ambitious
            if inv_aggressive is not None: chart_df["Aggressive (invested)"] = inv_aggressive
            chart_df = chart_df.reset_index().rename(columns={"index":"date"})
            long = chart_df.melt("date", var_name="series", value_name="value").dropna()

            st.subheader("Absolute Value Paths vs Invested (calendar-accurate DCA)")
            line = alt.Chart(long).mark_line().encode(
                x="date:T",
                y=alt.Y("value:Q", title="USD"),
                color="series:N"
            )
            st.altair_chart(line, use_container_width=True)
        except Exception as e:
            st.caption(f"(Altair chart not available: {e})")

        # Transparency block: show formulas and invested vs final
        st.subheader("Transparency — how we compute projections")
        if not paths.empty:
            # Robust month count: count unique year-month pairs
            if not paths.empty:
                months = len(set((d.year, d.month) for d in paths.index))
            else:
                months = 0
            finals_map = paths.tail(1).iloc[0].to_dict()
            plan_specs = [
                ("Baseline", lump_baseline, budget['baseline_monthly']),
                ("Ambitious", lump_ambitious, budget['ambitious_monthly']),
                ("Aggressive", lump_aggressive, budget['aggressive_monthly']),
            ]
            rows_t = []
            for name, lump, monthly in plan_specs:
                invested = (lump or 0.0) + (monthly or 0.0) * months
                final_val = float(finals_map.get(name, 0.0))
                profit = final_val - invested
                roi = (profit / invested) * 100.0 if invested > 0 else 0.0
                rows_t.append({
                    "Plan": name,
                    "Months": months,
                    "Invested total ($)": f"{invested:,.0f}",
                    "Projected Final ($)": f"{final_val:,.0f}",
                    "Profit / (Loss) ($)": f"{profit:,.0f}",
                    "ROI (%)": f"{roi:,.1f}"
                })
            st.table(pd.DataFrame(rows_t))

            with st.expander("Formula details", expanded=False):
                st.markdown("""
                **Normalized cumulative growth**: index starts at 1.0 on day 1. End index 1.25 = +25% cumulative growth.

                **Shares logic**  
                - Lump shares at start: `shares_lump = lump / index[start]`  
                - Monthly buys: for each month-end `d`, `shares_month[d] = monthly / index[d]`  
                - Total shares = lump shares + sum of monthly shares

                **Portfolio value**  
                - On date `t`: `value[t] = total_shares × index[t]`  
                - **Projected Final** = `value[last_day]`

                **Contributions & profit**  
                - **Invested total** = `lump + monthly × (number of months)`  
                - **Profit** = `Projected Final − Invested total`  
                - **ROI%** = `Profit / Invested total`
                """)

    except Exception as e:
        import traceback
        st.error(f"Simulation failed: {e}")
        with st.expander("Traceback (debug)"):
            st.code(traceback.format_exc())
        st.stop()
else:
    st.write("Use the sidebar, apply a preset (optionally add sectors/extras), review eligibility, then click **Run Simulation**.")

# ==================== Portfolio Tab: Candidates ====================
with tabs[0]:
    st.subheader("Candidates")
    try:
        # Use current pool and objective
        symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
        eligible_mask = df_cat.index.isin(symbols) & df_cat["eligible_now"].astype(bool)
        eligible_symbols = list(df_cat.index[eligible_mask])
        if len(eligible_symbols) >= 3:
            prices = get_prices(eligible_symbols, start="1900-01-01")
            rets = compute_returns(prices)
            # Objective config
            obj_key = objective[1]
            obj_cfg = DEFAULT_OBJECTIVES.get(obj_key)
            if obj_cfg is None:
                obj_cfg = ObjectiveConfig(name=obj_key)
            elif isinstance(obj_cfg, dict):
                obj_cfg = ObjectiveConfig(**obj_cfg)
            # Ensure bounds is always a dict
            if not hasattr(obj_cfg, "bounds") or obj_cfg.bounds is None:
                obj_cfg.bounds = {"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07}
            if constraint_overrides:
                b = dict(obj_cfg.bounds)
                b["core_min"] = constraint_overrides.get("core_min", b.get("core_min", 0.65))
                b["sat_max_total"] = constraint_overrides.get("satellite_max", b.get("sat_max_total", 0.35))
                b["sat_max_single"] = constraint_overrides.get("single_max", b.get("sat_max_single", 0.07))
                obj_cfg.bounds = b
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
            else:
                # ...existing code for metrics, table, expanders, exports...
                last_1y = rets.tail(252)
                try:
                    spy_w = get_prices(["SPY"], start=str(last_1y.index.min().date()))
                    spy_r = compute_returns(spy_w)["SPY"].reindex(last_1y.index).dropna()
                except Exception:
                    spy_r = pd.Series(dtype=float)
                rows = []
                curves = {}
                for cand in cands:
                    score = 0.0  # Initialize score for each candidate
                    w = pd.Series(cand["weights"]).reindex(rets.columns).fillna(0.0)
                    port_1y = (last_1y * w).sum(axis=1)
                    m = annualized_metrics(port_1y)
                    beta = beta_vs_benchmark(port_1y, spy_r) if not spy_r.empty else float("nan")
                    var95 = var_95(port_1y)
                    eq_pct = 0.0
                    bd_pct = 0.0
                    for t, wt in cand["weights"].items():
                        cls = df_cat.loc[str(t), "class"] if str(t) in df_cat.index else "unknown"
                        if isinstance(cls, str) and (cls.startswith("public_equity") or cls in {"public_equity","public_equity_intl","public_equity_em"}):
                            eq_pct += wt
                        elif cls in {"treasury_long","tbills","corporate_bond","high_yield","tax_eff_muni","public_bond","muni_bond"}:
                            bd_pct += wt
                    regime_sharpe = float("nan")
                    if use_regime_nudge:
                        try:
                            sw = port_1y.tail(63)
                            mm = annualized_metrics(sw)
                            regime_sharpe = mm.get("Sharpe", float("nan"))
                        except Exception:
                            pass
                    maxw = max(cand["weights"].values()) if cand["weights"] else 0.0
                    concentration_penalty = max(0.0, maxw - 0.20)
                    mix_penalty = max(0.0, 0.40 - bd_pct) + max(0.0, eq_pct - 0.85)
                    score: float = (
                        (m.get("Sharpe", 0.0) or 0.0)
                        - 0.2 * abs(m.get("MaxDD", 0.0) or 0.0)
                        - 0.1 * concentration_penalty
                        - 0.1 * mix_penalty
                        + (0.1 * (regime_sharpe if use_regime_nudge and not pd.isna(regime_sharpe) else 0.0))
                    )
                    curve = (1 + (rets * w).sum(axis=1)).cumprod().rename(cand["name"]).dropna()
                    curves[cand["name"]] = curve
                    rows.append({
                        "Name": cand["name"],
                        "Score": round(score, 4),
                        "Sharpe_1Y": round(m.get("Sharpe", float("nan")), 3) if m.get("Sharpe") is not None else float("nan"),
                        "CAGR_1Y": round(m.get("CAGR", float("nan")), 3) if m.get("CAGR") is not None else float("nan"),
                        "Vol_1Y": round(m.get("Volatility", float("nan")), 3) if m.get("Volatility") is not None else float("nan"),
                        "MaxDD_1Y": round(m.get("MaxDD", float("nan")), 3) if m.get("MaxDD") is not None else float("nan"),
                        "Beta": round(beta, 3) if pd.notna(beta) else float("nan"),
                        "VaR95": round(var95, 4) if pd.notna(var95) else float("nan"),
                        "Equity%": round(eq_pct * 100.0, 1),
                        "Bonds%": round(bd_pct * 100.0, 1),
                        "_weights": cand["weights"],
                    })
                df_rows = pd.DataFrame(rows).sort_values(by="Score", ascending=False)
                st.caption(f"Nudged ranking: {'ON' if use_regime_nudge else 'OFF'}")
                st.dataframe(df_rows[["Name","Score","Sharpe_1Y","CAGR_1Y","Vol_1Y","MaxDD_1Y","Beta","VaR95","Equity%","Bonds%"]], use_container_width=True)
                for idx, r in enumerate(df_rows.itertuples(index=False)):
                    name = getattr(r, "Name")
                    expanded = idx < 3
                    with st.expander(f"Details: {name}", expanded=expanded):
                        st.write("Weights")
                        st.dataframe(pd.Series(getattr(r, "_weights")).sort_values(ascending=False).to_frame("weight"))
                        try:
                            curve = curves.get(name)
                            if curve is not None:
                                try:
                                    bench_curve = (1 + spy_r.reindex(curve.index).fillna(0)).cumprod().rename("SPY (norm)")
                                    plot_df = pd.concat([curve.rename("Portfolio"), bench_curve], axis=1).dropna()
                                    plot_df = plot_df.resample("W").last()
                                except Exception:
                                    plot_df = curve.to_frame("Portfolio")
                                st.line_chart(plot_df)
                        except Exception:
                            pass
                try:
                    res = anova_bootstrap(curves)
                    badge = f"Shortlist (bootstrap ANOVA N=2000): {res.get('winner') or 'n/a'} (p={res.get('p_value') if res.get('p_value') is not None else 'n/a'})"
                    st.success(badge)
                except Exception as e:
                    st.caption(f"(ANOVA bootstrap unavailable: {e})")
                import json, os
                artifacts = ROOT / "dev" / "artifacts"
                artifacts.mkdir(parents=True, exist_ok=True)
                csv_path = artifacts / "candidates.csv"
                json_path = artifacts / "candidates.json"
                df_rows.drop(columns=["_weights"], errors="ignore").to_csv(csv_path, index=False)
                with open(json_path, "w") as f:
                    json.dump({"candidates": rows}, f, indent=2)
                st.download_button("Export candidates CSV", data=csv_path.read_bytes(), file_name="candidates.csv", mime="text/csv")
                st.download_button("Export candidates JSON", data=json_path.read_bytes(), file_name="candidates.json", mime="application/json")
                try:
                    winner = df_rows.iloc[0]
                    import io
                    buf = io.StringIO()
                    pd.Series(winner["_weights"]).sort_values(ascending=False).to_csv(buf, header=["weight"])
                    st.download_button("Export shortlist weights", data=buf.getvalue(), file_name="shortlist_weights.csv", mime="text/csv")
                except Exception:
                    pass
        else:
            st.info("Select ≥3 eligible symbols to generate candidates.")
    except Exception as e:
        st.error(f"Candidates failed: {e}")

# ==================== Macro Tab ====================
with tabs[1]:
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
                # Staleness policy
                limit = 60 if name in {"DGS10","T10Y2Y"} else 90
                if age_days > limit:
                    stale_msgs.append(f"{name} stale ({age_days}d>")
                col.line_chart(s.tail(365))
            except Exception:
                pass
        if stale_msgs:
            st.warning("; ".join(stale_msgs))
        st.caption(f"Nudged ranking: {'ON' if use_regime_nudge else 'OFF'}")
    except Exception as e:
        st.error(f"Macro load failed: {e}")

# ==================== Debug / Receipts Tab ====================
with tabs[2]:
    st.subheader("Receipts")
    try:
        # Build simple receipts from last loaded providers if available
        prov = st.session_state.get("prov_loaded", {}) or {}
        if prov:
            try:
                from core.utils.receipts import build_receipts
                prices_loaded = st.session_state.get("prices_loaded")
                if prices_loaded is not None and not isinstance(prices_loaded, pd.DataFrame):
                    prices_loaded = None
            except Exception:
                prices_loaded = None
            # Build a tiny table from provenance
            rows = []
            for sym, pinfo in prov.items():
                first = pinfo.get("first_date") if isinstance(pinfo, dict) else None
                last = pinfo.get("last_date") if isinstance(pinfo, dict) else None
                rows.append({
                    "ticker": sym,
                    "provider": pinfo.get("provider") if isinstance(pinfo, dict) else str(pinfo),
                    "backfill_pct": pinfo.get("backfill_pct", None) if isinstance(pinfo, dict) else None,
                    "first": first,
                    "last": last,
                    "hist_years": pinfo.get("hist_years", None) if isinstance(pinfo, dict) else None,
                    "nan_rate": pinfo.get("nan_rate", None) if isinstance(pinfo, dict) else None,
                    "ann_vol": pinfo.get("ann_vol", None) if isinstance(pinfo, dict) else None,
                    "sharpe": pinfo.get("sharpe", None) if isinstance(pinfo, dict) else None,
                })
            df_rec = pd.DataFrame(rows)
            st.dataframe(df_rec)
            import json
            data_csv = df_rec.to_csv(index=False).encode()
            data_json = json.dumps(rows, indent=2).encode()
            st.download_button("Download receipts CSV", data=data_csv, file_name="receipts.csv")
            st.download_button("Download receipts JSON", data=data_json, file_name="receipts.json")
        else:
            st.caption("No provider receipts yet — run a simulation once.")
    except Exception as e:
        st.error(f"Receipts build failed: {e}")

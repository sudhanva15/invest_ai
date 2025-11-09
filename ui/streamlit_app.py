import sys
from pathlib import Path

# Add repo root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# OLD (and causing ImportError / tuple misuse)
# from core.data_ingestion import get_prices_router_merge as get_prices

# NEW
from core.data_ingestion import get_prices, get_prices_with_provenance

# --- Data/Methodology helpers -------------------------------------------------
def _ui_fetch_prices_router(symbols):
    """
    Fetch long-history, adjusted-close-preferred prices via router.
    Returns (df_long, source_map) where df_long columns:
    ['date','open','high','low','close','adj_close','volume','ticker']
    """
    import pandas as pd
    parts, sources = [], {}
    for s in symbols:
        df, prov = router_fetch_daily_smart(s)   # earliest available, Tiingo precedence
        if df is None or len(df)==0:
            continue
        df = normalize_price_columns(df)         # ensures 'price' + parsed 'date'
        parts.append(df[ [c for c in ["date","open","high","low","close","adj_close","volume","ticker"] if c in df.columns] ].copy())
        sources[s] = prov
    if not parts:
        return pd.DataFrame(columns=["date","ticker","price"]), sources
    allf = pd.concat(parts, ignore_index=True)
    # Make a tidy frame (date, ticker, price=adj_close/close)
    allf["price"] = price_series(allf)
    allf = allf[["date","ticker","price"]].dropna().sort_values(["ticker","date"]).reset_index(drop=True)
    return allf, sources

def _ui_source_badge(sources: dict, chosen_col: str = "adj_close"):
    import streamlit as st
    if not sources:
        st.caption("Price source: *none*")
        return
    # summarize providers
    uniq = sorted(set(sources.values()))
    src_text = " + ".join(uniq) if uniq else "n/a"
    tip = (
        "Data routing uses Tiingo for long, adjusted series with Stooq as a free backup. "
        "When both exist on the same date, Tiingo wins (precedence). "
        "Backtest math prefers **adjusted close** when available, else **close**."
    )
    st.caption(f"**Price source:** {src_text}  ·  **Price column:** {chosen_col}")
    with st.expander("Methodology (tap to view)"):
        st.markdown(
            "- **Adjusted close (preferred):** includes splits/dividends → appropriate for total-return math.\n"
            "- **Vendor precedence:** Tiingo over Stooq on overlapping dates.\n"
            "- **Normalization:** for growth charts, we normalize to start at 1.0 for relative comparison.\n"
            "- **CAGR / Vol:** annualized from daily returns (≈252 trading days/year)."
        )
# ------------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
from core.utils.env_tools import load_env_once
# Robust imports for config/json utilities (fallbacks if not re-exported)
try:
    from core.utils import load_config, load_json  # preferred if re-exported
except Exception:
    # Fallback: load directly from env_tools or implement tiny helpers
    try:
        from core.utils.env_tools import load_config  # type: ignore
    except Exception:
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
from core.data_ingestion import get_prices
from core.data_sources.router_smart import router_fetch_daily_smart
from core.backtesting import normalize_price_columns, price_series
from core.preprocessing import to_wide, compute_returns
from core.recommendation_engine import recommend, UserProfile, anova_mean_returns
from core.simulation_runner import simulate_contributions, simulate_dca_calendar_series, compute_xirr
from core.investor_profiles import classify_investor, safe_investment_budget, min_risk_by_objective
from core.trust import credibility_score, data_coverage, effective_n_assets, bootstrap_interval


st.set_page_config(page_title="invest_ai (MVP)", layout="wide")
st.title("📈 invest_ai — Portfolio Simulator (MVP)")
st.caption("Education & simulation only. Not investment advice.")

# --- Quick status strip: providers & env presence ----------------------------
try:
    from core.data_sources.provider_registry import get_ordered_providers as _get_provs
    _provs = _get_provs(use_yfinance_fallback=True)
    _prov_names = [getattr(p, "provider_name", getattr(p, "__name__", "unknown")) for p in _provs]
except Exception:
    _prov_names = []
try:
    import os as _os
    _env_flags = {
        "TIINGO": bool(_os.getenv("TIINGO_API_KEY")),
        "FRED": bool(_os.getenv("FRED_API_KEY") or _os.getenv("FRED_API_TOKEN")),
        "POLYGON": bool(
            _os.getenv("POLYGON_API_KEY")
            or _os.getenv("POLYGON_API_TOKEN")
            or _os.getenv("POLYGON_ACCESS_KEY_ID")
            or _os.getenv("POLYGON_SECRET_ACCESS_KEY")
        ),
    }
except Exception:
    _env_flags = {}
if _prov_names or _env_flags:
    st.caption(f"**Providers detected:** {', '.join(_prov_names) if _prov_names else 'none'}  "
               f"· **Keys:** " + ", ".join([f"{k}:{'✓' if v else '✗'}" for k,v in _env_flags.items()]))
# ----------------------------------------------------------------------------


CFG = load_config()
CAT = load_json("config/assets_catalog.json")
TIER_ORDER = CAT["tier_order"]
MIN_RISK_OBJ = min_risk_by_objective()

# Initialize default symbols and fetch prices
symbols = CFG.get("data", {}).get("default_universe", ["SPY"])
prices, prov = get_prices_with_provenance(symbols)

# --- Wiring check (providers → fetch → returns → optimizer) ------------------
with st.expander("🧪 System wiring check (providers → fetch → returns → optimizer)", expanded=False):
    try:
        from core.data_sources.provider_registry import get_ordered_providers
        from core.data_sources.router_smart import fetch_first_available
        from core import portfolio_engine as pe
        from core import risk_metrics as rm

        provs = get_ordered_providers(use_yfinance_fallback=False)
        tickers = [s for s in CFG.get("data", {}).get("default_universe", []) if isinstance(s, str) and s.strip()]
        preview = tickers[: min(6, len(tickers))]

        pulled, failed = {}, []
        for t in preview:
            try:
                df = fetch_first_available(t, provs, period="3y", interval="1d")
                if df is not None and getattr(df, "empty", False) is False:
                    pulled[t] = df
                else:
                    failed.append(t)
            except Exception as e:
                failed.append(t)

        st.write({"preview_tickers": preview, "ok": list(pulled.keys()), "failed": failed})

        if pulled:
            import pandas as _pd
            def _to_close(df):
                for col in ("Adj Close", "Close", "close", "adj_close"):
                    if col in df.columns:
                        return df[col].dropna()
                return df.iloc[:, -1].dropna()

            aligned = _pd.concat({k: _to_close(v) for k, v in pulled.items()}, axis=1).dropna(how="all")
            rets = aligned.pct_change(fill_method=None).dropna(how="all")

            if not rets.empty and rets.shape[1] >= 2:
                try:
                    weights = pe.build_portfolio(rets, CFG)
                    risk = rm.summarize_risk(
                        rets,
                        weights=list(weights.values()) if hasattr(weights, "values") else None,
                        rfr=float(CFG.get("risk", {}).get("risk_free_rate", 0.015)),
                    )
                    st.write({"weights": weights, "risk": risk})
                except Exception as e:
                    st.write(f"Optimizer/risk check failed: {e}")
    except Exception as e:
        st.write(f"Wiring check failed early: {e}")
# ----------------------------------------------------------------------------

# --- Provider attempts (verbose) ---------------------------------------------
with st.expander("🔎 Provider attempts (verbose)", expanded=False):
    try:
        from core.data_sources.provider_registry import get_ordered_providers
        from core.data_sources.router_smart import fetch_first_available

        provs_dbg = get_ordered_providers(
            use_yfinance_fallback=bool(CFG.get("data", {}).get("use_yfinance_fallback", True))
        )
        tickers_dbg = [s for s in CFG.get("data", {}).get("default_universe", []) if isinstance(s, str) and s.strip()]
        preview_dbg = tickers_dbg[: min(6, len(tickers_dbg))]

        logs = {}
        winners = {}
        ok, failed = [], []

        for t in preview_dbg:
            attempts = []
            try:
                df, winner = fetch_first_available(
                    t, provs_dbg, period="3y", interval="1d",
                    return_provider=True, attempts=attempts
                )
                logs[t] = attempts
                winners[t] = winner
                if df is not None and getattr(df, "empty", False) is False:
                    ok.append(t)
                else:
                    failed.append(t)
            except Exception as e:
                logs[t] = attempts + [{"provider":"<exception>", "ms":0, "ok":False, "rows":0, "error":type(e).__name__}]
                winners[t] = None
                failed.append(t)

        st.write({"preview_tickers": preview_dbg, "ok": ok, "failed": failed})
        st.write({"winners": winners})
        st.write({"attempts": logs})
    except Exception as e:
        st.write({"error": f"Provider attempts block failed: {type(e).__name__}: {e}"})
# ----------------------------------------------------------------------------

# --- FRED connectivity check (optional keys) ---------------------------------
with st.expander("🏦 FRED connectivity check", expanded=False):
    try:
        from core.data_sources import fred as _fred
        # Attempt a simple series pull; adjust series code if your fred.py expects different names
        try_codes = ["DGS10", "CPIAUCSL"]
        status = {}
        for code in try_codes:
            try:
                s = _fred.load_series(code) if hasattr(_fred, "load_series") else None
                ok = s is not None and getattr(s, "empty", False) is False
                rows = int(len(s)) if ok else 0
                last = (s.index[-1].date().isoformat() if ok else None) if hasattr(s, "index") and len(s) else None
                status[code] = {"ok": ok, "rows": rows, "last_date": last}
            except Exception as e:
                status[code] = {"ok": False, "error": type(e).__name__}
        st.write(status)
    except Exception as e:
        st.write({"error": f"FRED check not available: {type(e).__name__}: {e}"})

# --- Provider registry status (modules, keys, detect) ------------------------
with st.expander("📦 Provider registry status", expanded=False):
    import os
    detected = []
    details = {}

    # 1) Detected providers from registry (names only)
    try:
        from core.data_sources.provider_registry import get_ordered_providers
        provs = get_ordered_providers(
            use_yfinance_fallback=bool(CFG.get("data", {}).get("use_yfinance_fallback", True))
        )
        detected = [getattr(p, "provider_name", getattr(p, "__name__", "unknown")) for p in (provs or [])]
    except Exception as e:
        details["registry_error"] = f"{type(e).__name__}: {e}"

    # 2) Module-level availability checks (import + attributes)
    def _check_import(mod_path, attrs=None):
        out = {"imported": False, "error": None, "attrs": {}}
        try:
            mod = __import__(mod_path, fromlist=["*"])
            out["imported"] = True
            for a in (attrs or []):
                out["attrs"][a] = hasattr(mod, a)
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {e}"
        return out

    details["stooq"]   = _check_import("core.data_sources.stooq", attrs=["load_daily"])
    details["tiingo"]  = _check_import("core.data_sources.tiingo", attrs=["load_daily", "fetch"])
    # yfinance is third-party, not our module
    try:
        import yfinance as yf
        details["yfinance"] = {"imported": True, "version": getattr(yf, "__version__", None)}
    except Exception as e:
        details["yfinance"] = {"imported": False, "error": f"{type(e).__name__}: {e}"}

    # 3) Environment keys presence (True/False only; never show values)
    details["env"] = {
        "TIINGO_API_KEY_present": bool(os.getenv("TIINGO_API_KEY")),
        "FRED_API_KEY_present": bool(os.getenv("FRED_API_KEY") or os.getenv("FRED_API_TOKEN")),
        "POLYGON_API_KEY_present": bool(
            os.getenv("POLYGON_API_KEY")
            or os.getenv("POLYGON_API_TOKEN")
            or os.getenv("POLYGON_ACCESS_KEY_ID")
            or os.getenv("POLYGON_SECRET_ACCESS_KEY")
        ),
    }

    details["env_aliases"] = {
        "POLYGON_API_KEY": bool(os.getenv("POLYGON_API_KEY")),
        "POLYGON_API_TOKEN": bool(os.getenv("POLYGON_API_TOKEN")),
        "POLYGON_ACCESS_KEY_ID": bool(os.getenv("POLYGON_ACCESS_KEY_ID")),
        "POLYGON_SECRET_ACCESS_KEY": bool(os.getenv("POLYGON_SECRET_ACCESS_KEY")),
    }

    # 4) Summary + details
    st.write({"detected_providers": detected})
    st.write(details)
# ----------------------------------------------------------------------------

# ---------- Presets & sectors ----------
PRESETS = {
    "grow":        ["SPY","QQQ","IWM","EFA","EEM","VNQ","GLD","DBC"],
    "grow_income": ["SPY","IWM","LQD","HYG","VNQ","TLT","MUB","GLD"],
    "income":      ["BIL","SHY","LQD","HYG","MUB","TLT","VNQ"],
    "tax_efficiency": ["MUB","BIL","SHY","SPY","VNQ"],
    "preserve":    ["BIL","SHY","TLT","MUB"]
}
SECTOR_ETFS = {
    "Tech (XLK)": "XLK", "Health (XLV)": "XLV", "Financials (XLF)": "XLF",
    "Industrials (XLI)": "XLI", "Consumer Disc (XLY)": "XLY", "Consumer Staples (XLP)": "XLP",
    "Utilities (XLU)": "XLU", "Materials (XLB)": "XLB", "Real Estate (XLRE)": "XLRE", "Energy (XLE)": "XLE"
}
EXTRAS = {
    "International (EFA/EEM)": ["EFA","EEM"],
    "Commodities (DBC/GSG)": ["DBC","GSG"],
    "REITs (VNQ)": ["VNQ"],
    "T-Bills (BIL/SHY)": ["BIL","SHY"]
}

def tier_index(name: str) -> int:
    return TIER_ORDER.index(name)

def project_lump_and_dca(curve: pd.Series, lump: float = 0.0, monthly: float = 0.0) -> float:
    if curve.empty:
        return 0.0
    growth = float(curve.iloc[-1] / curve.iloc[0])
    final_lump = lump * growth
    final_dca = simulate_contributions(curve, monthly) if monthly > 0 else 0.0
    return final_lump + final_dca

def dedupe_keep_order(symbols):
    seen, out = set(), []
    for s in symbols:
        if s and s not in seen:
            seen.add(s); out.append(s)
    return out

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your Profile")
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
    

    st.markdown("### Presets & Sectors")
    # Preset radio (just for description) + Apply button to prefill pool
    preset_key = objective[1]
    st.caption("Apply a curated Asset Pool for your objective. You can add sectors/extras, then edit manually.")
    col_p1, col_p2 = st.columns([1,1])
    with col_p1:
        apply_preset = st.button("Apply preset", help="Prefill Asset Pool with a vetted mix for your objective.")
    with col_p2:
        clear_pool = st.button("Clear", help="Empty the Asset Pool field.")

    # Sector toggles
    chosen_sectors = st.multiselect("Add sector ETFs (optional)", list(SECTOR_ETFS.keys()), default=[],
                                    help="Fine-tune exposures with SPDR sector ETFs.")
    # Extras toggles
    add_extras = st.multiselect("Add extras (optional)", list(EXTRAS.keys()), default=[],
                                help="Broaden or stabilize the mix with international, commodities, REITs, or T-Bills.")

    # Asset Pool field
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

# ---------- Derived profile ----------
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

# Lump-sum heuristics from net worth (tunable)
lump_baseline = round(networth * 0.02, 2)   # 2%
lump_ambitious = round(networth * 0.05, 2)  # 5%
lump_aggressive = round(networth * 0.10, 2) # 10%

# ---------- Eligibility ----------
df_cat = pd.DataFrame(CAT["assets"])
def tix(name: str) -> int: return TIER_ORDER.index(name)
df_cat["eligible_by_tier"] = df_cat["min_tier"].apply(lambda t: tix(tier) >= tix(t))
df_cat["eligible_by_risk"] = df_cat["min_risk_pct"].apply(lambda mr: risk_pct >= max(mr, min_required))
df_cat["eligible_now"] = df_cat["eligible_by_tier"] & df_cat["eligible_by_risk"]

st.subheader("Eligible Assets (based on your class & risk)")
with st.expander("What does 'eligible' mean?", expanded=False):
    st.write("""
    - **Eligible by tier**: your investor class (Retail, Accredited, etc.) can access the instrument.
    - **Eligible by risk**: your chosen risk level is above both the objective minimum and the asset’s minimum.
    - The optimizer only allocates across assets that are **eligible now**.
    """)

# Attach history length (years) per symbol for transparency
try:
    from core.data_ingestion import get_prices
    from core.data_sources.router_smart import router_fetch_daily_smart
    from core.backtesting import normalize_price_columns, price_series
    _syms = df_cat["symbol"].tolist()
    _px = get_prices(_syms, start="1900-01-01")  # union across providers, longest history
    hist_years = (_px.groupby("ticker")["date"].max() - _px.groupby("ticker")["date"].min()).dt.days / 365.25
    df_cat = df_cat.merge(hist_years.rename("hist_years").reset_index().rename(columns={"ticker":"symbol"}), on="symbol", how="left")
except Exception as _e:
    df_cat["hist_years"] = None

st.dataframe(df_cat[["name","symbol","class","min_tier","min_risk_pct","eligible_now","hist_years"]])


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


# ---------- Run ----------
if run_btn:
    symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
    eligible_symbols = df_cat[df_cat["symbol"].isin(symbols) & df_cat["eligible_now"]]["symbol"].tolist()
    if not eligible_symbols:
        st.error("No eligible tradable symbols in your Asset Pool. Adjust risk or pool.")
    else:
        prices = get_prices(eligible_symbols, start="1900-01-01")
        if prices.empty:
            st.error("No price data returned (source/network). Try fewer/different symbols.")
        else:
            wide = to_wide(prices)
            rets = compute_returns(wide)

            prof = UserProfile(monthly_contribution=budget["baseline_monthly"], horizon_years=horizon, risk_level="moderate")
            rec = recommend(
                rets,
                prof,
                objective=objective[1],
                risk_pct=risk_pct,
                catalog=CAT,
                method=st.session_state.get("opt_method", "hrp"),
            )
            w = pd.Series(rec["weights"]).sort_values(ascending=False)
            metrics = rec["metrics"]
            curve = rec["curve"]

            # --- Diagnostics / Transparency ---
            wvec = pd.Series(w).reindex(rets.columns).fillna(0.0)
            port_ret = rets.fillna(0).dot(wvec)

            cov_pct = data_coverage(rets)
            effN = effective_n_assets(w.to_dict())
            try:
                ci_lo, ci_hi = bootstrap_interval(port_ret, stat="cagr", n=500, alpha=0.10)
            except Exception:
                ci_lo, ci_hi = float("nan"), float("nan")

            # ANOVA across regimes (high-vol vs normal)
            try:
                rolling_vol = port_ret.rolling(21).std()
                thr = rolling_vol.quantile(0.8)
                groups = pd.Series(pd.NA, index=port_ret.index)
                groups.loc[rolling_vol >= thr] = "high_vol"
                groups.loc[rolling_vol <  thr] = "normal"
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
                
                # Create receipts DataFrame
                receipts = pd.DataFrame({
                    'Asset': w.index,
                    'Weight': w.values,
                    'History (years)': df_cat.set_index('symbol')['hist_years'].reindex(w.index).round(1),
                    'Provider': pd.Series(rec.get('providers', {})),
                    'Eligible': df_cat.set_index('symbol')['eligible_now'].reindex(w.index)
                })
                
                # Show weights table 
                st.dataframe(w.to_frame("weight"))
                
                # Download button for receipts
                csv = receipts.to_csv(index=False)
                st.download_button(
                    "Download receipts CSV",
                    csv,
                    "receipts.csv",
                    "text/csv",
                    key='download-receipts'
                )
                
                st.bar_chart(w)
                
                # Add allocation policy note
                st.info(
                    "**Allocation Policy:**  \n"
                    "Core positions must be ≥65% total, providing stability.  \n"
                    "Satellite positions limited to ≤35% total and ≤7% individually for diversification."
                )
            with b:
                st.subheader("Backtest Metrics")
                with st.expander("What do these mean?", expanded=False):
                    st.markdown("""
                    - **CAGR**: annualized growth rate over the backtest period.
                    - **Vol**: annualized volatility (standard deviation of returns).
                    - **Sharpe**: return per unit of volatility (simple backtest estimate).
                    - **MaxDD**: largest peak-to-trough loss.
                    - **N**: number of daily observations (trading days).
                    """)
                st.json(metrics)
                st.subheader("Cumulative Growth (normalized)")

                # Build optional SPY benchmark on same dates
                try:
                    bench_df = get_prices(["SPY"])
                    if not bench_df.empty:
                        from core.preprocessing import to_wide as _to_wide, compute_returns as _compute_returns
                        _w = _to_wide(bench_df)
                        _r = _compute_returns(_w)
                        # align to portfolio rets window if available
                        bench_curve = (1 + _r["SPY"].reindex(curve.index).fillna(0)).cumprod()
                        to_plot = pd.concat({"Portfolio": curve, "SPY (normalized)": bench_curve}, axis=1).dropna()
                    else:
                        to_plot = pd.DataFrame({"Portfolio": curve})
                except Exception:
                    to_plot = pd.DataFrame({"Portfolio": curve})

                st.line_chart(to_plot)

                # Explain normalization + show window & growth
                try:
                    start_dt = curve.index[0].date()
                    end_dt = curve.index[-1].date()
                    growth_pct = (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1.0) * 100.0
                    st.caption(f"Backtest window: {start_dt} → {end_dt}  •  Normalized growth over window: **{growth_pct:.2f}%**")
                except Exception:
                    pass
                with st.expander("What does 'normalized' mean?", expanded=False):
                    st.markdown(
                        "We convert the portfolio into an **index that starts at 1.0** on day one. "
                        "If the index ends at 1.25, that means **+25% cumulative growth** over the shown period. "
                        "<br/><br/>"
                        "**Relative to what?** Relative to **itself** (its own starting value). We also plot an optional **SPY benchmark** for context. "
                        "It’s useful to compare shapes of different portfolios, because they all start at the same baseline (1.0). "
                        "For dollar outcomes, see the **Contribution Paths** section below.",
                        unsafe_allow_html=True
                    )

            # ---- Contribution scenarios table ----
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

            # Align by date for plotting
            paths = pd.concat(
                {"Baseline": path_baseline, "Ambitious": path_ambitious, "Aggressive": path_aggressive},
                axis=1
            ).dropna(how="all")

            st.line_chart(paths)

            # Neater summary table using the final values of each path
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

            # ===== Detailed Receipt & Visualization =====
            # Compute month count and warn if short window
            try:
                n_months = paths.index.to_period('M').nunique()
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

            # Build invested series for the three plans
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
                    years = (n_months or 0) / 12.0 if (n_months is not None) else 0.0
                    # Build cashflows: negative invests on their actual dates, positive final on last date
                    xirr_val = 0.0
                    try:
                        if inv_ser is not None and len(inv_ser) > 0:
                            import pandas as _pd
                            cf = []
                            # incremental contributions: diff of invested series
                            inc = inv_ser.diff().fillna(inv_ser.iloc[0])
                            for dt, amt in inc.items():
                                if amt != 0:
                                    cf.append((dt.to_pydatetime(), -float(amt)))
                            # terminal positive value
                            cf.append((paths.index[-1].to_pydatetime(), float(final_val)))
                            xirr_val = compute_xirr(cf) * 100.0
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

            # ---- Transparency block: show formulas and invested vs final ----
            st.subheader("Transparency — how we compute projections")
            if not paths.empty:
                months = paths.index.to_period('M').nunique()
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

else:
    st.write("Use the sidebar, apply a preset (optionally add sectors/extras), review eligibility, then click **Run Simulation**.")


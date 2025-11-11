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

# ROOT already defined at top of file with proper Path import
# Removed duplicate: ROOT = pathlib.Path(...) 

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
# ---------- Run (after sidebar/profile/eligibility) ----------
if run_btn:
                try:
                    symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
                    eligible_mask = df_cat.index.isin(symbols) & df_cat["eligible_now"].astype(bool)
                    eligible_symbols = list(df_cat.index[eligible_mask])
                    if not eligible_symbols:
                        st.error("No eligible tradable symbols in your Asset Pool. Adjust risk or pool."); st.stop()
                    prices = get_prices(eligible_symbols, start="1900-01-01")
                    if prices.empty:
                        st.error("No price data returned (source/network). Try fewer/different symbols."); st.stop()
                    wide = to_wide(prices)
                    rets = compute_returns(wide)
                    prof = UserProfile(monthly_contribution=budget["baseline_monthly"], horizon_years=horizon, risk_level="moderate")
                    rec = recommend(rets, prof, objective=objective[1], risk_pct=risk_pct, catalog=CAT, method=st.session_state.get("opt_method", "hrp"))
                    w = pd.Series(rec.get("weights", {})).sort_values(ascending=False)
                    metrics = rec.get("metrics", {})
                    curve = rec.get("curve", pd.Series(dtype=float))
                    wvec = pd.Series(w).reindex(rets.columns).fillna(0.0)
                    port_ret = rets.fillna(0).dot(wvec)
                    cov_pct = data_coverage(rets); effN = effective_n_assets(w.to_dict())
                    try:
                        ci_lo, ci_hi = bootstrap_interval(port_ret, stat="cagr", n=300, alpha=0.10)
                    except Exception:
                        ci_lo, ci_hi = float("nan"), float("nan")
                    try:
                        rolling_vol = port_ret.rolling(21).std(); thr = rolling_vol.quantile(0.8)
                        groups = pd.Series(pd.NA, index=port_ret.index); groups.loc[rolling_vol >= thr] = "high_vol"; groups.loc[rolling_vol < thr] = "normal"
                        anova = anova_mean_returns(port_ret, groups)
                    except Exception:
                        anova = {"F": float("nan"), "p": float("nan"), "k": 2}
                    cred = credibility_score(n_obs=int(len(port_ret)), cov=float(cov_pct), effN=float(effN), sharpe_is=float(metrics.get("Sharpe",0.0)), sharpe_oos=None)
                    with st.expander("Methodology & diagnostics", expanded=False):
                        c1,c2,c3 = st.columns(3); c1.metric("Credibility score", f"{cred['score']}/100"); c2.metric("Coverage", f"{cov_pct*100:.1f}%"); c3.metric("Effective N", f"{effN:.2f}")
                        st.write(f"Bootstrap 90% CI (1Y CAGR): {ci_lo*100:.2f}% → {ci_hi*100:.2f}%"); st.write(f"ANOVA: F={anova['F']:.3f}, p={anova['p']:.3f}, groups={anova['k']}")
                    colA,colB = st.columns(2)
                    with colA:
                        st.subheader("Suggested Allocation")
                        try:
                            receipts = pd.DataFrame({
                                'Asset': w.index,
                                'Weight': w.values,
                                'History (years)': df_cat['hist_years'].reindex(w.index.astype(str)).round(1),
                                'Provider': pd.Series(rec.get('providers', {})),
                                'Eligible': df_cat['eligible_now'].reindex(w.index.astype(str))
                            })
                            st.dataframe(w.to_frame("weight"))
                            st.download_button("Download receipts CSV", receipts.to_csv(index=False), "receipts.csv", "text/csv")
                            st.bar_chart(w)
                            st.info("**Allocation Policy:** Core ≥65%. Satellites ≤35% total and ≤7% each.")
                        except Exception as e:
                            st.error(f"Portfolio table build failed: {e}"); import traceback as _tb; st.code("".join(_tb.format_exc())); st.stop()
                    with colB:
                        st.subheader("Backtest Metrics")
                        with st.expander("Metric glossary", expanded=False):
                            st.markdown("""- **CAGR** annualized growth\n- **Vol** annualized volatility\n- **Sharpe** return/volatility\n- **MaxDD** largest peak→trough drawdown\n- **N** observation count""")
                        st.json(metrics)
                        st.subheader("Cumulative Growth (normalized)")
                        try:
                            bench_df = get_prices(["SPY"])
                            if not bench_df.empty:
                                _w = to_wide(bench_df); _r = compute_returns(_w)
                                bench_curve = (1+_r["SPY"].reindex(curve.index).fillna(0)).cumprod()
                                to_plot = pd.concat({"Portfolio": curve, "SPY (normalized)": bench_curve}, axis=1).dropna()
                            else:
                                to_plot = pd.DataFrame({"Portfolio": curve})
                        except Exception:
                            to_plot = pd.DataFrame({"Portfolio": curve})
                        st.line_chart(to_plot)
                        try:
                            start_dt = curve.index[0].date(); end_dt = curve.index[-1].date(); growth_pct = (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1.0)*100.0
                            st.caption(f"Backtest window: {start_dt} → {end_dt} • Normalized growth: **{growth_pct:.2f}%**")
                        except Exception: pass
                    # Contribution projections (MVP)
                    st.subheader("Contribution Plans — Projected Outcome (MVP)")
                    plans = [("Baseline", lump_baseline, budget['baseline_monthly']), ("Ambitious", lump_ambitious, budget['ambitious_monthly']), ("Aggressive", lump_aggressive, budget['aggressive_monthly'])]
                    rows=[]
                    for name,lump,monthly in plans:
                        total = project_lump_and_dca(curve, lump=lump, monthly=monthly)
                        rows.append({"Plan": name, "One-time (lump-sum $)": f"{lump:,.0f}", "Monthly ($)": f"{monthly:,.0f}", "Projected Final ($)": f"{total:,.0f}"})
                    st.table(pd.DataFrame(rows)); st.caption("Projections reuse historical backtest curve — not guarantees.")
                    # DCA paths
                    st.subheader("Contribution Paths (calendar-accurate DCA)")
                    path_baseline = simulate_dca_calendar_series(curve, monthly=budget['baseline_monthly'], lump=lump_baseline)
                    path_ambitious = simulate_dca_calendar_series(curve, monthly=budget['ambitious_monthly'], lump=lump_ambitious)
                    path_aggressive = simulate_dca_calendar_series(curve, monthly=budget['aggressive_monthly'], lump=lump_aggressive)
                    paths = pd.concat({"Baseline": path_baseline, "Ambitious": path_ambitious, "Aggressive": path_aggressive}, axis=1).dropna(how="all")
                    st.line_chart(paths)
                    if not paths.empty:
                        finals = paths.tail(1).T.reset_index(); finals.columns=["Plan","Projected Final ($)"]
                        finals["Projected Final ($)"] = finals["Projected Final ($)"].map(lambda v: f"{v:,.0f}")
                        finals["Monthly ($)"] = [f"{budget['baseline_monthly']:,.0f}", f"{budget['ambitious_monthly']:,.0f}", f"{budget['aggressive_monthly']:,.0f}"]
                        finals["One-time (lump-sum $)"] = [f"{lump_baseline:,.0f}", f"{lump_ambitious:,.0f}", f"{lump_aggressive:,.0f}"]
                        st.table(finals[["Plan","One-time (lump-sum $)","Monthly ($)","Projected Final ($)"]])
                    st.caption("Calendar DCA buys at month-end (nearest trading day); lump sum at start date.")
                    try:
                        n_months = paths.index.to_period('M').nunique();
                        if n_months < 36: st.warning(f"Data window is short ({n_months} months). Results may be noisy.")
                    except Exception: n_months = None
                    def _invested_series(index, monthly, lump):
                        import pandas as _pd
                        s = _pd.Series(0.0, index=index)
                        if len(index)==0: return s
                        s.iloc[0] = float(lump or 0.0)
                        if monthly and monthly>0:
                            month_end_idx = _pd.Series(1, index=index).resample("ME").last().index
                            used=set()
                            for d in month_end_idx:
                                pos = index.get_indexer([d], method="pad")
                                if pos[0]==-1: continue
                                ts = index[pos[0]]
                                if ts in used: continue
                                used.add(ts); s.loc[ts] += monthly
                        return s.cumsum().ffill()
                    inv_baseline = _invested_series(paths.index, budget['baseline_monthly'], lump_baseline) if not paths.empty else None
                    inv_ambitious = _invested_series(paths.index, budget['ambitious_monthly'], lump_ambitious) if not paths.empty else None
                    inv_aggressive = _invested_series(paths.index, budget['aggressive_monthly'], lump_aggressive) if not paths.empty else None
                    try:
                        finals_map = paths.tail(1).iloc[0].to_dict() if not paths.empty else {}
                        plan_specs=[("Baseline", lump_baseline, budget['baseline_monthly'], inv_baseline, "Baseline"), ("Ambitious", lump_ambitious, budget['ambitious_monthly'], inv_ambitious, "Ambitious"), ("Aggressive", lump_aggressive, budget['aggressive_monthly'], inv_aggressive, "Aggressive")]
                        rows_det=[]
                        for name,lump,monthly,inv_ser,key in plan_specs:
                            invested_total = float(inv_ser.iloc[-1]) if inv_ser is not None and len(inv_ser)>0 else float((lump or 0)+(monthly or 0)*(n_months or 0))
                            final_val = float(finals_map.get(key,0.0)); profit = final_val - invested_total; roi = (profit/invested_total)*100.0 if invested_total>0 else 0.0
                            xirr_val=0.0
                            try:
                                if inv_ser is not None and len(inv_ser)>0:
                                    inc = inv_ser.diff().fillna(inv_ser.iloc[0]); cf=[]
                                    for dt,amt in inc.items():
                                        if amt!=0: cf.append((dt.to_pydatetime(), -float(amt)))
                                    cf.append((paths.index[-1].to_pydatetime(), float(final_val)))
                                    xirr_val = compute_xirr(cf)*100.0
                            except Exception: xirr_val=0.0
                            rows_det.append({"Plan":name,"Lump ($)":f"{lump:,.0f}","Monthly ($)":f"{monthly:,.0f}","Months":n_months if n_months is not None else "","Total Invested ($)":f"{invested_total:,.0f}","Final Value ($)":f"{final_val:,.0f}","Profit/Loss ($)":f"{profit:,.0f}","ROI (%)":f"{roi:,.2f}","XIRR (cash-flow annualized, %)":f"{xirr_val:,.2f}"})
                        st.subheader("Contribution Receipt (detailed)"); st.table(pd.DataFrame(rows_det))
                    except Exception as e: st.caption(f"(Could not build detailed receipt: {e})")
                    with st.expander("See per-plan calculation receipts", expanded=False):
                        try:
                            for lbl,lump,monthly,inv_ser,key in [("Baseline", lump_baseline, budget['baseline_monthly'], inv_baseline, "Baseline"),("Ambitious", lump_ambitious, budget['ambitious_monthly'], inv_ambitious, "Ambitious"),("Aggressive", lump_aggressive, budget['aggressive_monthly'], inv_aggressive, "Aggressive")]:
                                if inv_ser is None or paths.empty: continue
                                final_val=float(paths[key].iloc[-1]); invested_total=float(inv_ser.iloc[-1]); profit=final_val-invested_total
                                st.markdown(f"**{lbl}**  \nLump: ${lump:,.0f} at start  \nMonthly: ${monthly:,.0f} × **{n_months}** months  \nInvested total: **${invested_total:,.0f}**  \nFinal value: **${final_val:,.0f}**  \nProfit/Loss: **${profit:,.0f}**")
                        except Exception: pass
                    try:
                        import pandas as _pd, altair as alt
                        chart_df = _pd.DataFrame(index=paths.index)
                        chart_df["Baseline (value)"] = paths["Baseline"]; chart_df["Ambitious (value)"] = paths["Ambitious"]; chart_df["Aggressive (value)"] = paths["Aggressive"]
                        if inv_baseline is not None: chart_df["Baseline (invested)"] = inv_baseline
                        if inv_ambitious is not None: chart_df["Ambitious (invested)"] = inv_ambitious
                        if inv_aggressive is not None: chart_df["Aggressive (invested)"] = inv_aggressive
                        chart_df = chart_df.reset_index().rename(columns={"index":"date"})
                        long = chart_df.melt("date", var_name="series", value_name="value").dropna()
                        st.subheader("Absolute Value Paths vs Invested (calendar-accurate DCA)")
                        line = alt.Chart(long).mark_line().encode(x="date:T", y=alt.Y("value:Q", title="USD"), color="series:N")
                        st.altair_chart(line, use_container_width=True)
                    except Exception as e: st.caption(f"(Altair chart not available: {e})")
                    st.subheader("Transparency — how we compute projections")
                    if not paths.empty:
                        months = paths.index.to_period('M').nunique(); finals_map = paths.tail(1).iloc[0].to_dict()
                        plan_specs=[("Baseline", lump_baseline, budget['baseline_monthly']),("Ambitious", lump_ambitious, budget['ambitious_monthly']),("Aggressive", lump_aggressive, budget['aggressive_monthly'])]
                        rows_t=[]
                        for name,lump,monthly in plan_specs:
                            invested=(lump or 0.0)+(monthly or 0.0)*months; final_val=float(finals_map.get(name,0.0)); profit=final_val-invested; roi=(profit/invested)*100.0 if invested>0 else 0.0
                            rows_t.append({"Plan":name,"Months":months,"Invested total ($)":f"{invested:,.0f}","Projected Final ($)":f"{final_val:,.0f}","Profit / (Loss) ($)":f"{profit:,.0f}","ROI (%)":f"{roi:,.1f}"})
                        st.table(pd.DataFrame(rows_t))
                        with st.expander("Formula details", expanded=False):
                            st.markdown("""**Normalized cumulative growth** index starts 1.0; end 1.25 = +25%.\n**Shares** lump at start + monthly month-end buys.\n**Value** shares × index.\n**Profit** final − invested; ROI% = profit/invested.""")
                except Exception as e:
                    import traceback; st.error(f"Simulation failed: {e}")
                    with st.expander("Traceback (debug)"): st.code(traceback.format_exc())
                    st.stop()
            else:
                st.write("Use the sidebar, apply a preset, review eligibility, then click **Run Simulation**.")
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
    - **Eligible by risk**: your chosen risk level is above both the objective minimum and the asset’s minimum.
    - The optimizer only allocates across assets that are **eligible now**.
    """)

# Attach history length (years) per symbol for transparency
try:
    from core.data_ingestion import get_prices
    _syms = list(df_cat.index)
    _px = get_prices(_syms, start="1900-01-01")  # union across providers, longest history
    hist_years = (_px.groupby("ticker")["date"].max() - _px.groupby("ticker")["date"].min()).dt.days / 365.25
    hist_years = hist_years.rename("hist_years")
    df_cat = df_cat.join(hist_years, how="left")
except Exception as _e:
    df_cat["hist_years"] = None

# Display table with explicit symbol column from index
st.dataframe(
    df_cat.assign(symbol=df_cat.index)[["name","symbol","class","min_tier","min_risk_pct","eligible_now","hist_years"]]
)


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
    try:
    symbols = [s.strip().upper() for s in pool.split(",") if s.strip()]
    # Determine eligible symbols using symbol-indexed df_cat
    eligible_mask = df_cat.index.isin(symbols) & df_cat["eligible_now"].astype(bool)
    eligible_symbols = list(df_cat.index[eligible_mask])
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
                    # Use already symbol-indexed df_cat; align via string index
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
                    st.json(metrics)
                    st.subheader("Cumulative Growth (normalized)")
                    try:
                        bench_df = get_prices(["SPY"])
                        if not bench_df.empty:
                            _w = to_wide(bench_df)
                            _r = compute_returns(_w)
                            bench_curve = (1 + _r["SPY"].reindex(curve.index).fillna(0)).cumprod()
                            to_plot = pd.concat({"Portfolio": curve, "SPY (normalized)": bench_curve}, axis=1).dropna()
                        else:
                            to_plot = pd.DataFrame({"Portfolio": curve})
                    except Exception:
                        to_plot = pd.DataFrame({"Portfolio": curve})
                    st.line_chart(to_plot)
                    try:
                        start_dt = curve.index[0].date(); end_dt = curve.index[-1].date()
                        growth_pct = (float(curve.iloc[-1]) / float(curve.iloc[0]) - 1.0) * 100.0
                        st.caption(f"Backtest window: {start_dt} → {end_dt} • Normalized growth: **{growth_pct:.2f}%**")
                    except Exception:
                        pass
        
    except Exception as e:
        import traceback
        st.error(f"Simulation failed: {e}")
        with st.expander("Traceback (debug)"):
            st.code(traceback.format_exc())

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

    except Exception as e:
        import traceback
        st.error(f"Simulation failed: {e}")
        with st.expander("Traceback (debug)"):
            st.code(traceback.format_exc())
else:
    st.write("Use the sidebar, apply a preset (optionally add sectors/extras), review eligibility, then click **Run Simulation**.")


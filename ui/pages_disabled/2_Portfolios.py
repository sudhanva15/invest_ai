import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from core.data_ingestion import get_prices
from core.preprocessing import compute_returns
from core.recommendation_engine import DEFAULT_OBJECTIVES, ObjectiveConfig, generate_candidates
from core.universe_validate import build_validated_universe
from ui.state import (
    ensure_session,
    render_session_summary,
    set_universe_snapshot,
    set_sigma_band_candidates,
    set_chosen_candidate,
)

st.set_page_config(page_title="Invest AI — Portfolios", page_icon="📈", layout="wide")

ensure_session()
st.title("📈 Portfolio Suggestions")
render_session_summary(expanded=False)

# Universe summary (load or build)
snap_path = Path("data/outputs/universe_snapshot.json")
metrics_path = Path("data/outputs/universe_metrics.json")
valid_syms = []
if snap_path.exists():
    try:
        payload = json.loads(snap_path.read_text())
        total = payload.get("universe_size", 0)
        v = payload.get("valid_count", 0)
        d = payload.get("dropped_count", 0)
        valid_syms = payload.get("valid_symbols", [])
        try:
            set_universe_snapshot(payload)
        except Exception:
            pass
        st.success(f"Universe: {total} tickers (Valid: {v}, Dropped: {d})")
        # Extra summary lines: average history (years) and missing data percent over valid set
        try:
            recs = payload.get("records", {})
            if recs and valid_syms:
                import numpy as _np
                hist_vals = [_r.get("history_years", 0.0) for s, _r in recs.items() if s in valid_syms]
                miss_vals = [_r.get("missing_pct", 0.0) for s, _r in recs.items() if s in valid_syms]
                if hist_vals:
                    st.caption(f"Average history: {_np.nanmean(hist_vals):.1f} years")
                if miss_vals:
                    st.caption(f"Missing data: {_np.nanmean(miss_vals):.1f}%")
        except Exception:
            pass
        # If metrics file exists, show a compact line for avg vol and avg corr
        try:
            if metrics_path.exists():
                m = json.loads(metrics_path.read_text())
                av = m.get("avg_volatility")
                ac = m.get("avg_correlation")
                line_bits = []
                if isinstance(av, (int,float)):
                    line_bits.append(f"Avg vol: {av*100:.1f}%")
                if isinstance(ac, (int,float)):
                    line_bits.append(f"Avg corr: {ac:.2f}")
                if line_bits:
                    st.caption(" • ".join(line_bits))
        except Exception:
            pass
    except Exception:
        valid_syms = []
else:
    with st.spinner("Validating universe (first run)…"):
        valid_syms, snap_path = build_validated_universe()
    st.success(f"Universe snapshot saved: {snap_path}")

if not valid_syms:
    st.warning("No validated symbols available. Check data providers or run diagnostics.")

# Build candidate set using validated universe (cap size for speed)
max_syms = int(st.sidebar.number_input("Max symbols to use", min_value=8, max_value=80, value=24))
use_syms = valid_syms[:max_syms] if valid_syms else []
if use_syms:
    try:
        prices = get_prices(use_syms, start="1900-01-01")
        rets = compute_returns(prices)
    except Exception as e:
        st.error(f"Failed to load returns: {e}")
        rets = pd.DataFrame()
else:
    rets = pd.DataFrame()

# Objective selection and candidate generation
obj_key = st.selectbox("Objective", list(DEFAULT_OBJECTIVES.keys()), index=0)
obj_cfg = DEFAULT_OBJECTIVES.get(obj_key)
if isinstance(obj_cfg, dict):
    obj_cfg = ObjectiveConfig(**obj_cfg)

# No need to set enforce_validated_universe: generate_candidates will use the
# validated universe snapshot by default when present.

st.subheader("Candidates")
if rets.empty or len(rets.columns) < 3:
    st.info("Not enough return series to build candidates.")
else:
    try:
        if obj_cfg is None:
            obj_cfg = ObjectiveConfig(name=obj_key)
        cands = generate_candidates(
            returns=rets,
            objective_cfg=obj_cfg,
            catalog=json.loads((ROOT/"config"/"assets_catalog.json").read_text()),
            n_candidates=int(st.sidebar.number_input("# Candidates", min_value=5, max_value=10, value=8)),
        )
    except Exception as e:
        st.error(f"Candidate generation failed: {e}")
        cands = []

    if not cands:
        st.warning("No candidates produced.")
    else:
        rows = []
        curves = {}
        last_1y = rets.tail(252)
        for cand in cands:
            w = pd.Series(cand.get("weights", {})).reindex(rets.columns).fillna(0.0)
            port_1y = (last_1y * w).sum(axis=1)
            m = cand.get("metrics", {})
            rows.append({
                "Name": cand.get("name"),
                "CAGR": m.get("CAGR"),
                "Vol": m.get("Vol") or m.get("Volatility"),
                "Sharpe": m.get("Sharpe"),
                "MaxDD": m.get("MaxDD"),
                "Assets": int((w > 0).sum()),
            })
            curves[cand.get("name")] = (1 + (rets * w).sum(axis=1)).cumprod().rename(cand.get("name"))
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Mini chart grid (2 per row)
        st.subheader("Equity curves (normalized)")
        cols = st.columns(2)
        for i, (name, curve) in enumerate(curves.items()):
            with cols[i % 2]:
                try:
                    st.line_chart(curve.resample("W").last() if hasattr(curve, "resample") else curve)
                except Exception:
                    st.line_chart(curve)

        # Risk-based match using Profile score if available
        try:
            from core.recommendation_engine import select_candidates_for_risk_score, pick_portfolio_from_slider
        except Exception:
            select_candidates_for_risk_score = pick_portfolio_from_slider = None
        rscore = st.session_state.get("risk_score")
        if rscore is not None and select_candidates_for_risk_score and pick_portfolio_from_slider:
            st.subheader("Risk match")
            st.caption("Using your Profile page risk score to filter portfolios to a volatility band and pick along the return spectrum.")
            filtered = select_candidates_for_risk_score(cands, float(rscore))
            try:
                set_sigma_band_candidates([dict(fc) for fc in (filtered or [])])
            except Exception:
                pass
            if not filtered:
                st.info("No candidates fit your current risk band. Try adding more symbols or widening constraints.")
            else:
                # Show filtered list
                f_rows = []
                for fc in filtered:
                    fm = fc.get("metrics", {})
                    f_rows.append({
                        "Name": fc.get("name"),
                        "CAGR": fm.get("CAGR"),
                        "Vol": fm.get("Vol") or fm.get("Volatility"),
                        "Sharpe": fm.get("Sharpe"),
                    })
                st.dataframe(pd.DataFrame(f_rows), use_container_width=True)
                slider_val = st.slider("Choose within your band (safer ↔ higher growth)", 0.0, 1.0, value=0.5, step=0.01)
                picked = pick_portfolio_from_slider(filtered, slider_val)
                if picked:
                    st.success(f"Selected: {picked.get('name')}")
                    pw = pd.Series(picked.get("weights", {})).sort_values(ascending=False)
                    st.dataframe(pw.to_frame("weight"))
                    try:
                        set_chosen_candidate(dict(picked))
                    except Exception:
                        pass
                    try:
                        pcurve = curves.get(picked.get("name"))
                        if pcurve is None:
                            wvec = pd.Series(picked.get("weights", {})).reindex(rets.columns).fillna(0.0)
                            pcurve = (1 + (rets * wvec).sum(axis=1)).cumprod()
                        st.line_chart(pcurve.resample("W").last() if hasattr(pcurve, "resample") else pcurve)
                    except Exception:
                        pass

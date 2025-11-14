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
from core.utils.metrics import rolling_metrics
from ui.state import ensure_session, render_session_summary, get_chosen_candidate

st.set_page_config(page_title="Invest AI — Diagnostics", page_icon="🛠️", layout="wide")
st.title("🛠️ Diagnostics")

ensure_session()
render_session_summary(expanded=False)

snap_path = ROOT/"data/outputs/universe_snapshot.json"
metrics_path = ROOT/"data/outputs/universe_metrics.json"

col1, col2 = st.columns(2)
with col1:
    st.subheader("Universe Snapshot")
    if snap_path.exists():
        payload = json.loads(snap_path.read_text())
        st.write({
            "universe_size": payload.get("universe_size"),
            "valid": payload.get("valid_count"),
            "dropped": payload.get("dropped_count"),
        })
        st.dataframe(
            st.session_state.get("_universe_records_df")
            if "_universe_records_df" in st.session_state else None
        )
        if st.button("Show detailed records"):
            import pandas as pd
            df = pd.DataFrame.from_dict(payload.get("records", {}), orient="index")
            st.session_state["_universe_records_df"] = df
            st.dataframe(df)
    else:
        st.info("No snapshot found. Run universe scan from dev tools.")

with col2:
    st.subheader("Universe Metrics")
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text()))
    else:
        st.caption("metrics file not found.")

st.divider()
st.caption("Raw logs and receipts will appear here in future iterations.")

st.divider()
st.subheader("Rolling metrics")

# Load validated symbols if available for selection
valid_symbols = []
if snap_path.exists():
    try:
        payload = json.loads(snap_path.read_text())
        valid_symbols = payload.get("valid_symbols", []) or []
    except Exception:
        valid_symbols = []

default_sym = "SPY" if "SPY" in valid_symbols else (valid_symbols[0] if valid_symbols else "SPY")
colA, colB = st.columns([2,1])
with colB:
    sym = st.selectbox("Symbol", options=[default_sym] + [s for s in valid_symbols if s != default_sym] if valid_symbols else [default_sym])
    window = int(st.number_input("Window (days)", min_value=60, max_value=756, value=252, step=21))

with colA:
    try:
        px = get_prices([sym], start="2000-01-01")
        rets = compute_returns(px)
        ser = rets.iloc[:, 0] if isinstance(rets, pd.DataFrame) and rets.shape[1] >= 1 else pd.Series(dtype=float)
        if ser.empty:
            st.info("No returns available for selected symbol.")
        else:
            roll = rolling_metrics(ser, window=window)
            if not roll.empty:
                st.line_chart(roll[["vol", "sharpe"]].dropna().rename(columns={"vol": "Volatility", "sharpe": "Sharpe"}))
                st.caption("Annualized volatility and Sharpe over rolling window")
                st.area_chart(roll[["maxdd"]].dropna().rename(columns={"maxdd": "Max Drawdown"}))
            else:
                st.info("Rolling metrics could not be computed for current selection.")
    except Exception as e:
        st.error(f"Failed to compute rolling metrics: {e}")

st.divider()
st.subheader("Chosen portfolio (session)")
chosen = get_chosen_candidate()
if chosen:
    st.write({"name": chosen.get("name"), "optimizer": chosen.get("optimizer"), "sat_cap": chosen.get("sat_cap")})
    try:
        import pandas as pd
        st.dataframe(pd.Series(chosen.get("weights", {})).sort_values(ascending=False).to_frame("weight"))
    except Exception:
        pass
else:
    st.caption("No portfolio chosen yet.")

import sys
from pathlib import Path
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

st.set_page_config(page_title="Invest AI — Macro", page_icon="📈", layout="wide")
st.title("📉 Macro Dashboard")

try:
    from core.data_sources.fred import load_series
    import datetime as _dt
    series = {
        "CPI (YoY)": load_series("CPIAUCSL").pct_change(12)*100,
        "Fed Funds Rate": load_series("FEDFUNDS"),
        "10Y Treasury": load_series("DGS10"),
        "Industrial Production": load_series("INDPRO"),
        "Unemployment": load_series("UNRATE"),
    }
    cols = st.columns(len(series))
    now = _dt.datetime.utcnow()
    for (name, s), col in zip(series.items(), cols):
        s = s.dropna()
        last = float(s.iloc[-1]) if not s.empty else float("nan")
        col.metric(name, f"{last:.2f}")
        try:
            col.line_chart(s.tail(365))
        except Exception:
            col.line_chart(s)
except Exception as e:
    st.error(f"Macro load failed: {e}")

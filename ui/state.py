from __future__ import annotations
from typing import Any, Dict, List, Optional

import streamlit as st


# Centralized session keys used across pages
KEY_PROFILE = "_profile"
KEY_RISK_SCORE = "risk_score"
KEY_Q_ANSWERS = "_risk_answers"
KEY_UNIVERSE_SNAPSHOT = "_universe_snapshot"
KEY_SIGMA_BAND = "_sigma_band_candidates"
KEY_CHOSEN = "_chosen_candidate"


def ensure_session() -> None:
    """Initialize expected session keys with safe defaults."""
    ss = st.session_state
    ss.setdefault(KEY_PROFILE, {
        "income": None,
        "net_worth": None,
        "horizon_years": None,
        "objective": None,
        "risk_pct": None,
    })
    ss.setdefault(KEY_RISK_SCORE, None)
    ss.setdefault(KEY_Q_ANSWERS, {})
    ss.setdefault(KEY_UNIVERSE_SNAPSHOT, None)
    ss.setdefault(KEY_SIGMA_BAND, [])
    ss.setdefault(KEY_CHOSEN, None)


def set_profile(income: Optional[float], net_worth: Optional[float], horizon_years: Optional[int], objective: Optional[str], risk_pct: Optional[float]) -> None:
    ensure_session()
    st.session_state[KEY_PROFILE] = {
        "income": income,
        "net_worth": net_worth,
        "horizon_years": horizon_years,
        "objective": objective,
        "risk_pct": risk_pct,
    }


def get_profile() -> Dict[str, Any]:
    ensure_session()
    return dict(st.session_state.get(KEY_PROFILE) or {})


def set_questionnaire_answers(answers: Dict[str, float], risk_score: Optional[float]) -> None:
    ensure_session()
    st.session_state[KEY_Q_ANSWERS] = dict(answers or {})
    st.session_state[KEY_RISK_SCORE] = None if risk_score is None else float(risk_score)


def set_universe_snapshot(snapshot_payload: Dict[str, Any]) -> None:
    ensure_session()
    st.session_state[KEY_UNIVERSE_SNAPSHOT] = dict(snapshot_payload or {})


def set_sigma_band_candidates(cands: List[Dict[str, Any]]) -> None:
    """Persist a trimmed version of candidates for the sigma-band (risk-match) list."""
    ensure_session()
    trimmed: List[Dict[str, Any]] = []
    for c in cands or []:
        trimmed.append({
            "name": c.get("name"),
            "metrics": c.get("metrics", {}),
            "weights": c.get("weights", {}),
            "optimizer": c.get("optimizer"),
            "sat_cap": c.get("sat_cap"),
        })
    st.session_state[KEY_SIGMA_BAND] = trimmed


def get_sigma_band_candidates() -> List[Dict[str, Any]]:
    ensure_session()
    return list(st.session_state.get(KEY_SIGMA_BAND) or [])


def set_chosen_candidate(candidate: Dict[str, Any]) -> None:
    ensure_session()
    if candidate is None:
        st.session_state[KEY_CHOSEN] = None
        return
    st.session_state[KEY_CHOSEN] = {
        "name": candidate.get("name"),
        "metrics": candidate.get("metrics", {}),
        "weights": candidate.get("weights", {}),
        "optimizer": candidate.get("optimizer"),
        "sat_cap": candidate.get("sat_cap"),
    }


def get_chosen_candidate() -> Optional[Dict[str, Any]]:
    ensure_session()
    val = st.session_state.get(KEY_CHOSEN)
    return dict(val) if isinstance(val, dict) else None


def render_session_summary(expanded: bool = False) -> None:
    """Small sidebar/session block to show current risk score and chosen candidate."""
    ensure_session()
    with st.sidebar.expander("Session", expanded=expanded):
        prof = get_profile()
        rs = st.session_state.get(KEY_RISK_SCORE)
        chosen = get_chosen_candidate()
        st.caption(
            f"Objective: {prof.get('objective') or 'n/a'} • Risk slider: {prof.get('risk_pct') if prof.get('risk_pct') is not None else 'n/a'}%"
        )
        st.caption(f"Risk score: {rs if rs is not None else 'n/a'}")
        if chosen:
            st.caption(f"Chosen: {chosen.get('name')}")
            try:
                import pandas as pd
                w = pd.Series(chosen.get("weights", {})).sort_values(ascending=False)
                st.dataframe(w.head(10).to_frame("w"), use_container_width=True)
            except Exception:
                pass

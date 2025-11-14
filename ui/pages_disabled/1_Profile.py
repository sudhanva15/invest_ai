import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from ui.state import ensure_session, set_questionnaire_answers, render_session_summary

st.set_page_config(page_title="Invest AI — Profile", page_icon="👤", layout="wide")

st.title("👤 Profile & Questionnaire")
st.caption("Tell us about your horizon and preferences. We’ll keep your answers in session state.")

ensure_session()
render_session_summary(expanded=False)

# MCQ questionnaire
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

q1 = st.radio(
    "How long until you'll need most of this money?",
    ["0–3 years", "3–7 years", "7–15 years", "15+ years"],
    index=st.session_state.get("_idx_q1", 2),
    key="risk_q1_time_horizon_choice",
    horizontal=True,
)
q2 = st.radio(
    "How comfortable are you with temporary losses?",
    ["Very low", "Low", "Medium", "High", "Very high"],
    index=st.session_state.get("_idx_q2", 2),
    key="risk_q2_loss_tolerance_choice",
    horizontal=True,
)
q3 = st.radio(
    "If your portfolio dropped 20% soon after investing, what would you do?",
    ["Sell everything immediately", "Sell some to reduce risk", "Hold and wait", "Hold and might buy more", "Definitely buy more (opportunity)"],
    index=st.session_state.get("_idx_q3", 2),
    key="risk_q3_reaction20_choice",
)
q4 = st.radio(
    "How stable is your income over the next 3–5 years?",
    ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
    index=st.session_state.get("_idx_q4", 2),
    key="risk_q4_income_stability_choice",
    horizontal=True,
)
q5 = st.radio(
    "How dependent are you on this money for living expenses?",
    ["Critical for living expenses", "Important but not critical", "Helpful but have other income", "Nice-to-have growth money"],
    index=st.session_state.get("_idx_q5", 1),
    key="risk_q5_dependence_choice",
)
q6 = st.radio(
    "How experienced are you with investing and markets?",
    ["Beginner (first time)", "Some experience (< 3 years)", "Experienced (3-10 years)", "Advanced (10+ years)"],
    index=st.session_state.get("_idx_q6", 1),
    key="risk_q6_experience_choice",
)
q7 = st.radio(
    "Do you have an emergency fund and basic insurance?",
    ["No emergency fund or insurance", "Small emergency fund (< 3 months)", "Moderate safety net (3-6 months)", "Strong safety net (6+ months)"],
    index=st.session_state.get("_idx_q7", 2),
    key="risk_q7_safety_net_choice",
)
q8 = st.radio(
    "What's the main goal for this money?",
    ["Capital preservation (safety first)", "Income generation (steady returns)", "Balanced growth (moderate risk)", "Aggressive growth (max returns)"],
    index=st.session_state.get("_idx_q8", 2),
    key="risk_q8_goal_type_choice",
)

answers = {
    "q1_time_horizon": map_time_horizon_choice(str(q1)),
    "q2_loss_tolerance": map_loss_tolerance_choice(str(q2)),
    "q3_reaction_20_drop": map_reaction_20_drop_choice(str(q3)),
    "q4_income_stability": map_income_stability_choice(str(q4)),
    "q5_dependence_on_money": map_dependence_on_money_choice(str(q5)),
    "q6_investing_experience": map_investing_experience_choice(str(q6)),
    "q7_safety_net": map_safety_net_choice(str(q7)),
    "q8_goal_type": map_goal_type_choice(str(q8)),
}

try:
    rscore = float(compute_risk_score(answers))
except Exception:
    rscore = float("nan")

st.session_state["risk_score"] = rscore
try:
    set_questionnaire_answers(answers, rscore)
except Exception:
    pass
st.metric("Composite Risk Score", f"{rscore:.1f}" if rscore == rscore else "n/a")

st.divider()
st.button("Generate My Portfolio Suggestions", key="btn_generate_suggestions")

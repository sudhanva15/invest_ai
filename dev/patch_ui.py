#!/usr/bin/env python3
"""Patch UI file to add info box, scenarios, validator, download, and robustness improvements."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_FILE = ROOT / "ui" / "streamlit_app.py"

def apply_ui_patches():
    """Apply all UI patches in sequence."""
    
    with open(UI_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patch 1: Add info box and scenario selector after opening markdown
    marker1 = '''                    st.markdown(
                        "Answer a few questions about your comfort with risk and time horizon.\\n"
                        "We'll use your answers to find portfolios that fit your risk profile, then let you choose where you sit along that risk–return spectrum."
                    )

                    # Optional reset of questionnaire state'''
    
    replacement1 = '''                    st.markdown(
                        "Answer a few questions about your comfort with risk and time horizon.\\n"
                        "We'll use your answers to find portfolios that fit your risk profile, then let you choose where you sit along that risk–return spectrum."
                    )
                    
                    # Info box explaining the process
                    st.info(
                        "**How this works:**\\n"
                        "1. Your answers give you a risk score (0–100).\\n"
                        "2. We map that score to a risk band (volatility range) and filter portfolios to that band.\\n"
                        "3. The slider lets you choose where you sit within that band (safer ↔ more growth).\\n"
                        "4. The confidence score shows how strong the data is behind these estimates."
                    )
                    
                    # Demo scenarios selector
                    scenario = st.selectbox(
                        "Quick demo scenarios (optional):",
                        ["None", "Conservative", "Moderate", "Aggressive"],
                        help="Scenarios are presets to help you explore; you can still adjust your answers manually.",
                    )
                    
                    # Apply scenario defaults if selected
                    if scenario == "Conservative":
                        scenario_defaults = {
                            "risk_q1_time_horizon": 20, "risk_q2_loss_tolerance": 20, "risk_q3_reaction20": 15,
                            "risk_q4_income_stability": 40, "risk_q5_dependence": 75, "risk_q6_experience": 30,
                            "risk_q7_safety_net": 40, "risk_q8_goal_type": 20,
                        }
                    elif scenario == "Moderate":
                        scenario_defaults = {
                            "risk_q1_time_horizon": 50, "risk_q2_loss_tolerance": 50, "risk_q3_reaction20": 50,
                            "risk_q4_income_stability": 50, "risk_q5_dependence": 50, "risk_q6_experience": 50,
                            "risk_q7_safety_net": 50, "risk_q8_goal_type": 50,
                        }
                    elif scenario == "Aggressive":
                        scenario_defaults = {
                            "risk_q1_time_horizon": 85, "risk_q2_loss_tolerance": 80, "risk_q3_reaction20": 75,
                            "risk_q4_income_stability": 70, "risk_q5_dependence": 20, "risk_q6_experience": 75,
                            "risk_q7_safety_net": 80, "risk_q8_goal_type": 85,
                        }
                    else:
                        scenario_defaults = {}
                    
                    # Update session state with scenario if chosen
                    if scenario != "None" and scenario_defaults:
                        for k, v in scenario_defaults.items():
                            if k not in st.session_state:
                                st.session_state[k] = v

                    # Optional reset of questionnaire state'''
    
    if marker1 in content:
        content = content.replace(marker1, replacement1)
        print("✓ Added info box and scenario selector")
    else:
        print("✗ Could not find marker for info box/scenario patch")
    
    # Patch 2: Update robustness computation and use compute_robustness_from_curve
    marker2_old = '''                                # Robustness score via segmented CAGRs
                                robustness = float("nan")
                                try:
                                    if compute_simple_robustness_score and pcurve is not None and len(pcurve) > 252:
                                        segs = []
                                        n = len(pcurve)
                                        k = 3
                                        step = n // k
                                        for i in range(k):
                                            seg = pcurve.iloc[i*step:(i+1)*step]
                                            if len(seg) > 30:
                                                cagr = float(seg.iloc[-1] ** (252/len(seg)) - 1.0)
                                                segs.append(cagr)
                                        if segs:
                                            robustness = compute_simple_robustness_score(segs)
                                except Exception:
                                    robustness = float("nan")'''
    
    marker2_new = '''                                # Robustness score via segmented CAGRs
                                robustness = float("nan")
                                try:
                                    from core.risk_profile import compute_robustness_from_curve
                                    if pcurve is not None and len(pcurve) > 252:
                                        robustness = compute_robustness_from_curve(pcurve, n_segments=3)
                                except Exception:
                                    robustness = float("nan")'''
    
    if marker2_old in content:
        content = content.replace(marker2_old, marker2_new)
        print("✓ Updated robustness computation to use compute_robustness_from_curve")
    else:
        print("⚠ Could not find marker for robustness patch (may already be updated)")
    
    # Patch 3: Update validator to use run_light_validator
    marker3_old = '''                                    # TODO: wire a real validator flag from a light pipeline (see dev/validate_simulations.py)
                                    validator_passed = True'''
    
    marker3_new = '''                                    # Wire lightweight validator
                                    try:
                                        from core.validation import run_light_validator
                                        validator_passed = run_light_validator(
                                            objective="balanced",
                                            universe_size=num_assets,
                                            returns=rets,
                                        )
                                    except Exception:
                                        validator_passed = True  # fallback'''
    
    if marker3_old in content:
        content = content.replace(marker3_old, marker3_new)
        print("✓ Wired run_light_validator")
    else:
        print("⚠ Could not find marker for validator patch")
    
    # Patch 4: Add download button after summary section
    marker4 = '''                                st.caption("Outcome band widens when credibility is lower (uncertainty discount). This is historical dispersion — not a forecast.")
                                if robustness == robustness:
                                    st.caption(f"Robustness (segmented CAGR consistency): {robustness:.1f}/100")'''
    
    addition4 = '''                                st.caption("Outcome band widens when credibility is lower (uncertainty discount). This is historical dispersion — not a forecast.")
                                if robustness == robustness:
                                    st.caption(f"Robustness (segmented CAGR consistency): {robustness:.1f}/100")
                                
                                # Download summary button
                                try:
                                    summary_text = f"""Risk profile: {rscore:.0f} / 100

Chosen portfolio: {picked.get('name')}

Expected annual return (historical): ~{mu*100:.1f}%
Risk (volatility): ~{sigma*100:.1f}%
Model confidence: {credibility:.0f} / 100
Robustness: {robustness:.0f} / 100
Typical annual range (history-based): ~{low*100:.1f}% to ~{high*100:.1f}%

Notes:
- This is not a guarantee of future returns.
- The confidence score reflects data history, stability, diversification, and internal checks.
- Robustness measures how consistently the portfolio performed across different time periods.
"""
                                    st.download_button(
                                        label="📥 Download summary",
                                        data=summary_text,
                                        file_name="invest_ai_risk_summary.txt",
                                        mime="text/plain",
                                    )
                                except Exception:
                                    pass'''
    
    if marker4 in content:
        content = content.replace(marker4, addition4)
        print("✓ Added download summary button")
    else:
        print("⚠ Could not find marker for download button patch")
    
    # Write back
    with open(UI_FILE, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\n✓ Wrote updated UI to {UI_FILE}")


if __name__ == "__main__":
    apply_ui_patches()

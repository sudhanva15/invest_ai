#!/usr/bin/env python3
"""
Patch script to replace slider-based questionnaire with MCQ-style questions.
"""
import sys
import re

ui_file = "ui/streamlit_app.py"

print(f"Reading {ui_file}...")
with open(ui_file, 'r') as f:
    content = f.read()

# Pattern to match the 8 sliders section
old_questions_pattern = r'(                    # 8-question simple sliders.*?)\n(                    # Compute composite risk score)'

new_questions = r'''                    # 8-question MCQ-style questionnaire
                    q1_choice = st.radio(
                        "**Q1:** How long until you'll need most of this money?",
                        ["0–3 years", "3–7 years", "7–15 years", "15+ years"],
                        index=2,
                        key="risk_q1_time_horizon_choice",
                        help="Longer timelines can handle more ups and downs. Short timelines usually need safer portfolios.",
                        horizontal=True
                    )
                    
                    q2_choice = st.radio(
                        "**Q2:** How comfortable are you with temporary losses?",
                        ["Very low", "Low", "Medium", "High", "Very high"],
                        index=2,
                        key="risk_q2_loss_tolerance_choice",
                        help="Higher tolerance means you're OK seeing bigger short-term drops.",
                        horizontal=True
                    )
                    
                    q3_choice = st.radio(
                        "**Q3:** If your portfolio dropped 20% soon after investing, what would you do?",
                        ["Sell everything immediately", "Sell some to reduce risk", "Hold and wait", "Hold and might buy more", "Definitely buy more (opportunity)"],
                        index=2,
                        key="risk_q3_reaction20_choice",
                        help="This reflects your likely behavior under stress, not the textbook answer."
                    )
                    
                    q4_choice = st.radio(
                        "**Q4:** How stable is your income over the next 3–5 years?",
                        ["Very unstable", "Unstable", "Moderate", "Stable", "Very stable"],
                        index=2,
                        key="risk_q4_income_stability_choice",
                        help="More stable income makes it easier to ride out market swings.",
                        horizontal=True
                    )
                    
                    q5_choice = st.radio(
                        "**Q5:** How dependent are you on this money for living expenses?",
                        ["Critical for living expenses", "Important but not critical", "Helpful but have other income", "Nice-to-have growth money"],
                        index=2,
                        key="risk_q5_dependence_choice",
                        help="If this money is critical for essentials, your risk level should be lower."
                    )
                    
                    q6_choice = st.radio(
                        "**Q6:** How experienced are you with investing and markets?",
                        ["Beginner (first time)", "Some experience (< 3 years)", "Experienced (3-10 years)", "Advanced (10+ years)"],
                        index=1,
                        key="risk_q6_experience_choice",
                        help="More experience usually means less chance of panic-selling at the worst time."
                    )
                    
                    q7_choice = st.radio(
                        "**Q7:** Do you have an emergency fund and basic insurance?",
                        ["No emergency fund or insurance", "Small emergency fund (< 3 months)", "Moderate safety net (3-6 months)", "Strong safety net (6+ months)"],
                        index=2,
                        key="risk_q7_safety_net_choice",
                        help="A stronger safety net lets your investments be a bit more adventurous if you want."
                    )
                    
                    q8_choice = st.radio(
                        "**Q8:** What's the main goal for this money?",
                        ["Capital preservation (safety first)", "Income generation (steady returns)", "Balanced growth (moderate risk)", "Aggressive growth (max returns)"],
                        index=2,
                        key="risk_q8_goal_type_choice",
                        help="Capital preservation, steady income, or long-term growth all point to different risk levels."
                    )

'''

# Find the sliders section
match = re.search(r'# 8-question simple sliders.*?q8 = st\.slider\(.*?\)', content, re.DOTALL)
if not match:
    print("❌ Could not find sliders section")
    sys.exit(1)

print(f"✓ Found sliders section at position {match.start()}-{match.end()}")

# Replace
content_new = content[:match.start()] + new_questions.lstrip() + content[match.end():]

# Now update the answers dict and add mapping imports
# Find the answers dict
answers_old = r'''                    answers = {
                        "q1_time_horizon": q1,
                        "q2_loss_tolerance": q2,
                        "q3_reaction_20_drop": q3,
                        "q4_income_stability": q4,
                        "q5_dependence_on_money": q5,
                        "q6_investing_experience": q6,
                        "q7_safety_net": q7,
                        "q8_goal_type": q8,
                    }'''

answers_new = r'''                    # Map MCQ choices to numeric scores
                    from core.risk_profile import (
                        map_time_horizon_choice,
                        map_loss_tolerance_choice,
                        map_reaction_20_drop_choice,
                        map_income_stability_choice,
                        map_dependence_on_money_choice,
                        map_investing_experience_choice,
                        map_safety_net_choice,
                        map_goal_type_choice,
                    )
                    
                    answers = {
                        "q1_time_horizon": map_time_horizon_choice(q1_choice),
                        "q2_loss_tolerance": map_loss_tolerance_choice(q2_choice),
                        "q3_reaction_20_drop": map_reaction_20_drop_choice(q3_choice),
                        "q4_income_stability": map_income_stability_choice(q4_choice),
                        "q5_dependence_on_money": map_dependence_on_money_choice(q5_choice),
                        "q6_investing_experience": map_investing_experience_choice(q6_choice),
                        "q7_safety_net": map_safety_net_choice(q7_choice),
                        "q8_goal_type": map_goal_type_choice(q8_choice),
                    }'''

content_new = content_new.replace(answers_old, answers_new)

print("Writing updated file...")
with open(ui_file, 'w') as f:
    f.write(content_new)

print("✅ Successfully patched UI questionnaire to use MCQ-style questions")
print("\nNext steps:")
print("1. Update scenario defaults to use choice strings instead of numeric scores")
print("2. Test the UI")

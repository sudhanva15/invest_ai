"""
Phase 3 UI Components: Risk-aligned explanations and fallback transparency.

Provides:
- Risk score → CAGR/volatility explanation cards
- Fallback stage indicators
- Per-ticker receipts with provenance
- Allocation pie charts with asset class grouping
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Any


def display_risk_explanation(risk_profile) -> None:
    """
    Display risk score → expected CAGR/volatility explanation in beginner-friendly format.
    
    Args:
        risk_profile: RiskProfileResult with true_risk, cagr_min, cagr_target, vol_target
    """
    st.markdown("### 🎯 Your Risk Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Risk Score",
            f"{risk_profile.true_risk:.0f}/100",
            help="Your overall risk tolerance based on questionnaire, financial situation, and slider"
        )
    
    with col2:
        cagr_range = f"{risk_profile.cagr_min*100:.1f}%-{risk_profile.cagr_target*100:.1f}%"
        st.metric(
            "Expected Growth",
            cagr_range,
            help="Annual return range aligned with your risk score"
        )
    
    with col3:
        vol_range = f"{risk_profile.vol_min*100:.1f}%-{risk_profile.vol_max*100:.1f}%"
        st.metric(
            "Volatility Range",
            vol_range,
            help="Expected portfolio fluctuation range"
        )
    
    # Explanation text
    if risk_profile.true_risk <= 30:
        risk_label = "Conservative"
        explanation = "Your portfolio prioritizes **capital preservation** with lower expected returns and minimal volatility."
    elif risk_profile.true_risk <= 60:
        risk_label = "Moderate"
        explanation = "Your portfolio balances **growth and stability** with moderate returns and acceptable fluctuations."
    else:
        risk_label = "Aggressive"
        explanation = "Your portfolio targets **maximum growth** with higher expected returns but larger swings."
    
    st.info(f"**{risk_label} Profile**: {explanation}")


def display_fallback_indicator(portfolio: dict) -> None:
    """
    Display fallback stage indicator with explanation.
    
    Args:
        portfolio: Portfolio dict with 'fallback', 'fallback_level', 'hard_fallback' flags
    """
    is_fallback = portfolio.get("fallback", False)
    fallback_level = portfolio.get("fallback_level")
    hard_fallback = portfolio.get("hard_fallback", False)
    
    if not is_fallback:
        st.success("✅ **Primary Portfolio** - Passed all strict filters")
        return
    
    if hard_fallback or fallback_level == 4:
        st.error("⚠️ **Emergency Fallback (Stage 4)** - Equal-weight portfolio created due to insufficient viable candidates")
        with st.expander("Why did this happen?"):
            st.markdown("""
            **Stage 4 Emergency Fallback** activates when:
            - No portfolios passed strict filters (Stage 1)
            - No portfolios passed relaxed filters (Stage 2)
            - No portfolios met compositional criteria (Stage 3)
            
            This typically means:
            - Insufficient historical data for assets
            - Extremely restrictive risk profile settings
            - Market conditions causing all strategies to fail filters
            
            **Recommendation**: Consider adjusting your risk tolerance or expanding the asset universe.
            """)
    elif fallback_level == 3:
        st.warning("⚡ **Compositional Portfolio (Stage 3)** - Promoted from soft violations")
        with st.expander("What does this mean?"):
            st.markdown("""
            **Stage 3 Compositional** portfolios:
            - Met basic viability criteria (positive CAGR, positive Sharpe, drawdown >-80%)
            - Failed some strict thresholds but show reasonable performance
            - Represent best available options when optimal filters are too restrictive
            
            This is **normal** for moderate-to-high risk profiles with limited asset universes.
            """)
    elif fallback_level == 2:
        st.info("🔄 **Relaxed Portfolio (Stage 2)** - Passed risk-adaptive filters")
        with st.expander("What does this mean?"):
            st.markdown("""
            **Stage 2 Relaxed** portfolios:
            - Failed strict uniform filters
            - Passed risk-adaptive thresholds tailored to your risk score
            - Use dynamic CAGR, Sharpe, and drawdown requirements
            
            This indicates filters were appropriately loosened for your risk profile.
            """)
    else:
        st.warning(f"🔀 **Fallback Portfolio (Level {fallback_level})** - Did not pass all strict filters")


def display_allocation_pie_chart(portfolio: dict, catalog: dict = None) -> None:
    """
    Display allocation as pie chart with asset class grouping.
    
    Args:
        portfolio: Portfolio dict with 'weights' key
        catalog: Assets catalog for classification (optional)
    """
    weights = portfolio.get("weights", {})
    if not weights:
        return
    
    # Build asset class mapping
    symbols = list(weights.keys())
    asset_classes = {}
    
    if catalog:
        # Use catalog for classification
        assets_list = catalog.get("assets", [])
        cat_map = {a.get("symbol", "").upper(): a.get("asset_class", "Unknown") for a in assets_list if isinstance(a, dict)}
        for sym in symbols:
            asset_classes[sym] = cat_map.get(sym.upper(), "Unknown")
    else:
        # Simple heuristic
        for sym in symbols:
            if sym.upper() in {"SPY", "VTI", "QQQ", "DIA", "IWM", "EFA", "EEM"}:
                asset_classes[sym] = "Equity"
            elif sym.upper() in {"TLT", "IEF", "BND", "LQD", "HYG", "MUB", "AGG", "TIP"}:
                asset_classes[sym] = "Bonds"
            elif sym.upper() in {"BIL", "SHY"}:
                asset_classes[sym] = "Cash"
            elif sym.upper() in {"GLD", "DBC", "GSG"}:
                asset_classes[sym] = "Commodities"
            elif sym.upper() in {"VNQ"}:
                asset_classes[sym] = "Real Estate"
            else:
                asset_classes[sym] = "Other"
    
    # Create DataFrame
    df = pd.DataFrame({
        "Symbol": symbols,
        "Weight": [weights[s] * 100 for s in symbols],
        "Asset Class": [asset_classes[s] for s in symbols]
    })
    
    # Sort by weight descending
    df = df.sort_values("Weight", ascending=False)
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=df["Symbol"],
        values=df["Weight"],
        text=df["Asset Class"],
        hovertemplate="<b>%{label}</b><br>" +
                      "Weight: %{value:.1f}%<br>" +
                      "Class: %{text}<br>" +
                      "<extra></extra>",
        textposition='auto',
        textinfo='label+percent',
        marker=dict(
            line=dict(color='white', width=2)
        )
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        showlegend=True,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show weights table
    with st.expander("📋 Detailed Weights"):
        st.dataframe(df, use_container_width=True, hide_index=True)


def display_per_ticker_receipts(
    portfolio: dict,
    asset_receipts: pd.DataFrame = None,
    provenance: dict = None
) -> None:
    """
    Display per-ticker receipt with data provenance, metrics, and filters.
    
    Args:
        portfolio: Portfolio dict with 'weights' and 'risk_contrib'
        asset_receipts: DataFrame from build_recommendations() with asset filter results
        provenance: Dict from get_prices_with_provenance() mapping symbol → provider
    """
    st.markdown("### 🧾 Per-Ticker Receipts")
    
    weights = portfolio.get("weights", {})
    risk_contrib = portfolio.get("risk_contrib", {})
    
    if not weights:
        st.info("No ticker data available")
        return
    
    for symbol in sorted(weights.keys(), key=lambda s: weights[s], reverse=True):
        weight = weights[symbol]
        
        with st.expander(f"**{symbol}** - {weight*100:.1f}% allocation"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Allocation", f"{weight*100:.1f}%")
            
            with col2:
                rc = risk_contrib.get(symbol, 0.0)
                if isinstance(rc, (int, float)):
                    st.metric("Risk Contribution", f"{rc*100:.2f}%")
                else:
                    st.metric("Risk Contribution", "N/A")
            
            with col3:
                if provenance and symbol in provenance:
                    provider = provenance[symbol]
                    st.metric("Data Source", provider)
                else:
                    st.metric("Data Source", "Unknown")
            
            # Asset filter results
            if asset_receipts is not None and not asset_receipts.empty:
                asset_row = asset_receipts[asset_receipts["symbol"] == symbol]
                if not asset_row.empty:
                    row = asset_row.iloc[0]
                    
                    st.markdown("**Asset Metrics:**")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        if "sharpe" in row and pd.notna(row["sharpe"]):
                            st.write(f"Sharpe: {row['sharpe']:.2f}")
                        if "cagr" in row and pd.notna(row["cagr"]):
                            st.write(f"CAGR: {row['cagr']*100:.1f}%")
                    
                    with metric_col2:
                        if "volatility" in row and pd.notna(row["volatility"]):
                            st.write(f"Volatility: {row['volatility']*100:.1f}%")
                        if "max_drawdown" in row and pd.notna(row["max_drawdown"]):
                            st.write(f"Max DD: {row['max_drawdown']*100:.1f}%")
                    
                    with metric_col3:
                        if "years_history" in row and pd.notna(row["years_history"]):
                            st.write(f"History: {row['years_history']:.1f} years")
                        if "passed" in row:
                            status = "✅ Passed" if row["passed"] else "❌ Failed"
                            st.write(f"Filter: {status}")
                    
                    # Show fail reason if failed
                    if "passed" in row and not row["passed"] and "fail_reason" in row:
                        st.caption(f"⚠️ {row['fail_reason']}")


def display_debug_panel(result: dict, cfg: dict) -> None:
    """
    Display debug panel with filter thresholds, fallback stages, and candidate stats.
    
    Args:
        result: Result dict from build_recommendations()
        cfg: Config dict with multifactor settings
    """
    with st.expander("🔬 Debug: Filter Pipeline", expanded=False):
        st.markdown("**Recommendation Pipeline Statistics**")
        
        all_candidates = result.get("all_candidates", [])
        recommended = result.get("recommended", [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Generated", len(all_candidates))
        
        with col2:
            passed = sum(1 for c in all_candidates if c.get("passed_filters"))
            st.metric("Passed Strict", passed)
        
        with col3:
            st.metric("Recommended", len(recommended))
        
        # Fallback stage breakdown
        st.markdown("**Fallback Stage Breakdown:**")
        
        stage_counts = {}
        for c in recommended:
            if c.get("hard_fallback") or c.get("fallback_level") == 4:
                stage_counts[4] = stage_counts.get(4, 0) + 1
            elif c.get("fallback_level") == 3:
                stage_counts[3] = stage_counts.get(3, 0) + 1
            elif c.get("fallback_level") == 2:
                stage_counts[2] = stage_counts.get(2, 0) + 1
            elif c.get("passed_filters"):
                stage_counts[1] = stage_counts.get(1, 0) + 1
            else:
                stage_counts.setdefault("unknown", 0)
                stage_counts["unknown"] += 1
        
        for stage in [1, 2, 3, 4, "unknown"]:
            if stage in stage_counts:
                label = f"Stage {stage}" if isinstance(stage, int) else "Unknown"
                st.write(f"- {label}: {stage_counts[stage]} portfolio(s)")
        
        # Filter thresholds
        st.markdown("**Active Filter Thresholds:**")
        mf_cfg = cfg.get("multifactor", {})
        
        threshold_data = {
            "Filter": [
                "Min Portfolio Sharpe",
                "Max Drawdown",
                "Max Risk Contribution",
                "Min Diversification",
                "Min Holdings",
                "Volatility Soft Factor"
            ],
            "Threshold": [
                f"{mf_cfg.get('min_portfolio_sharpe', 0.3):.2f}",
                f"{mf_cfg.get('max_portfolio_drawdown', -0.50):.1%}",
                f"{mf_cfg.get('max_risk_contribution', 0.40):.1%}",
                f"{mf_cfg.get('min_diversification_ratio', 1.2):.2f}",
                str(mf_cfg.get('min_holdings', 3)),
                f"{mf_cfg.get('vol_soft_lower_factor', 0.6):.1%}"
            ]
        }
        
        st.dataframe(pd.DataFrame(threshold_data), use_container_width=True, hide_index=True)
        
        # Show top failures
        if all_candidates:
            st.markdown("**Top Failure Reasons:**")
            fail_reasons = {}
            for c in all_candidates:
                if not c.get("passed_filters"):
                    reason = c.get("fail_reason", "unknown")
                    fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            
            if fail_reasons:
                fail_df = pd.DataFrame([
                    {"Reason": k, "Count": v}
                    for k, v in sorted(fail_reasons.items(), key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(fail_df.head(10), use_container_width=True, hide_index=True)


def apply_dark_theme():
    """Apply dark theme styling via custom CSS."""
    st.markdown("""
    <style>
        /* Dark theme overrides */
        .stApp {
            background-color: #0E1117;
        }
        
        .stMetric {
            background-color: #1E2126;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #2E3238;
        }
        
        .stExpander {
            background-color: #1E2126;
            border: 1px solid #2E3238;
        }
        
        .stAlert {
            background-color: #1E2126;
            border-left: 4px solid #FF4B4B;
        }
        
        /* Info boxes */
        .stInfo {
            background-color: #1E2126;
            border-left: 4px solid #4B9EFF;
        }
        
        .stSuccess {
            background-color: #1E2126;
            border-left: 4px solid #00CC66;
        }
        
        .stWarning {
            background-color: #1E2126;
            border-left: 4px solid #FFA500;
        }
        
        /* DataFrames */
        .dataframe {
            background-color: #1E2126 !important;
            color: #FAFAFA !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #FAFAFA;
        }
    </style>
    """, unsafe_allow_html=True)

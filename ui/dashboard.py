import streamlit as st
import pandas as pd
from datetime import datetime
from core.data_ingestion import get_prices_with_provenance
from core.utils.receipts import build_receipts
from core.data_sources.fred import load_series
from core.portfolio.simulation import cumprod_from_returns, quick_metrics

# Core/Satellite configuration
CORE = ["VTI", "VXUS", "TLT", "BIL"]
SATS = ["MTUM", "QUAL", "SIZE", "VLUE"]

from core.recommendation_engine import recommend, UserProfile
from typing import Dict, List, Optional

def show_allocation(weights: Dict[str, float], prices_df: pd.DataFrame) -> None:
    """Display portfolio allocation with metrics"""
    if not weights:
        st.warning("No allocation computed.")
        return
        
    # Show weights
    st.subheader("Portfolio Allocation")
    df_weights = pd.DataFrame(
        {"Weight": weights}
    ).sort_values("Weight", ascending=False)
    
    # Format as percentages
    df_weights["Weight"] = df_weights["Weight"].map("{:.1%}".format)
    st.table(df_weights)
    
    # Show contribution breakdown
    core_weight = sum(weights.get(s, 0) for s in CORE)
    sat_weight = sum(weights.get(s, 0) for s in SATS)
    
    cols = st.columns(2)
    cols[0].metric("Core Allocation", f"{core_weight:.1%}")
    cols[1].metric("Satellite Allocation", f"{sat_weight:.1%}")

def main():
    st.title("Portfolio Dashboard V3")

    # Tabs for different views
    tab_main, tab_macro, tab_debug = st.tabs(["Portfolio", "Macro", "Debug"])

    try:
        # Main portfolio view tab
        with tab_main:
            # Allocation mode selector
            alloc_mode = st.selectbox("Allocation Mode", ["Core", "Core + Satellites"])
            
            # Show available universes
            st.caption(f"**Core Assets:** {', '.join(CORE)}")
            if alloc_mode == "Core + Satellites":
                st.caption(f"**Satellite Assets:** {', '.join(SATS)}")
            
            # Determine universe based on mode
            universe = CORE.copy()
            if alloc_mode == "Core + Satellites":
                universe.extend(s for s in SATS if s not in universe)

            # Get prices with provenance info
            with st.spinner("Loading price data..."):
                try:
                    prices_df, prov = get_prices_with_provenance(universe)
                    if len(prices_df) == 0:
                        st.error("Could not load price data. Please check your internet connection or data sources.")
                        return
                except Exception as e:
                    st.error(f"Error loading price data: {str(e)}")
                    return

            # Build portfolio
            with st.spinner("Computing optimal allocation..."):
                try:
                    rets = prices_df.pct_change().dropna()
                    
                    # Get weights through optimization
                    rec = recommend(
                        returns=rets,
                        profile=UserProfile(monthly_contribution=1000, horizon_years=10, risk_level="moderate"),
                        method="hrp"  # Using HRP for stability
                    )
                    
                    # Post-process weights with constraints
                    from core.portfolio.simulation import apply_weight_constraints
                    weights = apply_weight_constraints(
                        rec["weights"],
                        core_symbols=CORE,
                        satellite_symbols=SATS if alloc_mode == "Core + Satellites" else None,
                        core_min=0.65,
                        satellite_max=0.35,
                        single_max=0.07
                    )
                    
                    # Compute portfolio curve
                    from core.portfolio.simulation import compute_portfolio_curve, quick_metrics
                    ptf_curve = compute_portfolio_curve(prices_df, weights)
                    spy_curve = compute_portfolio_curve(prices_df, {"SPY": 1.0})
                    
                    # Show allocation
                    st.subheader("Portfolio Allocation")
                    col1, col2 = st.columns([2,1])
                    
                    with col1:
                        # Display weights table
                        df_weights = pd.DataFrame(
                            {"Weight": weights}
                        ).sort_values("Weight", ascending=False)
                        df_weights["Weight"] = df_weights["Weight"].map("{:.1%}".format)
                        st.table(df_weights)
                        
                    with col2:
                        # Show core/satellite breakdown
                        core_weight = sum(weights.get(s, 0) for s in CORE)
                        sat_weight = sum(weights.get(s, 0) for s in SATS)
                        st.metric("Core Allocation", f"{core_weight:.1%}")
                        if alloc_mode == "Core + Satellites":
                            st.metric("Satellite Allocation", f"{sat_weight:.1%}")
                    
                    # Performance metrics
                    st.subheader("Performance vs SPY")
                    for yrs in (1, 5):
                        m_ptf = quick_metrics(ptf_curve, yrs)
                        m_spy = quick_metrics(spy_curve, yrs)
                        cols = st.columns(3)
                        cols[0].metric(f"HRP CAGR ({yrs}Y)", f"{m_ptf['CAGR']:.1%}")
                        cols[1].metric(f"SPY CAGR ({yrs}Y)", f"{m_spy['CAGR']:.1%}")
                        cols[2].metric(f"HRP Sharpe ({yrs}Y)", f"{m_ptf['Sharpe']:.2f}")
                    
                    # Plot cumulative curves
                    st.line_chart(pd.DataFrame({
                        "Portfolio": ptf_curve,
                        "SPY": spy_curve
                    }))
                    
                except Exception as e:
                    st.error(f"Error computing allocation: {str(e)}")
                    return
        
        # Macro tab
        with tab_macro:
            st.subheader("Macro Indicators")
            macro_desc = {
                "DGS10": "10Y Treasury Yield",
                "T10Y2Y": "Yield Curve Spread (10Y-2Y)",
                "CPIAUCSL": "CPI (All Urban Consumers)"
            }
            
            from core.data_sources.fred import load_series
            for series_id, desc in macro_desc.items():
                st.write(f"**{desc}** ({series_id})")
                series = load_series(series_id)
                if len(series) > 0:
                    df = pd.DataFrame({desc: series})
                    st.line_chart(df)
                    st.caption(f"Latest: {series.iloc[-1]:.2f}% ({series.index[-1].strftime('%Y-%m-%d')})")
                else:
                    st.warning(f"No data available for {series_id}")
            
            st.caption(f"Data refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Debug tab
        with tab_debug:
            with st.expander("Data receipts (per ticker)"):
                from core.utils.receipts import build_receipts
                receipts = build_receipts(universe, prices_df)
                st.dataframe(receipts, use_container_width=True)
                    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        raise

        # Portfolio metrics
        ptf_curve = cumprod_from_returns((prices_df.pct_change().fillna(0) @ weights))
        spy_curve = cumprod_from_returns(prices_df["SPY"].pct_change())

        st.subheader("Performance Metrics")
        for yrs in (1, 5):
            m_ptf = quick_metrics(ptf_curve, yrs)
            m_spy = quick_metrics(spy_curve, yrs)
            cols = st.columns(3)
            cols[0].metric(f"HRP CAGR ({yrs}Y)", f"{m_ptf['CAGR']:.2%}")
            cols[1].metric(f"SPY CAGR ({yrs}Y)", f"{m_spy['CAGR']:.2%}")
            cols[2].metric(f"HRP Sharpe ({yrs}Y)", f"{m_ptf['Sharpe']:.2f}")

    # Macro data tab
    with tab_macro:
        st.subheader("Macro Indicators")
        macro_desc = {
            "DGS10": "10Y Treasury Yield",
            "T10Y2Y": "Yield Curve Spread (10Y-2Y)",
            "CPIAUCSL": "CPI (All Urban Consumers)"
        }
        
        # Load and display macro data with descriptions
        for series_id, desc in macro_desc.items():
            st.write(f"**{desc}** ({series_id})")
            series = load_series(series_id)
            if len(series) > 0:
                df = pd.DataFrame({desc: series})
                st.line_chart(df)
                st.caption(f"Latest: {series.iloc[-1]:.2f}% ({series.index[-1].strftime('%Y-%m-%d')})")
            else:
                st.warning(f"No data available for {series_id}")
        
        st.caption(f"Data refreshed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with st.expander("About these indicators", expanded=False):
            st.markdown("""
            - **10Y Treasury Yield**: Benchmark interest rate, key indicator of economic outlook
            - **Yield Curve Spread**: Difference between 10Y and 2Y yields; negative can signal recession
            - **CPI**: Consumer Price Index, measures inflation in consumer goods and services
            
            Data source: Federal Reserve Economic Data (FRED)
            """)

    # Debug tab with receipts
    with tab_debug:
        with st.expander("Data receipts (per ticker)"):
            receipts = build_receipts(universe, prices_df)
            st.dataframe(receipts, use_container_width=True)

def post_process_weights(weights: dict, mode: str = "Core") -> dict:
    """Apply Core-Satellite constraints to weights"""
    # Cap individual weights at 7%
    weights = {k: min(v, 0.07) for k, v in weights.items()}
    
    if mode == "Core + Satellites":
        # Cap satellite allocation at 35%
        sat_sum = sum(weights.get(s, 0) for s in SATS)
        if sat_sum > 0.35:
            scale = 0.35 / sat_sum
            weights = {
                k: (v * scale if k in SATS else v)
                for k, v in weights.items()
            }
    
    # Ensure Core minimum is maintained
    core_sum = sum(weights.get(c, 0) for c in CORE)
    if core_sum < 0.65:
        needed = 0.65 - core_sum
        # Scale up core proportionally
        if core_sum > 0:
            scale = (core_sum + needed) / core_sum
            weights.update({
                c: weights.get(c, 0) * scale 
                for c in CORE
            })
    
    # Renormalize to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    
    return weights

if __name__ == "__main__":
    main()
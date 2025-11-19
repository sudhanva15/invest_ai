"""
Portfolio Display Components for Phase 3 UI

This module provides reusable components for displaying portfolio recommendations,
metrics cards, allocation visualizations, DCA projections, and receipts.
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from typing import Dict, List, Any

# ---------------- Asset Classification Helpers -----------------
def _build_asset_class_map(symbols: List[str], catalog: dict) -> Dict[str, str]:
    """Return symbol -> asset_class using priority:
    1. universe_records (session_state)
    2. catalog.asset_class (assets_catalog.json entries)
    3. derived from catalog.class prefix heuristics
    4. 'Unknown'

    catalog format expected: {"assets": [...]}
    Each asset entry may contain: symbol, asset_class, class
    """
    rec_map = st.session_state.get("universe_records", {}) or {}
    assets_list = []
    try:
        assets_list = (catalog or {}).get("assets", []) if isinstance(catalog, dict) else []
    except Exception:
        assets_list = []
    cat_by_sym = {str(a.get("symbol", "")).upper(): a for a in assets_list if isinstance(a, dict)}

    def classify(sym: str) -> str:
        su = str(sym).upper()
        # 1. universe snapshot record
        r = rec_map.get(su) or rec_map.get(sym)
        if r is not None:
            try:
                if isinstance(r, dict):
                    ac = r.get("asset_class")
                else:
                    ac = getattr(r, "asset_class", None)
                if ac:
                    return str(ac)
            except Exception:
                pass
        # 2. catalog asset_class
        a = cat_by_sym.get(su)
        if a:
            ac = a.get("asset_class")
            if ac:
                return str(ac)
            # 3. derived from 'class'
            cls = str(a.get("class", ""))
            if cls.startswith("equity"):
                return "equity"
            if cls.startswith("bond") or cls in {"high_yield", "munis", "treasury_long", "tbills"}:
                return "bond"
            if cls in {"cash", "tbills"}:
                return "cash"
            if cls in {"commodities", "gold"}:
                return "commodity"
            if cls in {"reit"}:
                return "reit"
        # 4. heuristics by symbol (coarse)
        if su in {"SPY","VTI","QQQ","DIA","IWM","EFA","EEM","VEA","VXUS"}:
            return "equity"
        if su in {"TLT","IEF","BND","LQD","HYG","MUB","AGG","TIP"}:
            return "bond"
        if su in {"BIL","SHY"}:
            return "cash"
        if su in {"GLD","DBC","GSG"}:
            return "commodity"
        if su in {"VNQ"}:
            return "reit"
        return "Unknown"

    return {sym: classify(sym) for sym in symbols}


def display_metrics_cards(portfolio: dict, beginner_mode: bool = True):
    """
    Display portfolio metrics as cards in a grid layout.
    
    Args:
        portfolio: Portfolio dict with 'metrics' key containing CAGR, Vol, Sharpe, MaxDD, etc.
        beginner_mode: Show explanations if True
    """
    metrics = portfolio.get("metrics", {})
    
    st.subheader("📊 Portfolio Metrics")
    
    # Row 1: Returns and Risk
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cagr = metrics.get("cagr", 0.0)
        st.metric(
            "CAGR",
            f"{cagr*100:.2f}%",
            help="Compound Annual Growth Rate - the average yearly return" if beginner_mode else None
        )
    
    with col2:
        vol = metrics.get("volatility", 0.0)
        st.metric(
            "Volatility",
            f"{vol*100:.2f}%",
            help="Annualized standard deviation - how much the portfolio bounces around" if beginner_mode else None
        )
    
    with col3:
        sharpe = metrics.get("sharpe", 0.0)
        st.metric(
            "Sharpe Ratio",
            f"{sharpe:.2f}",
            help="Risk-adjusted return - higher is better. Above 1.0 is good, above 2.0 is excellent" if beginner_mode else None
        )
    
    with col4:
        max_dd = metrics.get("max_drawdown", 0.0)
        st.metric(
            "Max Drawdown",
            f"{max_dd*100:.2f}%",
            help="Worst peak-to-trough decline - this is how much you could have lost" if beginner_mode else None
        )
    
    # Row 2: Diversification and Holdings
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        div_ratio = metrics.get("diversification_ratio", 0.0)
        st.metric(
            "Diversification",
            f"{div_ratio:.2f}",
            help="Ratio of weighted volatilities to portfolio vol - higher means better diversification" if beginner_mode else None
        )
    
    with col6:
        num_holdings = metrics.get("num_holdings", 0)
        st.metric(
            "Holdings",
            str(num_holdings),
            help="Number of assets in the portfolio" if beginner_mode else None
        )
    
    with col7:
        total_return = metrics.get("total_return", 0.0)
        st.metric(
            "Total Return",
            f"{total_return*100:.2f}%",
            help="Cumulative return over the full backtest period" if beginner_mode else None
        )
    
    with col8:
        # Compute credibility score if possible
        # This would need additional data about history, provider quality, etc.
        # For now, show as placeholder
        credibility = 75.0  # Placeholder
        st.metric(
            "Credibility",
            f"{credibility:.0f}%",
            help="Data quality score based on history length and source reliability" if beginner_mode else None
        )


def display_allocation_chart(weights: dict, catalog: dict, view: str = "ticker"):
    """Display allocation either by ticker or aggregated by asset class.

    view: 'ticker' | 'asset_class'
    """
    st.subheader("🥧 Allocation")
    if not weights:
        st.caption("No allocation data available")
        return

    # Normalize weights just in case
    total_w = sum(float(w) for w in weights.values()) or 1.0
    norm_weights = {k: float(v) / total_w for k, v in weights.items() if float(v) > 0}

    if view == "ticker":
        labels = list(norm_weights.keys())
        sizes = [v * 100 for v in norm_weights.values()]
        fig = px.pie(values=sizes, names=labels, title="Allocation by Ticker", hole=0.0)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        df = pd.DataFrame({"Ticker": labels, "Weight (%)": sizes}).set_index("Ticker")
        st.dataframe(df.sort_values("Weight (%)", ascending=False), use_container_width=True)
        return

    # Asset class view
    class_map = _build_asset_class_map(list(norm_weights.keys()), catalog)
    agg: Dict[str, float] = {}
    for sym, w in norm_weights.items():
        cls = class_map.get(sym, "Unknown")
        agg[cls] = agg.get(cls, 0.0) + w
    # Remove tiny buckets
    agg = {k: v for k, v in agg.items() if v >= 0.001}
    if not agg:
        st.caption("All asset classes unknown; switch back to ticker view.")
        return
    labels = list(agg.keys())
    sizes = [v * 100 for v in agg.values()]
    fig = px.pie(values=sizes, names=labels, title="Allocation by Asset Class", hole=0.0)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)
    df = pd.DataFrame({"Asset Class": labels, "Weight (%)": sizes}).set_index("Asset Class")
    st.dataframe(df.sort_values("Weight (%)", ascending=False), use_container_width=True)


def display_holdings_table(weights: dict, returns: pd.DataFrame):
    """
    Display detailed holdings table with weights and performance metrics.
    
    Args:
        weights: Dict of {symbol: weight}
        returns: DataFrame of daily returns for computing metrics
    """
    st.subheader("📋 Holdings Detail")
    
    holdings_data = []
    for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        if weight < 0.001:
            continue
        
        # Compute individual asset metrics
        if symbol in returns.columns:
            asset_returns = returns[symbol].dropna()
            if len(asset_returns) >= 252:
                cagr = (1 + asset_returns).prod() ** (252 / len(asset_returns)) - 1
                vol = asset_returns.std() * np.sqrt(252)
                sharpe = cagr / vol if vol > 0 else 0.0
            else:
                cagr, vol, sharpe = np.nan, np.nan, np.nan
        else:
            cagr, vol, sharpe = np.nan, np.nan, np.nan
        
        holdings_data.append({
            "Symbol": symbol,
            "Weight (%)": weight * 100,
            "CAGR (%)": cagr * 100 if not np.isnan(cagr) else np.nan,
            "Volatility (%)": vol * 100 if not np.isnan(vol) else np.nan,
            "Sharpe": sharpe if not np.isnan(sharpe) else np.nan,
        })
    
    if holdings_data:
        holdings_df = pd.DataFrame(holdings_data)
        st.dataframe(
            holdings_df.set_index("Symbol").style.format({
                "Weight (%)": "{:.2f}",
                "CAGR (%)": "{:.2f}",
                "Volatility (%)": "{:.2f}",
                "Sharpe": "{:.2f}",
            }),
            use_container_width=True
        )
    else:
        st.caption("No holdings data available")


def compute_dca_projections(
    initial_amount: float,
    monthly_contribution: float,
    years: int,
    expected_return: float,
    volatility: float,
    scenarios: List[str] = ["Baseline", "Ambitious", "Aggressive"],
    scenario_multipliers: List[float] = [1.0, 1.5, 2.0]
) -> Dict[str, pd.Series]:
    """
    Compute DCA (Dollar Cost Averaging) projection scenarios.
    
    Args:
        initial_amount: Starting investment
        monthly_contribution: Monthly DCA amount
        years: Projection horizon
        expected_return: Annual expected return (from portfolio CAGR)
        volatility: Annual volatility
        scenarios: List of scenario names
        scenario_multipliers: Multipliers for monthly contribution in each scenario
    
    Returns:
        Dict of {scenario_name: Series of balances over time}
    """
    months = years * 12
    results = {}
    
    for scenario, multiplier in zip(scenarios, scenario_multipliers):
        contribution = monthly_contribution * multiplier
        monthly_return = expected_return / 12
        monthly_vol = volatility / np.sqrt(12)
        
        balances = [initial_amount]
        for month in range(1, months + 1):
            # Add contribution
            balance = balances[-1] + contribution
            # Apply monthly return with some randomness
            return_this_month = monthly_return + np.random.normal(0, monthly_vol)
            balance *= (1 + return_this_month)
            balances.append(balance)
        
        # Create time index
        dates = pd.date_range(start="2024-01-01", periods=len(balances), freq="MS")
        results[scenario] = pd.Series(balances, index=dates)
    
    return results


def display_dca_simulation(
    initial_amount: float,
    monthly_contribution: float,
    years: int,
    portfolio: dict,
    beginner_mode: bool = True
):
    """
    Display DCA projection simulation with multiple scenarios.
    
    Args:
        initial_amount: Starting investment
        monthly_contribution: Monthly DCA amount
        years: Projection horizon
        portfolio: Portfolio dict with metrics
        beginner_mode: Show explanations if True
    """
    st.subheader("💰 Investment Projection (DCA)")
    
    if beginner_mode:
        st.markdown("""
        **What is this?** This shows how your investment could grow if you:
        - Start with your initial amount
        - Add money every month (Dollar Cost Averaging)
        - Earn returns similar to the portfolio's historical performance
        
        ⚠️ **Important:** This is NOT a guarantee. Real markets are unpredictable.
        """)
    
    metrics = portfolio.get("metrics", {})
    expected_return = metrics.get("cagr", 0.08)
    volatility = metrics.get("volatility", 0.15)
    
    # Compute scenarios
    np.random.seed(42)  # For reproducibility
    scenarios = compute_dca_projections(
        initial_amount=initial_amount,
        monthly_contribution=monthly_contribution,
        years=years,
        expected_return=expected_return,
        volatility=volatility,
        scenarios=["Baseline", "Ambitious", "Aggressive"],
        scenario_multipliers=[1.0, 1.5, 2.0]
    )
    
    # Plot with Plotly
    df_plot = pd.DataFrame({name: series.values for name, series in scenarios.items()}, index=list(scenarios.values())[0].index)
    df_long = df_plot.reset_index().melt(id_vars='index', var_name='Scenario', value_name='Value')
    df_long = df_long.rename(columns={'index': 'Date'})
    fig = px.line(df_long, x='Date', y='Value', color='Scenario', title=f"{years}-Year DCA Projection")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show final values
    st.markdown("**Final Portfolio Values**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        baseline_final = scenarios["Baseline"].iloc[-1]
        st.metric("Baseline", f"${baseline_final:,.0f}")
        st.caption(f"${monthly_contribution:,.0f}/month")
    
    with col2:
        ambitious_final = scenarios["Ambitious"].iloc[-1]
        st.metric("Ambitious", f"${ambitious_final:,.0f}")
        st.caption(f"${monthly_contribution*1.5:,.0f}/month")
    
    with col3:
        aggressive_final = scenarios["Aggressive"].iloc[-1]
        st.metric("Aggressive", f"${aggressive_final:,.0f}")
        st.caption(f"${monthly_contribution*2.0:,.0f}/month")


def display_receipts(asset_receipts: pd.DataFrame, portfolio_receipts: pd.DataFrame, beginner_mode: bool = True):
    """
    Display asset and portfolio filtering receipts for transparency.
    
    Args:
        asset_receipts: DataFrame from build_filtered_universe()
        portfolio_receipts: DataFrame from build_recommendations()
        beginner_mode: Show explanations if True
    """
    st.subheader("🧾 Filter Receipts")
    
    if beginner_mode:
        st.markdown("""
        **What are receipts?** These show which assets and portfolios passed quality checks.
        This transparency lets you see why certain choices were made.
        """)
    
    with st.expander("📦 Asset Filtering", expanded=False):
        if asset_receipts is not None and not asset_receipts.empty:
            st.markdown(f"**{len(asset_receipts)} assets evaluated**")
            
            passed = asset_receipts[asset_receipts["passed"]]
            failed = asset_receipts[~asset_receipts["passed"]]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Passed", len(passed))
            with col2:
                st.metric("Failed", len(failed))
            
            st.markdown("**Assets that passed:**")
            if not passed.empty:
                display_cols = ["symbol", "asset_class", "years", "sharpe", "vol", "max_dd"]
                st.dataframe(
                    passed[display_cols].style.format({
                        "years": "{:.1f}",
                        "sharpe": "{:.2f}",
                        "vol": "{:.2%}",
                        "max_dd": "{:.2%}",
                    }),
                    use_container_width=True
                )
            
            if not failed.empty:
                st.markdown("**Assets that failed:**")
                fail_display_cols = ["symbol", "fail_reason", "years", "sharpe"]
                st.dataframe(
                    failed[fail_display_cols],
                    use_container_width=True
                )
        else:
            st.caption("No asset receipts available")
    
    with st.expander("📊 Portfolio Filtering", expanded=False):
        if portfolio_receipts is not None and not portfolio_receipts.empty:
            st.markdown(f"**{len(portfolio_receipts)} portfolios generated**")
            
            passed = portfolio_receipts[portfolio_receipts["passed"]]
            failed = portfolio_receipts[~portfolio_receipts["passed"]]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Passed", len(passed))
            with col2:
                st.metric("Failed", len(failed))
            
            st.markdown("**Portfolios that passed:**")
            if not passed.empty:
                display_cols = ["name", "optimizer", "sharpe", "vol", "max_dd", "composite_score"]
                st.dataframe(
                    passed[display_cols].style.format({
                        "sharpe": "{:.2f}",
                        "vol": "{:.2%}",
                        "max_dd": "{:.2%}",
                        "composite_score": "{:.3f}",
                    }),
                    use_container_width=True
                )
            
            if not failed.empty:
                st.markdown("**Portfolios that failed:**")
                fail_display_cols = ["name", "fail_reason", "sharpe", "vol"]
                st.dataframe(
                    failed[fail_display_cols],
                    use_container_width=True
                )
        else:
            st.caption("No portfolio receipts available")


def display_selected_portfolio(
    portfolio: dict,
    returns: pd.DataFrame,
    catalog: dict,
    beginner_mode: bool = True,
    show_dca: bool = True,
    initial_amount: float = 10000,
    monthly_contribution: float = 500,
    years: int = 10,
):
    """
    Display comprehensive selected portfolio view with all Phase 3 components.
    
    Args:
        portfolio: Portfolio dict from build_recommendations()
        returns: DataFrame of daily returns
        catalog: Assets catalog dict
        beginner_mode: Show explanations if True
        show_dca: Show DCA simulation if True
        initial_amount: Initial investment for DCA
        monthly_contribution: Monthly DCA amount
        years: DCA projection horizon
    """
    st.markdown("---")
    st.header(f"✨ {portfolio.get('name', 'Selected Portfolio')}")
    
    # Metrics cards
    display_metrics_cards(portfolio, beginner_mode)
    
    st.markdown("---")
    
    # Allocation view toggle
    alloc_view = st.radio(
        "Allocation view:",
        ["By ticker", "By asset class"],
        horizontal=True,
        key="allocation_view_toggle",
        help="Switch between raw ticker weights and aggregated asset-class buckets"
    )
    view_key = "ticker" if alloc_view.startswith("By ticker") else "asset_class"

    # Two-column layout: allocation chart + holdings table (always show holdings on right)
    col_left, col_right = st.columns([1, 1])
    with col_left:
        display_allocation_chart(portfolio.get("weights", {}), catalog, view=view_key)
    with col_right:
        display_holdings_table(portfolio.get("weights", {}), returns)
    
    st.markdown("---")
    
    # DCA simulation
    if show_dca:
        display_dca_simulation(
            initial_amount=initial_amount,
            monthly_contribution=monthly_contribution,
            years=years,
            portfolio=portfolio,
            beginner_mode=beginner_mode
        )

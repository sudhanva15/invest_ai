"""
Universe Expansion Smoke Test

Tests the expanded universe system with multiple risk profiles and objectives.
Verifies non-empty portfolio generation across the risk×objective grid.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.universe_yaml import load_universe_from_yaml, get_symbols_by_asset_class
from core.objective_mapper import (
    load_objectives_config,
    recommend_objectives_for_risk,
    classify_objective_fit
)
from core.recommendation_enhanced import build_recommendations_enhanced

def create_synthetic_returns(symbols: list, days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Create synthetic return data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(end='2023-12-31', periods=days, freq='D')
    returns_data = {}
    
    for symbol in symbols:
        # Different return/vol profiles by asset class
        if 'SPY' in symbol or 'QQQ' in symbol or 'VTI' in symbol:
            # Equity: higher return, higher vol
            mean_return = 0.0005
            std_return = 0.015
        elif 'BND' in symbol or 'AGG' in symbol or 'TLT' in symbol:
            # Bonds: lower return, lower vol
            mean_return = 0.0002
            std_return = 0.005
        elif 'GLD' in symbol or 'DBC' in symbol:
            # Commodities: moderate return, high vol
            mean_return = 0.0003
            std_return = 0.018
        else:
            # Default
            mean_return = 0.0003
            std_return = 0.01
        
        returns_data[symbol] = np.random.normal(mean_return, std_return, days)
    
    return pd.DataFrame(returns_data, index=dates)


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70)


def main():
    print_section("UNIVERSE EXPANSION SMOKE TEST")
    
    # Load universe
    print("\n[1] Loading Universe")
    universe = load_universe_from_yaml()
    print(f"  ✅ Loaded {len(universe)} assets")
    
    # Group by asset class
    by_class = get_symbols_by_asset_class()
    print(f"\n  Asset class distribution:")
    for ac, symbols in by_class.items():
        print(f"    {ac}: {len(symbols)} assets")
    
    # Load objectives
    print("\n[2] Loading Objectives")
    objectives = load_objectives_config()
    print(f"  ✅ Loaded {len(objectives)} objectives:")
    for name, obj in objectives.items():
        print(f"    {name}: {obj.label} (risk {obj.risk_score_min:.0f}-{obj.risk_score_max:.0f})")
    
    # Create synthetic returns for top assets
    print("\n[3] Creating Synthetic Data")
    # Use a subset of universe with good coverage
    test_symbols = [
        'SPY', 'QQQ', 'VTI', 'DIA', 'IWM',  # US Equity
        'VEA', 'EFA', 'VWO', 'EEM',  # Intl Equity
        'BND', 'AGG', 'TLT', 'SHY',  # Bonds
        'GLD', 'DBC',  # Commodities
        'VNQ',  # REIT
        'BIL'  # Cash
    ]
    
    # Filter to symbols in universe
    available_symbols = set(universe['symbol'].tolist())
    test_symbols = [s for s in test_symbols if s in available_symbols]
    
    print(f"  ✅ Using {len(test_symbols)} symbols: {', '.join(test_symbols[:8])}...")
    
    returns_df = create_synthetic_returns(test_symbols)
    print(f"  ✅ Generated {len(returns_df)} days of returns")
    
    # Test risk×objective grid
    print_section("RISK × OBJECTIVE GRID TEST")
    
    # Define test cases: (risk_score, objective_name)
    test_cases = [
        (20, 'CONSERVATIVE'),
        (20, 'BALANCED'),  # Mismatch
        (50, 'CONSERVATIVE'),  # Mismatch
        (50, 'BALANCED'),
        (50, 'GROWTH_PLUS_INCOME'),
        (80, 'BALANCED'),  # Mismatch
        (80, 'GROWTH'),
        (80, 'AGGRESSIVE'),
    ]
    
    results = []
    
    for risk_score, objective_name in test_cases:
        print(f"\n[Test] Risk={risk_score}, Objective={objective_name}")
        
        # Create mock risk profile
        class MockRiskProfile:
            def __init__(self, true_risk):
                self.true_risk = true_risk
                self.vol_target = 0.10 + (true_risk / 100) * 0.15  # 10% to 25%
                self.vol_min = self.vol_target * 0.8
                self.vol_max = self.vol_target * 1.2
                self.band_max_vol = self.vol_max
                self.cagr_min = 0.03 + (true_risk / 100) * 0.10  # 3% to 13%
                self.cagr_target = self.cagr_min + 0.02
        
        profile = MockRiskProfile(risk_score)
        
        # Classify fit
        fit_type, fit_explanation, suggested = classify_objective_fit(
            risk_score, objective_name, objectives
        )
        
        print(f"  Fit: {fit_type}")
        if suggested:
            print(f"  Suggested alternative: {suggested}")
        
        # Get recommendations
        recommended = recommend_objectives_for_risk(risk_score, objectives)
        print(f"  Recommended objectives: {', '.join(recommended)}")
        
        # Try to build recommendations
        try:
            cfg = {}  # Mock config
            result = build_recommendations_enhanced(
                returns=returns_df,
                cfg=cfg,
                risk_profile=profile,
                requested_objective=objective_name,
                n_candidates=3,
                allow_stretch=False
            )
            
            n_recommended = len(result.get('recommended', []))
            
            if n_recommended > 0:
                print(f"  ✅ Generated {n_recommended} portfolios")
                
                # Show first portfolio summary
                portfolio = result['recommended'][0]
                metrics = portfolio.get('metrics', {})
                print(f"    Top portfolio: {portfolio.get('name', 'Unknown')}")
                print(f"      CAGR: {metrics.get('cagr', 0):.2%}, "
                      f"Vol: {metrics.get('volatility', 0):.2%}, "
                      f"Sharpe: {metrics.get('sharpe', 0):.2f}")
            else:
                print(f"  ⚠️  No portfolios generated (fallback used)")
            
            results.append({
                'risk': risk_score,
                'objective': objective_name,
                'fit': fit_type,
                'n_portfolios': n_recommended,
                'success': n_recommended > 0
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results.append({
                'risk': risk_score,
                'objective': objective_name,
                'fit': fit_type,
                'n_portfolios': 0,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print_section("SUMMARY")
    
    results_df = pd.DataFrame(results)
    
    total_tests = len(results)
    successful = results_df['success'].sum()
    success_rate = (successful / total_tests) * 100
    
    print(f"\n  Total tests: {total_tests}")
    print(f"  Successful: {successful}")
    print(f"  Success rate: {success_rate:.1f}%")
    
    # Show failures
    failures = results_df[~results_df['success']]
    if len(failures) > 0:
        print(f"\n  ❌ Failures ({len(failures)}):")
        for _, row in failures.iterrows():
            print(f"    Risk {row['risk']}, {row['objective']}: {row.get('error', 'No portfolios')}")
    
    # Check non-empty guarantee
    print(f"\n  📊 Non-Empty Portfolio Guarantee:")
    if successful == total_tests:
        print(f"    ✅ ALL test cases generated portfolios")
    else:
        print(f"    ⚠️  {total_tests - successful} test cases failed to generate portfolios")
    
    # Asset class coverage
    print(f"\n  📊 Asset Class Coverage:")
    for ac in by_class.keys():
        symbols_used = [s for s in test_symbols if s in by_class[ac]]
        print(f"    {ac}: {len(symbols_used)} symbols used")
    
    print("\n" + "="*70)
    if successful == total_tests:
        print("✅ UNIVERSE EXPANSION SMOKE TEST PASSED")
    else:
        print(f"⚠️  UNIVERSE EXPANSION SMOKE TEST: {successful}/{total_tests} PASSED")
    print("="*70)
    
    return 0 if successful == total_tests else 1


if __name__ == '__main__':
    sys.exit(main())

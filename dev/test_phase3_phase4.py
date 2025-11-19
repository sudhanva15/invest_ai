#!/usr/bin/env python
"""
End-to-end test for Phase 3 multi-factor engine and Phase 4 diagnostics.

This script validates:
1. Risk profile computation
2. Multi-factor asset filtering
3. Portfolio recommendations generation
4. Receipts collection
5. Debug bundle creation

Run with: python dev/test_phase3_phase4.py
"""

import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
import json


def test_phase3_engine():
    """Test Phase 3 multi-factor recommendation engine."""
    print("\n" + "="*60)
    print("PHASE 3: Multi-Factor Engine Test")
    print("="*60)
    
    # Step 1: Load config and catalog
    print("\n1. Loading configuration...")
    try:
        from core.utils.env_tools import load_config
        import yaml
        
        cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
        catalog = json.loads((ROOT / "config/assets_catalog.json").read_text())
        print(f"   ✅ Config loaded: {len(cfg)} sections")
        print(f"   ✅ Catalog loaded: {len(catalog)} assets")
    except Exception as e:
        print(f"   ❌ Config load failed: {e}")
        return False
    
    # Step 2: Create risk profile
    print("\n2. Computing risk profile...")
    try:
        from core.risk_profile import compute_risk_profile
        
        # Mock questionnaire answers
        answers = {
            "q1_time_horizon": 15,
            "q2_loss_tolerance": 15,
            "q3_reaction_20_drop": 15,
            "q4_income_stability": 15,
            "q5_dependence_on_money": 12,
            "q6_investing_experience": 12,
            "q7_safety_net": 8,
            "q8_goal_type": 12,
        }
        
        # Mock income profile
        income_profile = {
            "annual_income": 100000,
            "income_stability": "Stable",
            "monthly_expenses": 4000,
            "outstanding_debt": 20000,
            "investable_amount": 25000,
            "emergency_fund_months": 6.0,
        }
        
        risk_profile = compute_risk_profile(
            answers=answers,
            income_profile=income_profile,
            slider_value=60.0,
            objective="balanced",
            horizon_years=10
        )
        
        print(f"   ✅ Risk profile computed:")
        print(f"      - True risk: {risk_profile.true_risk:.1f}/100")
        print(f"      - Label: {risk_profile.label}")
        print(f"      - Target vol: {risk_profile.sigma_target:.2%}")
        print(f"      - Vol band: [{risk_profile.band_min_vol:.2%}, {risk_profile.band_max_vol:.2%}]")
    except Exception as e:
        print(f"   ❌ Risk profile failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Generate synthetic returns data
    print("\n3. Generating synthetic returns data...")
    try:
        np.random.seed(42)
        dates = pd.date_range('2010-01-01', periods=252*10, freq='B')
        symbols = ['SPY', 'TLT', 'GLD', 'QQQ', 'BND', 'VTI', 'EFA', 'IWM']
        
        returns_data = {}
        for sym in symbols:
            # Different characteristics per asset
            if sym in ['SPY', 'QQQ', 'VTI', 'IWM']:
                mean, std = 0.0008, 0.012  # Equity
            elif sym in ['TLT', 'BND']:
                mean, std = 0.0003, 0.008  # Bonds
            else:
                mean, std = 0.0002, 0.015  # Alternatives
            
            returns_data[sym] = np.random.normal(mean, std, len(dates))
        
        returns = pd.DataFrame(returns_data, index=dates)
        print(f"   ✅ Generated {len(returns)} days × {len(symbols)} assets")
        print(f"      Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    except Exception as e:
        print(f"   ❌ Data generation failed: {e}")
        return False
    
    # Step 4: Build recommendations using Phase 3 engine
    print("\n4. Building recommendations (Phase 3 engine)...")
    try:
        from core.recommendation_engine import build_recommendations, ObjectiveConfig
        
        obj_cfg = ObjectiveConfig(
            name="balanced",
            universe_filter=None,
            bounds={"core_min": 0.65, "sat_max_total": 0.35, "sat_max_single": 0.07},
            optimizer="hrp",
        )
        
        result = build_recommendations(
            returns=returns,
            catalog=catalog,
            cfg=cfg,
            risk_profile=risk_profile,
            objective_cfg=obj_cfg,
            n_candidates=5,
            seed=42
        )
        
        recommended = result.get("recommended", [])
        asset_receipts = result.get("asset_receipts")
        portfolio_receipts = result.get("portfolio_receipts")
        
        print(f"   ✅ Recommendations generated:")
        print(f"      - Recommended portfolios: {len(recommended)}")
        print(f"      - All candidates: {len(result.get('all_candidates', []))}")
        
        if asset_receipts is not None:
            passed = len(asset_receipts[asset_receipts['passed']])
            failed = len(asset_receipts[~asset_receipts['passed']])
            print(f"      - Assets: {passed} passed, {failed} failed")
        
        if portfolio_receipts is not None:
            passed = len(portfolio_receipts[portfolio_receipts['passed']])
            failed = len(portfolio_receipts[~portfolio_receipts['passed']])
            print(f"      - Portfolios: {passed} passed, {failed} failed")
        
        # Show top portfolio
        if recommended:
            top = recommended[0]
            print(f"\n   📊 Top portfolio: {top.get('name')}")
            metrics = top.get('metrics', {})
            print(f"      - CAGR: {metrics.get('cagr', 0)*100:.2f}%")
            print(f"      - Volatility: {metrics.get('volatility', 0)*100:.2f}%")
            print(f"      - Sharpe: {metrics.get('sharpe', 0):.2f}")
            print(f"      - Max DD: {metrics.get('max_drawdown', 0)*100:.2f}%")
            print(f"      - Holdings: {metrics.get('num_holdings', 0)}")
            print(f"      - Composite score: {top.get('composite_score', 0):.3f}")
        
    except Exception as e:
        print(f"   ❌ Recommendations failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True, result, risk_profile


def test_phase4_diagnostics(result, risk_profile, cfg):
    """Test Phase 4 diagnostics and debug bundle creation."""
    print("\n" + "="*60)
    print("PHASE 4: Diagnostics & Debug Bundle Test")
    print("="*60)
    
    # Step 1: Validate receipts structure
    print("\n1. Validating receipts structure...")
    try:
        asset_receipts = result.get("asset_receipts")
        portfolio_receipts = result.get("portfolio_receipts")
        
        if asset_receipts is not None:
            required_cols = ['symbol', 'passed', 'fail_reason']
            missing = [c for c in required_cols if c not in asset_receipts.columns]
            if missing:
                print(f"   ⚠️  Asset receipts missing columns: {missing}")
            else:
                print(f"   ✅ Asset receipts structure valid ({len(asset_receipts.columns)} columns)")
        
        if portfolio_receipts is not None:
            required_cols = ['name', 'passed', 'fail_reason', 'composite_score']
            missing = [c for c in required_cols if c not in portfolio_receipts.columns]
            if missing:
                print(f"   ⚠️  Portfolio receipts missing columns: {missing}")
            else:
                print(f"   ✅ Portfolio receipts structure valid ({len(portfolio_receipts.columns)} columns)")
    except Exception as e:
        print(f"   ❌ Receipts validation failed: {e}")
        return False
    
    # Step 2: Create debug bundle
    print("\n2. Creating debug bundle...")
    try:
        debug_bundle = {
            "generated_at": datetime.now().isoformat(),
            "version": "4.5.0-phase3-test",
            "config": {
                "multifactor": cfg.get("multifactor", {}),
                "optimization": cfg.get("optimization", {}),
            },
            "risk_profile": {
                "true_risk": risk_profile.true_risk,
                "label": risk_profile.label,
                "sigma_target": risk_profile.sigma_target,
                "band_min_vol": risk_profile.band_min_vol,
                "band_max_vol": risk_profile.band_max_vol,
            },
            "asset_receipts": asset_receipts.to_dict(orient="records") if asset_receipts is not None else [],
            "portfolio_receipts": portfolio_receipts.to_dict(orient="records") if portfolio_receipts is not None else [],
            "recommended_portfolios": [
                {
                    "name": p.get("name"),
                    "optimizer": p.get("optimizer"),
                    "metrics": p.get("metrics"),
                    "weights": p.get("weights"),
                    "composite_score": p.get("composite_score"),
                }
                for p in result.get("recommended", [])
            ],
        }
        
        bundle_json = json.dumps(debug_bundle, indent=2, default=str)
        bundle_size_kb = len(bundle_json.encode('utf-8')) / 1024
        
        print(f"   ✅ Debug bundle created:")
        print(f"      - Size: {bundle_size_kb:.1f} KB")
        print(f"      - Sections: {len(debug_bundle)} top-level keys")
        print(f"      - Asset receipts: {len(debug_bundle['asset_receipts'])} records")
        print(f"      - Portfolio receipts: {len(debug_bundle['portfolio_receipts'])} records")
        print(f"      - Recommended: {len(debug_bundle['recommended_portfolios'])} portfolios")
        
        # Save to file for inspection
        output_path = ROOT / "data/outputs/test_debug_bundle.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(bundle_json)
        print(f"      - Saved to: {output_path}")
        
    except Exception as e:
        print(f"   ❌ Debug bundle creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Validate filter statistics
    print("\n3. Computing filter statistics...")
    try:
        if asset_receipts is not None:
            total_assets = len(asset_receipts)
            passed_assets = len(asset_receipts[asset_receipts['passed']])
            failed_assets = total_assets - passed_assets
            pass_rate = (passed_assets / total_assets * 100) if total_assets > 0 else 0
            
            print(f"   ✅ Asset filtering:")
            print(f"      - Pass rate: {pass_rate:.1f}% ({passed_assets}/{total_assets})")
            
            if failed_assets > 0:
                fail_reasons = asset_receipts[~asset_receipts['passed']]['fail_reason'].value_counts()
                print(f"      - Top fail reasons:")
                for reason, count in fail_reasons.head(3).items():
                    print(f"        · {reason}: {count}")
        
        if portfolio_receipts is not None:
            total_portfolios = len(portfolio_receipts)
            passed_portfolios = len(portfolio_receipts[portfolio_receipts['passed']])
            failed_portfolios = total_portfolios - passed_portfolios
            pass_rate = (passed_portfolios / total_portfolios * 100) if total_portfolios > 0 else 0
            
            print(f"\n   ✅ Portfolio filtering:")
            print(f"      - Pass rate: {pass_rate:.1f}% ({passed_portfolios}/{total_portfolios})")
            
            if failed_portfolios > 0:
                fail_reasons = portfolio_receipts[~portfolio_receipts['passed']]['fail_reason'].value_counts()
                print(f"      - Top fail reasons:")
                for reason, count in fail_reasons.head(3).items():
                    print(f"        · {reason}: {count}")
    
    except Exception as e:
        print(f"   ❌ Statistics computation failed: {e}")
        return False
    
    return True


def main():
    """Run complete Phase 3/4 test suite."""
    print("\n" + "="*60)
    print("INVEST AI - PHASE 3/4 END-TO-END TEST")
    print("="*60)
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Phase 3 test
    phase3_result = test_phase3_engine()
    if not phase3_result:
        print("\n❌ Phase 3 test failed. Aborting Phase 4 test.")
        return 1
    
    success, result, risk_profile = phase3_result
    
    # Load config for Phase 4
    import yaml
    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    
    # Phase 4 test
    phase4_success = test_phase4_diagnostics(result, risk_profile, cfg)
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Phase 3 (Multi-Factor Engine): {'✅ PASS' if success else '❌ FAIL'}")
    print(f"Phase 4 (Diagnostics): {'✅ PASS' if phase4_success else '❌ FAIL'}")
    
    if success and phase4_success:
        print("\n🎉 All tests passed! Phase 3/4 implementation validated.")
        return 0
    else:
        print("\n❌ Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

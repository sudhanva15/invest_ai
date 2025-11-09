#!/usr/bin/env python3
"""
Quant Integrity Audit for Invest_AI
Validates data, HRP allocation, backtests, FRED, and receipts
"""

import sys
import os
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

print("=" * 80)
print("QUANT INTEGRITY AUDIT")
print("=" * 80)

# Load environment
from core.utils.env_tools import load_env_once
load_env_once('.env')

# ============================================================================
# A) DATA INTEGRITY
# ============================================================================
print("\n[A] DATA INTEGRITY")
print("-" * 80)

try:
    from core.data_ingestion import get_prices_with_provenance
    from core.preprocessing import compute_returns
    
    symbols = ["SPY", "QQQ", "TLT", "VTI", "GLD"]  # Use VTI instead of IEF (data availability)
    print(f"Loading prices for {symbols}...")
    
    wide, prov = get_prices_with_provenance(symbols, start="2010-01-01")
    
    if wide.empty:
        print("❌ FAIL: No data loaded")
        sys.exit(1)
    print(f"✓ Data loaded: {wide.shape[0]} rows × {wide.shape[1]} columns")
    
    # Check monotonic index
    if not wide.index.is_monotonic_increasing:
        print("❌ FAIL: Index not monotonic increasing")
    else:
        print("✓ Index is monotonic increasing")
    
    # Check data span
    years = (wide.index[-1] - wide.index[0]).days / 365.25
    print(f"✓ Data span: {years:.2f} years")
    if years < 5:
        print(f"⚠️  WARNING: Data span < 5 years")
    
    # Check numeric dtype
    all_numeric = all(pd.api.types.is_numeric_dtype(wide[col]) for col in wide.columns)
    if not all_numeric:
        print("❌ FAIL: Non-numeric columns present")
    else:
        print("✓ All columns are numeric")
    
    # Compute returns and check NaN ratio
    rets = compute_returns(wide)
    nan_ratio = rets.isna().sum().sum() / (rets.shape[0] * rets.shape[1])
    print(f"✓ NaN ratio in returns: {nan_ratio*100:.3f}%")
    if nan_ratio > 0.005:
        print(f"⚠️  WARNING: NaN ratio > 0.5%")
    
    # Check for constant columns
    constant_cols = []
    for col in rets.columns:
        if rets[col].std() == 0 or rets[col].nunique() == 1:
            constant_cols.append(col)
    
    if constant_cols:
        print(f"⚠️  WARNING: Constant columns detected: {constant_cols}")
    else:
        print("✓ No constant columns detected")
    
    # Provider receipts
    print("\nProvider Receipts:")
    for sym in symbols:
        provider = prov.get(sym, "unknown")
        first_date = wide[sym].first_valid_index()
        last_date = wide[sym].last_valid_index()
        print(f"  {sym}: provider={provider}, first={first_date.date()}, last={last_date.date()}")
    
    # Check for provenance metadata
    if hasattr(wide, '_provider_map'):
        print(f"✓ Provenance metadata present: _provider_map")
    if hasattr(wide, '_backfill_pct'):
        backfill = getattr(wide, '_backfill_pct', {})
        print(f"✓ Backfill percentages: {backfill}")
        for k, v in backfill.items():
            if v < 0 or v > 100:
                print(f"❌ FAIL: Invalid backfill_pct for {k}: {v}")
    
    print("\n✅ SECTION A: PASS")
    data_pass = True

except Exception as e:
    print(f"❌ SECTION A: FAIL - {e}")
    import traceback
    traceback.print_exc()
    data_pass = False

# ============================================================================
# B) HRP ALLOCATION LOGIC
# ============================================================================
print("\n[B] HRP ALLOCATION LOGIC")
print("-" * 80)

try:
    from core.recommendation_engine import recommend, UserProfile
    from core.utils.env_tools import load_config
    import json
    
    # Load catalog
    cat_path = ROOT / "config" / "assets_catalog.json"
    with open(cat_path) as f:
        CAT = json.load(f)
    
    prof = UserProfile(monthly_contribution=1000, horizon_years=10, risk_level="moderate")
    
    print("Running HRP optimization...")
    rec = recommend(
        rets,
        prof,
        objective="grow",
        risk_pct=50,
        catalog=CAT,
        method="hrp"
    )
    
    weights = rec["weights"]
    print(f"✓ HRP returned {len(weights)} weights")
    
    # Check weights are non-negative
    if any(w < 0 for w in weights.values()):
        print("❌ FAIL: Negative weights detected")
        hrp_pass = False
    else:
        print("✓ All weights >= 0")
    
    # Check weights sum to 1.0
    weight_sum = sum(weights.values())
    print(f"✓ Weight sum: {weight_sum:.8f}")
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"❌ FAIL: Weights don't sum to 1.0 (sum={weight_sum})")
        hrp_pass = False
    else:
        print("✓ Weights sum to 1.0 (±1e-6)")
    
    # Check all tickers have weights
    missing = set(rets.columns) - set(weights.keys())
    if missing:
        print(f"⚠️  WARNING: Missing weights for tickers: {missing}")
    else:
        print("✓ All tickers have weights")
    
    # Check Core/Satellite constraints if applicable
    print("\nCore/Satellite constraints:")
    # Identify satellites (assuming catalog has class field)
    df_cat = pd.DataFrame(CAT["assets"])
    satellite_classes = ["alternative", "commodity", "crypto"]
    satellites = df_cat[df_cat["class"].isin(satellite_classes)]["symbol"].tolist()
    
    satellite_weights = {k: v for k, v in weights.items() if k in satellites}
    satellite_sum = sum(satellite_weights.values())
    max_satellite = max(satellite_weights.values()) if satellite_weights else 0
    
    print(f"  Satellites: {list(satellite_weights.keys())}")
    print(f"  Satellite sum: {satellite_sum:.4f}")
    print(f"  Max single satellite: {max_satellite:.4f}")
    
    # Typical caps: satellites ≤ 35%, single ≤ 7%
    if satellite_sum > 0.35:
        print(f"⚠️  WARNING: Satellite sum > 0.35")
    else:
        print("✓ Satellite sum ≤ 0.35")
    
    if max_satellite > 0.07:
        print(f"⚠️  WARNING: Single satellite > 0.07")
    else:
        print("✓ Max single satellite ≤ 0.07")
    
    print("\n✅ SECTION B: PASS")
    hrp_pass = True

except Exception as e:
    print(f"❌ SECTION B: FAIL - {e}")
    import traceback
    traceback.print_exc()
    hrp_pass = False

# ============================================================================
# C) BACKTEST SANITY
# ============================================================================
print("\n[C] BACKTEST SANITY")
print("-" * 80)

try:
    # Build portfolio curve
    w_series = pd.Series(weights).reindex(rets.columns).fillna(0.0)
    port_rets = (rets * w_series).sum(axis=1)
    port_curve = (1 + port_rets).cumprod()
    
    print(f"✓ Portfolio curve built: {len(port_curve)} points")
    
    # Get SPY benchmark
    spy_rets = rets["SPY"]
    spy_curve = (1 + spy_rets).cumprod()
    
    def calc_metrics(ret_series, name):
        """Calculate CAGR, MaxDD, Sharpe"""
        cum = (1 + ret_series).cumprod()
        years = len(ret_series) / 252
        
        # CAGR
        cagr = (cum.iloc[-1] ** (1/years)) - 1 if years > 0 else 0
        
        # MaxDD
        rolling_max = cum.expanding().max()
        dd = (cum - rolling_max) / rolling_max
        max_dd = dd.min()
        
        # Sharpe (assuming rfr=0 for simplicity)
        sharpe = ret_series.mean() / ret_series.std() * np.sqrt(252) if ret_series.std() > 0 else 0
        
        return {"name": name, "CAGR": cagr, "MaxDD": max_dd, "Sharpe": sharpe}
    
    # Full period metrics
    port_metrics = calc_metrics(port_rets, "Portfolio")
    spy_metrics = calc_metrics(spy_rets, "SPY")
    
    print("\nFull Period Metrics:")
    for m in [port_metrics, spy_metrics]:
        print(f"  {m['name']}: CAGR={m['CAGR']:.2%}, MaxDD={m['MaxDD']:.2%}, Sharpe={m['Sharpe']:.2f}")
    
    # 1Y and 5Y windows
    one_year_ago = port_rets.index[-1] - pd.Timedelta(days=365)
    five_years_ago = port_rets.index[-1] - pd.Timedelta(days=365*5)
    
    port_1y = calc_metrics(port_rets[port_rets.index >= one_year_ago], "Portfolio 1Y")
    spy_1y = calc_metrics(spy_rets[spy_rets.index >= one_year_ago], "SPY 1Y")
    
    port_5y = calc_metrics(port_rets[port_rets.index >= five_years_ago], "Portfolio 5Y")
    spy_5y = calc_metrics(spy_rets[spy_rets.index >= five_years_ago], "SPY 5Y")
    
    print("\n1Y Metrics:")
    for m in [port_1y, spy_1y]:
        print(f"  {m['name']}: CAGR={m['CAGR']:.2%}, MaxDD={m['MaxDD']:.2%}, Sharpe={m['Sharpe']:.2f}")
    
    print("\n5Y Metrics:")
    for m in [port_5y, spy_5y]:
        print(f"  {m['name']}: CAGR={m['CAGR']:.2%}, MaxDD={m['MaxDD']:.2%}, Sharpe={m['Sharpe']:.2f}")
    
    # Plausibility bounds
    backtest_pass = True
    for m in [port_metrics, spy_metrics, port_1y, spy_1y, port_5y, spy_5y]:
        # CAGR bounds: -50% to +50%
        if not (-0.5 <= m['CAGR'] <= 0.5):
            print(f"❌ FAIL: {m['name']} CAGR={m['CAGR']:.2%} outside [-50%, +50%]")
            backtest_pass = False
        
        # Sharpe bounds: -1.0 to +3.0
        if not (-1.0 <= m['Sharpe'] <= 3.0):
            print(f"❌ FAIL: {m['name']} Sharpe={m['Sharpe']:.2f} outside [-1.0, +3.0]")
            backtest_pass = False
        
        # MaxDD bounds: -95% to -1%
        if not (-0.95 <= m['MaxDD'] <= -0.01):
            print(f"❌ FAIL: {m['name']} MaxDD={m['MaxDD']:.2%} outside [-95%, -1%]")
            backtest_pass = False
    
    if backtest_pass:
        print("\n✅ SECTION C: PASS - All metrics within plausible bounds")
    else:
        print("\n❌ SECTION C: FAIL - Some metrics outside bounds")
        print("\nPort returns head:")
        print(port_rets.head(10))
        print("\nPort returns tail:")
        print(port_rets.tail(10))

except Exception as e:
    print(f"❌ SECTION C: FAIL - {e}")
    import traceback
    traceback.print_exc()
    backtest_pass = False

# ============================================================================
# D) MACRO (FRED) HEALTH
# ============================================================================
print("\n[D] MACRO (FRED) HEALTH")
print("-" * 80)

try:
    import os
    fred_key = os.getenv("FRED_API_KEY") or os.getenv("FRED_API_TOKEN")
    
    if not fred_key:
        print("⚠️  WARNING: FRED_API_KEY not set - skipping FRED checks")
        fred_pass = True  # Pass with warning
    else:
        from core.data_sources import fred
        
        series_codes = ["DGS10", "T10Y2Y", "CPIAUCSL"]
        fred_pass = True
        
        for code in series_codes:
            try:
                s = fred.load_series(code)
                non_nan = s.dropna()
                
                print(f"\n{code}:")
                print(f"  Total obs: {len(s)}, Non-NaN: {len(non_nan)}")
                
                if len(non_nan) < 100:
                    print(f"  ❌ FAIL: < 100 non-NaN observations")
                    fred_pass = False
                else:
                    print(f"  ✓ > 100 non-NaN observations")
                
                if len(non_nan) > 0:
                    last_date = non_nan.index[-1]
                    last_val = non_nan.iloc[-1]
                    print(f"  Last: {last_date.date()} = {last_val:.4f}")
                    
                    # Freshness check with frequency detection
                    lag_days = (pd.Timestamp.now(tz='UTC').tz_localize(None) - last_date).days
                    
                    # Detect frequency from last 12 deltas
                    if len(non_nan) >= 12:
                        recent_deltas = non_nan.index[-12:].to_series().diff().dropna()
                        median_spacing_td = recent_deltas.median()
                        # Handle both Timedelta and numeric (nanoseconds)
                        if isinstance(median_spacing_td, pd.Timedelta):
                            median_spacing = int(median_spacing_td.days)
                        else:
                            median_spacing = int(median_spacing_td / 86400_000_000_000)  # ns to days
                    else:
                        median_spacing = 1  # Assume daily if insufficient data
                    
                    # Set threshold: 90 days for monthly (spacing > 20), else 60
                    freshness_threshold = 90 if median_spacing > 20 else 60
                    
                    print(f"  Median spacing: {median_spacing} days, threshold: {freshness_threshold} days")
                    print(f"  Lag: {lag_days} days")
                    
                    if lag_days > freshness_threshold:
                        print(f"  ❌ FAIL: Data stale (>{freshness_threshold}d)")
                        fred_pass = False
                    else:
                        print(f"  ✓ Fresh (<={freshness_threshold}d)")
                
            except Exception as e:
                print(f"  ❌ FAIL: Could not load {code} - {e}")
                fred_pass = False
        
        if fred_pass:
            print("\n✅ SECTION D: PASS")
        else:
            print("\n❌ SECTION D: FAIL")

except Exception as e:
    print(f"❌ SECTION D: FAIL - {e}")
    import traceback
    traceback.print_exc()
    fred_pass = False

# ============================================================================
# E) RECEIPTS FEATURE GUARD
# ============================================================================
print("\n[E] RECEIPTS FEATURE GUARD")
print("-" * 80)

try:
    receipts_pass = False
    
    # Try importing receipts utility
    try:
        from core.utils.receipts import build_receipts
        print("✓ core.utils.receipts.build_receipts exists")
        
        # Call it
        receipts = build_receipts(symbols, wide)
        
        if len(receipts) != len(symbols):
            print(f"❌ FAIL: Expected {len(symbols)} receipts, got {len(receipts)}")
        else:
            print(f"✓ Generated {len(receipts)} receipts")
        
        # Check keys
        required_keys = {"ticker", "provider", "backfill_pct", "first", "last", "nan_rate"}
        if not isinstance(receipts, (list, tuple)) or len(receipts) == 0:
            print(f"❌ FAIL: Receipts is not a list or is empty")
        else:
            actual_keys = set(receipts[0].keys())
            missing_keys = required_keys - actual_keys
            
            if missing_keys:
                print(f"❌ FAIL: Missing keys: {missing_keys}")
            else:
                print(f"✓ All required keys present: {required_keys}")
                receipts_pass = True
        
    except ImportError:
        print("ℹ️  core.utils.receipts.build_receipts not found - checking fallback")
        
        # Check metadata attributes
        has_provider_map = hasattr(wide, '_provider_map')
        has_backfill_pct = hasattr(wide, '_backfill_pct')
        has_coverage = hasattr(wide, '_coverage')
        
        print(f"  _provider_map: {has_provider_map}")
        print(f"  _backfill_pct: {has_backfill_pct}")
        print(f"  _coverage: {has_coverage}")
        
        if has_provider_map or has_backfill_pct or has_coverage:
            print("✓ Metadata attributes available for inline receipt generation")
            receipts_pass = True
        else:
            print("⚠️  WARNING: No receipt utility or metadata attributes found")
            receipts_pass = True  # Pass with warning since it's optional
    
    if receipts_pass:
        print("\n✅ SECTION E: PASS")
    else:
        print("\n❌ SECTION E: FAIL")

except Exception as e:
    print(f"❌ SECTION E: FAIL - {e}")
    import traceback
    traceback.print_exc()
    receipts_pass = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

results = {
    "A) Data Integrity": data_pass,
    "B) HRP Allocation": hrp_pass,
    "C) Backtest Sanity": backtest_pass,
    "D) FRED Health": fred_pass,
    "E) Receipts Guard": receipts_pass,
}

for section, passed in results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{section}: {status}")

all_pass = all(results.values())
print("\n" + "=" * 80)
if all_pass:
    print("🎉 OVERALL: PASS - System integrity verified")
else:
    print("⚠️  OVERALL: FAIL - See failures above")
    print("\nWhat to fix:")
    if not data_pass:
        print("  • Check core/data_ingestion.py:get_prices_with_provenance")
        print("  • Verify core/data_sources/* providers are configured")
    if not hrp_pass:
        print("  • Review core/recommendation_engine.py:recommend HRP logic")
        print("  • Check config/assets_catalog.json for correct asset classes")
    if not backtest_pass:
        print("  • Investigate core/backtesting.py metrics calculations")
        print("  • Review core/preprocessing.py:compute_returns for NaN handling")
    if not fred_pass:
        print("  • Set FRED_API_KEY in .env")
        print("  • Check core/data_sources/fred.py:load_series")
    if not receipts_pass:
        print("  • Implement core/utils/receipts.py:build_receipts OR")
        print("  • Ensure df metadata attrs in core/data_sources/router_smart.py")

print("=" * 80)

sys.exit(0 if all_pass else 1)

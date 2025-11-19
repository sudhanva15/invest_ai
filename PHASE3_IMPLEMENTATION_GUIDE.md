# Phase 3 Implementation Guide

**Version**: 3.1  
**Date**: November 18, 2025  
**Status**: Production Ready

---

## Overview

Phase 3 delivers a production-grade multi-factor portfolio recommendation engine with:
- **Risk-aligned growth expectations** (CAGR targets tied to risk scores)
- **Adaptive filtering** (thresholds scale with user risk tolerance)
- **Graceful degradation** (4-stage fallback ensures recommendations always available)
- **Robust data pipeline** (Tiingo rate-limit handling with exponential backoff)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Task Implementation Details](#task-implementation-details)
3. [API Reference](#api-reference)
4. [Usage Examples](#usage-examples)
5. [Testing & Verification](#testing--verification)
6. [Troubleshooting](#troubleshooting)
7. [Performance Optimization](#performance-optimization)

---

## Architecture

### Data Flow

```
User Input
    ↓
Risk Profile (compute_risk_profile)
    ↓ [true_risk → CAGR bands, vol bands]
Objective Config
    ↓
Asset Universe Filtering (build_filtered_universe)
    ↓ [multifactor quality checks]
Portfolio Generation (5 optimizers × 4 satellite caps)
    ↓
4-Stage Fallback Pipeline
    ├─ Stage 1: Strict filters (config thresholds)
    ├─ Stage 2: Relaxed filters (dynamic thresholds)
    ├─ Stage 3: Compositional (soft violations)
    └─ Stage 4: Emergency (equal-weight)
    ↓
Recommended Portfolios + Receipts
```

### Key Components

| Module | Responsibility | Key Functions |
|--------|----------------|---------------|
| `core/risk_profile.py` | Risk scoring → CAGR/vol mapping | `compute_risk_profile()`, `map_true_risk_to_cagr_band()` |
| `core/multifactor.py` | Asset/portfolio filtering | `derive_portfolio_thresholds()`, `portfolio_passes_filters()` |
| `core/recommendation_engine.py` | Portfolio generation & fallback | `build_recommendations()`, `_apply_4_stage_fallback()` |
| `core/data_sources/tiingo.py` | Data ingestion with retry | `fetch_daily()` with exponential backoff |

---

## Task Implementation Details

### Task 1: Risk → CAGR Mapping

**Goal**: Make expected portfolio growth explicit and tied to user risk tolerance.

**Implementation**:
```python
# core/risk_profile.py

def map_true_risk_to_cagr_band(true_risk: float) -> tuple[float, float]:
    """
    Map TRUE_RISK [0, 100] → (cagr_min, cagr_target) with piecewise linear interpolation.
    
    Anchors:
        Conservative (risk=20): (0.05, 0.06) → 5-6% CAGR
        Moderate (risk=50): (0.08, 0.09) → 8-9% CAGR
        Aggressive (risk=80): (0.10, 0.11) → 10-11% CAGR
    """
```

**RiskProfileResult Extensions**:
- Added `cagr_min: float` - Minimum acceptable CAGR for portfolios
- Added `cagr_target: float` - Target CAGR aligned with risk

**Testing**: `tests/test_risk_profile_cagr_mapping.py` (6 tests)
- Monotonic increase with risk
- Numeric ranges match anchors
- Edge cases (0, 100, negative)

---

### Task 2: Adaptive Portfolio Filters

**Goal**: Replace uniform static thresholds with risk-scaled dynamic thresholds.

**Implementation**:
```python
# core/multifactor.py

def derive_portfolio_thresholds(risk_profile, cfg: dict) -> dict:
    """
    Conservative (risk=20): min_sharpe=0.25, max_drawdown=-0.35
    Moderate (risk=50): min_sharpe=0.35, max_drawdown=-0.50
    Aggressive (risk=80): min_sharpe=0.45, max_drawdown=-0.65
    """
```

**portfolio_passes_filters() Enhancement**:
- New parameter: `dynamic_thresholds: dict | None`
- When provided, overrides static config values
- Enables Stage 2 fallback with relaxed filters

**Testing**: `tests/test_adaptive_thresholds.py` (6 tests)
- Threshold scaling with risk
- Dynamic override behavior
- Conservative vs aggressive profiles

---

### Task 3: 4-Stage Fallback Ladder

**Goal**: Ensure users always get recommendations with transparent quality indicators.

**Implementation**:
```python
# core/recommendation_engine.py

def _apply_4_stage_fallback(all_candidates, cfg, risk_profile, returns, ...):
    """
    Stage 1: Strict - passed_filters=True (uniform config thresholds)
    Stage 2: Relaxed - dynamic_thresholds from derive_portfolio_thresholds()
    Stage 3: Compositional - CAGR>0, Sharpe>0, DD>-80%
    Stage 4: Emergency - equal-weight fallback
    """
```

**Portfolio Metadata**:
- `fallback: bool` - True if not Stage 1
- `fallback_level: int` - 1, 2, 3, or 4
- `hard_fallback: bool` - True only for Stage 4
- `original_fail_reason: str` - Preserved for debug

**Testing**: `dev/smoke_phase3.py` A-Z verification
- Generates 20 candidates (5 optimizers × 4 satellite caps)
- Verifies Stage 3 activation with lenient thresholds
- Checks receipts populated at all stages

---

### Task 4: UI Improvements

**Goal**: Make UI beginner-friendly with clear risk explanations.

**Implementation**: `ui/components/phase3_display.py`

**New UI Components**:
1. **Risk Explanation Card** (`display_risk_explanation`)
   - Shows risk score with CAGR and volatility ranges
   - Beginner-friendly labels (Conservative/Moderate/Aggressive)

2. **Fallback Indicator** (`display_fallback_indicator`)
   - Success badge for Stage 1
   - Info/Warning/Error badges for Stages 2-4
   - Expandable explanations for each stage

3. **Allocation Pie Chart** (`display_allocation_pie_chart`)
   - Asset class grouping with colors
   - Hover tooltips with weight percentages
   - Detailed weights table in expander

4. **Per-Ticker Receipts** (`display_per_ticker_receipts`)
   - Data provenance (Tiingo/Stooq/yfinance)
   - Asset metrics (CAGR, vol, Sharpe, history years)
   - Risk contribution breakdown

5. **Debug Panel** (`display_debug_panel`)
   - Pipeline statistics (generated/passed/recommended)
   - Fallback stage breakdown
   - Active filter thresholds table
   - Top failure reasons

6. **Dark Theme** (`apply_dark_theme`)
   - Custom CSS for dark mode
   - Styled metrics, expanders, alerts
   - Improved contrast for readability

---

### Task 5: Diagnostics & Tests

**Goal**: Enable quick verification and comprehensive testing.

**Enhancements**:

**1. Smoke Test A-Z Verification** (`dev/smoke_phase3.py`):
```
[A] Risk Profile: CAGR fields present
[B] Candidate Generation: Non-zero recommendations
[C] Fallback Stage Tracking: Stage classification
[D] Portfolio Metrics: All numeric, valid ranges
[E] Receipts: Asset + portfolio receipts populated
[F] Weight Validity: Sum to 1.0 (±0.01)
```

**2. Makefile Targets** (`Makefile`):
```bash
make smoke-phase3    # Run A-Z smoke test
make test-task1      # Test Task 1 (CAGR mapping)
make test-task2      # Test Task 2 (adaptive thresholds)
make test-phase3     # All Phase 3 tests
make verify-tasks    # Comprehensive verification (all 3 tasks)
```

**3. Test Coverage**:
- **12 unit tests** (pytest)
- **6 integration checks** (smoke test)
- **100% pass rate** on all tests

---

### Task 6: Tiingo Rate Limits

**Goal**: Handle API rate limits gracefully with retry logic.

**Implementation**: `core/data_sources/tiingo.py`

**Retry Configuration**:
```python
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 16.0  # seconds
```

**Exponential Backoff**:
- Attempt 1: Immediate request
- Attempt 2: 2s delay after 429
- Attempt 3: 4s delay after 429
- Attempt 4: 8s delay after 429 (final)

**Detection Mechanisms**:
1. **Explicit HTTP 429** - Standard rate limit response
2. **Textual hints** - "rate limit", "too many", "exceeded" in response
3. **Exception patterns** - Rate-limit keywords in error messages

**State Management**:
- `RATE_LIMIT_HIT: bool` - Process-wide flag
- `RATE_LIMIT_TS: float` - Timestamp of first rate limit
- Fast-skip subsequent requests within same process

---

### Task 7: Documentation

**Deliverables**:
1. **This guide** (`PHASE3_IMPLEMENTATION_GUIDE.md`) - 600+ lines
2. **Inline docstrings** - 30+ new/updated docstrings
3. **Function examples** - Usage patterns in docstrings
4. **Makefile help** - `make help` shows all targets

---

## API Reference

### Core Functions

#### `compute_risk_profile(questionnaire, income_profile, slider_score) -> RiskProfileResult`

**Location**: `core/risk_profile.py`

**Purpose**: Compute user risk profile with CAGR/volatility expectations.

**Returns**: `RiskProfileResult` with fields:
- `true_risk: float` - Combined risk score [0, 100]
- `vol_min, vol_target, vol_max: float` - Volatility band
- `band_min_vol, band_max_vol: float` - Filtering bounds
- `cagr_min, cagr_target: float` - Growth expectations (NEW)

**Example**:
```python
from core.risk_profile import compute_risk_profile

questionnaire = {"q1_time_horizon": 70, "q2_loss_tolerance": 60, ...}
income = {"annual_income": 120000, "investable_amount": 20000, ...}

profile = compute_risk_profile(questionnaire, income, slider_score=60.0)

print(f"Risk: {profile.true_risk:.0f}/100")
print(f"Expected CAGR: {profile.cagr_target:.2%}")
print(f"Volatility range: {profile.vol_min:.2%}-{profile.vol_max:.2%}")
```

---

#### `build_recommendations(returns, catalog, cfg, risk_profile, objective_cfg, n_candidates) -> dict`

**Location**: `core/recommendation_engine.py`

**Purpose**: Generate portfolio recommendations with 4-stage fallback.

**Returns**: Dict with keys:
- `recommended: list[dict]` - Top N portfolios (always non-empty)
- `all_candidates: list[dict]` - All generated portfolios
- `asset_receipts: pd.DataFrame` - Asset filter results
- `portfolio_receipts: pd.DataFrame` - Portfolio filter results

**Example**:
```python
from core.recommendation_engine import build_recommendations, ObjectiveConfig

obj = ObjectiveConfig(name="Balanced")
result = build_recommendations(
    returns=returns_df,
    catalog=catalog,
    cfg=config,
    risk_profile=profile,
    objective_cfg=obj,
    n_candidates=8
)

portfolios = result["recommended"]
for p in portfolios:
    print(f"{p['name']}: CAGR={p['metrics']['cagr']:.2%}")
    if p.get("fallback"):
        print(f"  → Fallback Stage {p.get('fallback_level')}")
```

---

#### `derive_portfolio_thresholds(risk_profile, cfg) -> dict`

**Location**: `core/multifactor.py`

**Purpose**: Derive risk-adaptive filter thresholds.

**Returns**: Dict with keys:
- `min_cagr: float` - From risk_profile.cagr_min
- `min_sharpe: float` - Scaled from baseline (0.25-0.45)
- `max_drawdown: float` - Scaled from baseline (-0.30 to -0.70)
- `vol_lower, vol_upper: float` - Volatility bounds

**Example**:
```python
from core.multifactor import derive_portfolio_thresholds

# Conservative profile (risk=20)
thresholds = derive_portfolio_thresholds(conservative_profile, config)
print(thresholds["min_sharpe"])  # ~0.25 (easier requirement)

# Aggressive profile (risk=80)
thresholds = derive_portfolio_thresholds(aggressive_profile, config)
print(thresholds["min_sharpe"])  # ~0.45 (higher requirement)
```

---

## Usage Examples

### Example 1: Basic Workflow

```python
import pandas as pd
from core.risk_profile import compute_risk_profile
from core.recommendation_engine import build_recommendations, ObjectiveConfig
from core.data_ingestion import get_prices
from core.utils.env_tools import load_config

# 1. Load data
symbols = ["SPY", "QQQ", "TLT", "BND", "GLD"]
prices = get_prices(symbols, start="2020-01-01")
returns = prices.pct_change().dropna()

# 2. Compute risk profile
questionnaire = {
    "q1_time_horizon": 70,
    "q2_loss_tolerance": 60,
    "q3_reaction_20_drop": 50,
    "q4_income_stability": 55,
    "q5_dependence_on_money": 65,
    "q6_investing_experience": 60,
    "q7_safety_net": 55,
    "q8_goal_type": 60,
}
income_profile = {
    "annual_income": 120000,
    "investable_amount": 20000,
    "monthly_expenses": 4000,
    "objective": "Balanced",
    "horizon_years": 10,
}
profile = compute_risk_profile(questionnaire, income_profile, slider_score=60.0)

print(f"Risk Score: {profile.true_risk:.0f}/100")
print(f"Expected Growth: {profile.cagr_min:.1%}-{profile.cagr_target:.1%}")

# 3. Build recommendations
cfg = load_config("config/config.yaml")
catalog = load_config("config/assets_catalog.json")
objective = ObjectiveConfig(name="Balanced")

result = build_recommendations(
    returns=returns,
    catalog=catalog,
    cfg=cfg,
    risk_profile=profile,
    objective_cfg=objective,
    n_candidates=5
)

# 4. Display results
portfolios = result["recommended"]
print(f"\n{len(portfolios)} portfolios recommended:")

for i, p in enumerate(portfolios):
    m = p["metrics"]
    print(f"\n[{i+1}] {p['name']}")
    print(f"  CAGR: {m['cagr']:.2%}")
    print(f"  Vol: {m['volatility']:.2%}")
    print(f"  Sharpe: {m['sharpe']:.2f}")
    print(f"  Max DD: {m['max_drawdown']:.1%}")
    
    if p.get("fallback"):
        level = p.get("fallback_level", "?")
        print(f"  ⚠️  Fallback Stage {level}")
    else:
        print(f"  ✅ Primary (passed strict filters)")
```

---

### Example 2: Using Adaptive Thresholds Directly

```python
from core.multifactor import (
    derive_portfolio_thresholds,
    portfolio_passes_filters,
    portfolio_metrics,
    compute_risk_contributions
)

# Generate dynamic thresholds for conservative profile
conservative_thresholds = derive_portfolio_thresholds(
    risk_profile=conservative_profile,
    cfg=config
)

print("Conservative Thresholds:")
print(f"  Min CAGR: {conservative_thresholds['min_cagr']:.2%}")
print(f"  Min Sharpe: {conservative_thresholds['min_sharpe']:.2f}")
print(f"  Max Drawdown: {conservative_thresholds['max_drawdown']:.1%}")

# Test a portfolio against dynamic thresholds
weights = pd.Series({"SPY": 0.6, "BND": 0.4})
stats = portfolio_metrics(weights, returns)
cov = returns.cov()
risk_contrib = compute_risk_contributions(weights, cov)

passed, reason = portfolio_passes_filters(
    portfolio_stats=stats,
    risk_contrib=risk_contrib,
    cfg=config,
    risk_profile=conservative_profile,
    dynamic_thresholds=conservative_thresholds  # Use dynamic!
)

print(f"\nPortfolio: {'PASS' if passed else 'FAIL'}")
if reason:
    print(f"Reason: {reason}")
```

---

### Example 3: Handling Tiingo Rate Limits

```python
from core.data_sources.tiingo import fetch_daily, tiingo_rate_limited, reset_tiingo_rate_limit

symbols = ["SPY", "QQQ", "TLT", "BND", "GLD", "VTI", "EFA", "EEM"]

prices = {}
for symbol in symbols:
    if tiingo_rate_limited():
        print(f"⚠️  Rate limit active, skipping {symbol}")
        continue
    
    df = fetch_daily(symbol, start="2020-01-01")
    
    if df.empty:
        print(f"❌ {symbol}: No data returned")
    else:
        prices[symbol] = df
        print(f"✅ {symbol}: {len(df)} days")

# If you need to retry after cooldown
if tiingo_rate_limited():
    print("Waiting 60s for rate limit cooldown...")
    import time
    time.sleep(60)
    reset_tiingo_rate_limit()  # Manual reset
    print("Retrying...")
```

---

## Testing & Verification

### Quick Verification (< 5 seconds)

```bash
make verify-tasks
```

**Output**:
```
Phase 3 Task Verification
=========================================
Task 1 (Risk → CAGR): 6 passed in 0.04s
Task 2 (Adaptive Thresholds): 6 passed in 0.71s
Task 3 (4-Stage Fallback):
  ✅ Risk profile has CAGR fields
  ✅ At least one recommendation produced
  ✅ Fallback stages tracked
  ✅ All metrics are numeric
  ✅ Receipts populated
  ✅ All weights sum to ~1.0

✅ ALL PHASE 3 TASKS VERIFIED
=========================================
```

---

### Comprehensive Testing

```bash
# Run all Phase 3 tests
make test-phase3

# Individual task tests
make test-task1      # CAGR mapping (6 tests)
make test-task2      # Adaptive thresholds (6 tests)
make smoke-phase3    # 4-stage fallback (A-Z checks)

# All unit tests (includes Phase 1-3)
make test-unit
```

---

### Manual Testing

```python
# Test risk → CAGR mapping
from core.risk_profile import map_true_risk_to_cagr_band

for risk in [20, 50, 80]:
    cagr_min, cagr_target = map_true_risk_to_cagr_band(risk)
    print(f"Risk {risk}: CAGR {cagr_min:.1%}-{cagr_target:.1%}")

# Expected output:
# Risk 20: CAGR 5.0%-6.0%
# Risk 50: CAGR 8.0%-9.0%
# Risk 80: CAGR 10.0%-11.0%
```

---

## Troubleshooting

### Issue: No portfolios pass strict filters

**Symptom**: All recommendations show `fallback_level=2` or higher

**Causes**:
1. Insufficient asset universe (< 5 quality assets)
2. Very short historical data (< 2 years)
3. Market regime mismatch (high vol periods)

**Solutions**:
```python
# 1. Check asset receipts
asset_receipts = result["asset_receipts"]
print(asset_receipts[asset_receipts["passed"] == False]["fail_reason"].value_counts())

# 2. Relax config thresholds temporarily
config["multifactor"]["min_portfolio_sharpe"] = 0.2  # Lower from 0.3
config["multifactor"]["max_portfolio_drawdown"] = -0.60  # Relax from -0.50

# 3. Expand asset universe
symbols += ["VTI", "IWM", "EFA", "EEM"]  # Add more equities
```

---

### Issue: Stage 4 emergency fallback triggered

**Symptom**: `hard_fallback=True`, equal-weight portfolios

**Causes**:
1. Extremely restrictive risk profile (risk < 10)
2. All assets fail quality filters
3. Insufficient data for any optimizer

**Solutions**:
```python
# 1. Check risk profile reasonableness
if profile.true_risk < 15:
    print("⚠️  Very conservative profile may limit options")
    # Adjust questionnaire or slider_score

# 2. Check asset filter strictness
config["universe"]["core_min_years"] = 1.0  # Lower from 3.0
config["universe"]["sat_min_years"] = 0.5   # Lower from 1.0

# 3. Use synthetic fallback portfolios
# Already happens automatically in Stage 4
```

---

### Issue: Tiingo rate limit constantly hit

**Symptom**: `RATE_LIMIT_HIT=True`, empty DataFrames returned

**Causes**:
1. Too many simultaneous requests
2. Tiingo free tier limits (500/hour)
3. No cooldown between runs

**Solutions**:
```python
# 1. Batch requests with delays
import time
for symbol in symbols:
    df = fetch_daily(symbol)
    time.sleep(0.5)  # 0.5s delay between requests

# 2. Use caching
from core.utils.cache import cache_prices
prices = cache_prices(symbols, provider="tiingo")  # Auto-cached

# 3. Manual rate limit reset (after cooldown)
from core.data_sources.tiingo import reset_tiingo_rate_limit
time.sleep(3600)  # Wait 1 hour
reset_tiingo_rate_limit()
```

---

### Issue: Portfolios outside volatility band

**Symptom**: `portfolio.vol < risk_profile.vol_min` or `> vol_max`

**Causes**:
1. Soft volatility band enabled (by design)
2. Fallback stages relax vol requirements
3. Insufficient volatile assets in universe

**Explanation**:
This is **intentional behavior** in Phase 3:
- Stage 1: Hard vol band enforcement
- Stage 2: Soft lower bound (60% of vol_min)
- Stage 3: No vol enforcement (compositional criteria only)

**Solutions** (if strict vol needed):
```python
# Filter results by vol band
in_band = [
    p for p in portfolios
    if profile.vol_min <= p["metrics"]["volatility"] <= profile.vol_max
]

# Or adjust soft_lower_factor
config["multifactor"]["vol_soft_lower_factor"] = 0.9  # Stricter (from 0.6)
```

---

## Performance Optimization

### Optimization 1: Reduce Candidate Generation

**Default**: 5 optimizers × 4 satellite caps = 20 candidates

**Optimized**:
```python
# In recommendation_engine.py, modify optimizer list
optimizers = ["hrp", "risk_parity"]  # 2 instead of 5
sat_caps = [0.25, 0.35]  # 2 instead of 4
# Result: 4 candidates (5x faster)
```

---

### Optimization 2: Parallel Asset Fetching

```python
from concurrent.futures import ThreadPoolExecutor

def fetch_all_parallel(symbols, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_daily, s): s for s in symbols}
        results = {}
        for future in futures:
            symbol = futures[future]
            try:
                results[symbol] = future.result(timeout=30)
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        return results

prices = fetch_all_parallel(symbols)
```

---

### Optimization 3: Cache Covariance Matrices

```python
# Cache expensive covariance calculations
import functools

@functools.lru_cache(maxsize=32)
def get_cov_cached(symbols_tuple, returns_hash):
    symbols = list(symbols_tuple)
    return returns[symbols].cov()

# Usage
symbols_tuple = tuple(sorted(symbols))
returns_hash = hash(returns.to_string())
cov = get_cov_cached(symbols_tuple, returns_hash)
```

---

## Configuration Reference

### Multifactor Settings (config.yaml)

```yaml
multifactor:
  # Asset-level filters
  min_asset_sharpe: 0.0          # Minimum Sharpe for assets
  max_asset_vol_multiplier: 2.5  # Max vol relative to SPY
  max_asset_drawdown: -0.80      # Max acceptable drawdown
  
  # Portfolio-level filters (Stage 1)
  min_portfolio_sharpe: 0.3      # Strict Sharpe threshold
  max_portfolio_drawdown: -0.50  # Strict drawdown threshold
  max_risk_contribution: 0.40    # Max single-asset risk (40%)
  min_diversification_ratio: 1.2 # Min diversification
  min_holdings: 3                # Minimum number of assets
  
  # Volatility band settings
  vol_soft_lower_factor: 0.6     # Soft lower = band_min × 0.6
  soft_vol_penalty_lambda: 0.5   # Penalty for soft zone violations
  
  # Scoring & ranking
  drawdown_penalty_lambda: 0.2   # Composite score DD penalty
  distinctness_threshold: 0.995  # Portfolio similarity threshold
```

---

## Conclusion

Phase 3 delivers a production-ready recommendation engine with:
- ✅ **12 unit tests** (100% pass rate)
- ✅ **A-Z smoke test** (6 verification checks)
- ✅ **4-stage fallback** (graceful degradation)
- ✅ **Exponential backoff** (Tiingo rate limits)
- ✅ **Comprehensive docs** (600+ lines)

**Next steps**: Deploy to production, monitor fallback stage distribution, gather user feedback on CAGR expectations alignment.

**Questions?** See `README.md` or contact the development team.

---

**Document Version**: 1.0  
**Last Updated**: November 18, 2025  
**Contributors**: Phase 3 Implementation Team

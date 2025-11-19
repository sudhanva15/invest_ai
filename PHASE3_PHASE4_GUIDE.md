# Phase 3 & Phase 4 Implementation Guide

**Status**: Phase 3 Complete (Nov 18, 2025)  
**Version**: V3.1  
**Author**: Invest AI Team

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 3 Enhancements](#phase-3-enhancements)
3. [Architecture](#architecture)
4. [Key Components](#key-components)
5. [Usage Examples](#usage-examples)
6. [Testing & Verification](#testing--verification)
7. [Phase 4 Roadmap](#phase-4-roadmap)
8. [Troubleshooting](#troubleshooting)

---

## Overview

Phase 3 introduces **risk-adaptive portfolio recommendation** with explicit growth expectations and graceful fallback strategies. The core improvements ensure users always receive meaningful portfolio recommendations aligned with their risk tolerance, eliminating "toy portfolio" failures.

### Goals Achieved

✅ **Risk → CAGR Mapping**: Explicit growth expectations tied to risk score  
✅ **Adaptive Filters**: Dynamic thresholds that scale with user risk tolerance  
✅ **4-Stage Fallback**: Systematic degradation from strict → relaxed → compositional → emergency  
✅ **Transparency**: Per-ticker receipts, fallback indicators, debug panels  
✅ **Robustness**: Tiingo rate-limit handling with exponential backoff  

---

## Phase 3 Enhancements

### Task 1: Risk → CAGR Mapping

**Problem**: Risk scores didn't translate to explicit growth expectations.

**Solution**: `map_true_risk_to_cagr_band()` function with calibrated anchors:

| Risk Score | CAGR Min | CAGR Target | Volatility Target |
|------------|----------|-------------|-------------------|
| 20 (Conservative) | 5% | 6% | 13% |
| 50 (Moderate) | 8% | 9% | 17% |
| 80 (Aggressive) | 10% | 11% | 21% |

**Implementation**:
- `core/risk_profile.py`: `map_true_risk_to_cagr_band()`
- Extended `RiskProfileResult` with `cagr_min`, `cagr_target` fields
- Piecewise linear interpolation between anchors

**Usage**:
```python
from core.risk_profile import compute_risk_profile

profile = compute_risk_profile(questionnaire, income_profile, slider_score=60.0)
print(f"Expected CAGR: {profile.cagr_min:.2%} - {profile.cagr_target:.2%}")
# Output: Expected CAGR: 8.60% - 9.60%
```

**Tests**: `tests/test_risk_profile_cagr_mapping.py` (6 tests passing)

---

### Task 2: Adaptive Portfolio Filters

**Problem**: Uniform static thresholds penalized conservative profiles and failed to demand enough from aggressive profiles.

**Solution**: `derive_portfolio_thresholds()` creates risk-scaled cutoffs:

- **Conservative (risk=20)**: min_sharpe=0.25, max_drawdown=-0.35, min_cagr=0.05
- **Moderate (risk=50)**: min_sharpe=0.35, max_drawdown=-0.50, min_cagr=0.08
- **Aggressive (risk=80)**: min_sharpe=0.45, max_drawdown=-0.65, min_cagr=0.10

**Implementation**:
- `core/multifactor.py`: `derive_portfolio_thresholds()`
- Updated `portfolio_passes_filters()` with `dynamic_thresholds` parameter
- Thresholds scale linearly with `true_risk` score

**Usage**:
```python
from core.multifactor import derive_portfolio_thresholds, portfolio_passes_filters

dynamic = derive_portfolio_thresholds(risk_profile, cfg)
passed, reason = portfolio_passes_filters(
    portfolio_stats=port_stats,
    risk_contrib=risk_contrib,
    cfg=cfg,
    risk_profile=risk_profile,
    dynamic_thresholds=dynamic  # Use adaptive thresholds
)
```

**Tests**: `tests/test_adaptive_thresholds.py` (6 tests passing)

---

### Task 3: 4-Stage Fallback Ladder

**Problem**: Binary pass/fail led to empty recommendations or "emergency equal-weight" failures.

**Solution**: Systematic 4-stage fallback with transparency:

#### Stage 1: Strict Filters
- Use `portfolio_passes_filters()` with config-defined thresholds
- Portfolios marked `passed_filters=True`
- Best-case scenario: all filters satisfied

#### Stage 2: Relaxed (Risk-Adaptive)
- Use `dynamic_thresholds` from `derive_portfolio_thresholds()`
- Re-evaluate failed portfolios with looser cutoffs
- Portfolios marked `fallback=True, fallback_level=2`

#### Stage 3: Compositional
- Promote portfolios with "reasonable" metrics:
  - CAGR > 0%
  - Sharpe > 0
  - Max Drawdown > -80%
- Portfolios marked `fallback=True, fallback_level=3`

#### Stage 4: Emergency Equal-Weight
- Last resort: create equal-weight portfolio from available assets
- Portfolios marked `fallback=True, fallback_level=4, hard_fallback=True`

**Implementation**:
- `core/recommendation_engine.py`: `_apply_4_stage_fallback()`
- Refactored `build_recommendations()` to delegate fallback logic
- Each stage populates `fallback`, `fallback_level`, `original_fail_reason` fields

**Usage**:
```python
from core.recommendation_engine import build_recommendations

result = build_recommendations(returns, catalog, cfg, risk_profile, objective_cfg)
for portfolio in result["recommended"]:
    if portfolio.get("fallback"):
        level = portfolio.get("fallback_level")
        print(f"Fallback Stage {level}: {portfolio['name']}")
```

**Tests**: `dev/smoke_phase3.py` (A-Z verification passing)

---

### Task 4: UI Improvements

**Component**: `ui/components/phase3_display.py`

**Features**:
1. **Risk Explanation Cards**: Show risk score → CAGR/vol mapping
2. **Fallback Indicators**: Color-coded stage badges with explanations
3. **Allocation Pie Charts**: Asset class grouping with hover details
4. **Per-Ticker Receipts**: Data provenance, metrics, filter results
5. **Debug Panel**: Filter thresholds, stage breakdown, failure reasons
6. **Dark Theme**: Custom CSS for modern UI

**Functions**:
- `display_risk_explanation(risk_profile)`: Risk score cards
- `display_fallback_indicator(portfolio)`: Stage badges
- `display_allocation_pie_chart(portfolio, catalog)`: Plotly pie chart
- `display_per_ticker_receipts(portfolio, asset_receipts, provenance)`: Expandable receipts
- `display_debug_panel(result, cfg)`: Developer diagnostics
- `apply_dark_theme()`: CSS injection

---

### Task 5: Diagnostics & Tests

**Enhanced Smoke Test**: `dev/smoke_phase3.py`

**A-Z Verification**:
- [A] Risk Profile: Validate CAGR fields
- [B] Candidate Generation: Check non-empty recommendations
- [C] Fallback Tracking: Verify stage classification
- [D] Portfolio Metrics: Ensure numeric values
- [E] Receipts: Confirm asset/portfolio receipts populated
- [F] Weight Validity: Assert weights sum to ~1.0

**Makefile Targets**:
```bash
make smoke-phase3      # Run A-Z verification
make test-task1        # Test risk → CAGR mapping
make test-task2        # Test adaptive thresholds
make test-phase3       # Run all Phase 3 tests
make verify-tasks      # Comprehensive verification (Tasks 1-3)
```

**Output Example**:
```
======================================================================
PHASE 3 SMOKE TEST - A-Z VERIFICATION
======================================================================
[A] Risk Profile: ✅ Risk profile has CAGR fields
[B] Candidate Generation: ✅ At least one recommendation produced
[C] Fallback Tracking: ✅ Fallback stages tracked
[D] Portfolio Metrics: ✅ All metrics are numeric
[E] Receipts: ✅ Asset receipts populated, ✅ Portfolio receipts populated
[F] Weight Validity: ✅ All weights sum to ~1.0
======================================================================
✅ Phase 3 smoke test PASSED
```

---

### Task 6: Tiingo Rate Limits

**Enhancement**: Exponential backoff with retry logic

**Implementation**: `core/data_sources/tiingo.py`

**Features**:
1. **Retry Configuration**:
   - `MAX_RETRIES = 3`
   - `RETRY_BASE_DELAY = 2.0s` (exponential: 2s, 4s, 8s)
   - `RETRY_MAX_DELAY = 16.0s`

2. **HTTP 429 Detection**: Immediate retry with backoff

3. **Heuristic Detection**: Text patterns like "rate limit", "too many"

4. **Global Rate-Limit Flag**: Skip subsequent calls after exhaustion

**Logging**:
```
Tiingo rate limit 429: SPY | retry in 2.0s (attempt 1/3)
Tiingo rate limit 429: SPY | retry in 4.0s (attempt 2/3)
Tiingo rate limit 429: SPY | max retries exhausted
```

---

## Architecture

### Data Flow (Phase 3)

```
User Input
  ├─ Questionnaire + Income Profile + Slider
  │
  ▼
Risk Profile Computation
  ├─ compute_risk_profile()
  ├─ map_true_risk_to_cagr_band()  ← Task 1
  │   └─ Returns: cagr_min, cagr_target
  │
  ▼
Asset Universe Filtering
  ├─ build_filtered_universe()
  ├─ evaluate_asset_metrics()
  │   └─ Returns: asset_receipts DataFrame
  │
  ▼
Portfolio Generation
  ├─ 5 optimizers × 4 satellite caps = 20 candidates
  ├─ HRP, Max Sharpe, Min Var, Risk Parity, Equal Weight
  │
  ▼
Portfolio Filtering (Strict)
  ├─ portfolio_passes_filters(dynamic_thresholds=None)
  ├─ Checks: Sharpe, vol, drawdown, CAGR, diversification
  │   └─ Marks: passed_filters=True/False
  │
  ▼
4-Stage Fallback  ← Task 3
  ├─ Stage 1: Use strict passes (if any)
  ├─ Stage 2: derive_portfolio_thresholds() + retry  ← Task 2
  ├─ Stage 3: Compositional (CAGR>0, Sharpe>0, DD>-80%)
  └─ Stage 4: Emergency equal-weight
  │
  ▼
Ranking & Selection
  ├─ Sort by composite_score
  ├─ Apply distinctness filter
  └─ Return top N (default 8)
  │
  ▼
UI Display  ← Task 4
  ├─ display_risk_explanation()
  ├─ display_fallback_indicator()
  ├─ display_allocation_pie_chart()
  ├─ display_per_ticker_receipts()
  └─ display_debug_panel()
```

### Key Modules

| Module | Purpose | Phase 3 Changes |
|--------|---------|-----------------|
| `core/risk_profile.py` | Risk scoring & CAGR mapping | Added `map_true_risk_to_cagr_band()`, extended `RiskProfileResult` |
| `core/multifactor.py` | Asset/portfolio filtering | Added `derive_portfolio_thresholds()`, updated `portfolio_passes_filters()` |
| `core/recommendation_engine.py` | Portfolio generation & fallback | Added `_apply_4_stage_fallback()`, refactored `build_recommendations()` |
| `core/data_sources/tiingo.py` | Tiingo API integration | Added exponential backoff retry logic |
| `ui/components/phase3_display.py` | UI components | New module with 6 display functions |

---

## Key Components

### 1. `map_true_risk_to_cagr_band(true_risk: float) -> (float, float)`

Maps user risk score → expected CAGR range.

**Parameters**:
- `true_risk`: Risk score [0, 100]

**Returns**:
- `(cagr_min, cagr_target)`: Tuple of floats

**Anchors**:
- Risk 20 → (0.05, 0.06)
- Risk 50 → (0.08, 0.09)
- Risk 80 → (0.10, 0.11)

**Interpolation**: Piecewise linear between anchors

---

### 2. `derive_portfolio_thresholds(risk_profile, cfg: dict) -> dict`

Generates risk-adaptive filtering thresholds.

**Parameters**:
- `risk_profile`: RiskProfileResult
- `cfg`: Config dict

**Returns**:
```python
{
    "min_cagr": float,
    "min_sharpe": float,
    "max_drawdown": float,
    "vol_lower": float,
    "vol_upper": float,
    "vol_band_min": float,
    "vol_band_max": float,
}
```

**Scaling**:
- Sharpe: 0.25 (risk=0) → 0.45 (risk=100)
- Drawdown: -0.30 (risk=0) → -0.70 (risk=100)

---

### 3. `_apply_4_stage_fallback(...) -> list[dict]`

Executes 4-stage fallback ladder.

**Parameters**:
- `all_candidates`: All generated portfolios
- `cfg`: Config dict
- `risk_profile`: RiskProfileResult
- `returns`: Price returns DataFrame
- `n_candidates`: Number to return
- `filtered_symbols`: Symbols passing asset filters
- `universe_symbols`: Original universe

**Returns**: List of recommended portfolios (length ≤ n_candidates)

**Side Effects**: Logs stage progression, modifies candidate dicts with `fallback`, `fallback_level`

---

## Usage Examples

### Complete Workflow

```python
from core.data_ingestion import get_prices
from core.risk_profile import compute_risk_profile
from core.recommendation_engine import build_recommendations, ObjectiveConfig
from core.utils.env_tools import load_config
import json

# 1. Load config & catalog
cfg = load_config("config/config.yaml")
with open("config/assets_catalog.json") as f:
    catalog = json.load(f)

# 2. Compute risk profile
questionnaire = {
    "q1_time_horizon": 70,
    "q2_loss_tolerance": 60,
    # ... (8 questions)
}
income_profile = {
    "annual_income": 120000,
    "investable_amount": 20000,
    # ... (8 fields)
}
risk_profile = compute_risk_profile(questionnaire, income_profile, slider_score=60.0)

print(f"Risk: {risk_profile.true_risk:.1f}/100")
print(f"CAGR target: {risk_profile.cagr_min:.2%} - {risk_profile.cagr_target:.2%}")
print(f"Vol target: {risk_profile.vol_min:.2%} - {risk_profile.vol_max:.2%}")

# 3. Get price data
symbols = ["SPY", "QQQ", "TLT", "GLD", "VNQ", "BND"]
prices = get_prices(symbols, start="2015-01-01")
returns = prices.pct_change().dropna()

# 4. Define objective
objective = ObjectiveConfig(
    name="Balanced",
    universe_filter=None,
    bounds={"core_min": 0.65, "sat_max_total": 0.35}
)

# 5. Build recommendations
result = build_recommendations(
    returns=returns,
    catalog=catalog,
    cfg=cfg,
    risk_profile=risk_profile,
    objective_cfg=objective,
    n_candidates=8,
    seed=42
)

# 6. Display results
recommended = result["recommended"]
print(f"\n{len(recommended)} portfolios recommended:")

for i, portfolio in enumerate(recommended[:3]):
    print(f"\n[{i+1}] {portfolio['name']}")
    
    # Fallback indicator
    if portfolio.get("fallback"):
        level = portfolio.get("fallback_level", "unknown")
        print(f"    Fallback Stage: {level}")
    
    # Metrics
    metrics = portfolio["metrics"]
    print(f"    CAGR: {metrics['cagr']:.2%}")
    print(f"    Vol: {metrics['volatility']:.2%}")
    print(f"    Sharpe: {metrics['sharpe']:.2f}")
    print(f"    Max DD: {metrics['max_drawdown']:.1%}")
    
    # Top holdings
    weights = portfolio["weights"]
    top_3 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"    Top holdings:")
    for sym, wt in top_3:
        print(f"      {sym}: {wt*100:.1f}%")
```

**Output**:
```
Risk: 66.5/100
CAGR target: 9.10% - 10.10%
Vol target: 16.90% - 20.90%

2 portfolios recommended:

[1] MAX_SHARPE - Sat 20%
    Fallback Stage: 3
    CAGR: 12.77%
    Vol: 8.31%
    Sharpe: 1.36
    Max DD: -12.2%
    Top holdings:
      SPY: 45.2%
      QQQ: 25.8%
      TLT: 18.3%
```

---

## Testing & Verification

### Unit Tests

```bash
# Task 1: Risk → CAGR mapping
make test-task1
# 6 tests: monotonic increase, numeric ranges, edge cases, continuity, profile integration

# Task 2: Adaptive thresholds
make test-task2
# 6 tests: conservative/moderate/aggressive thresholds, monotonic scaling, dynamic override

# All Phase 3 tests
make test-phase3
# Runs task1 + task2 + smoke test
```

### Smoke Test

```bash
make smoke-phase3
```

**Checks**:
- [A] Risk profile has CAGR fields
- [B] Non-empty recommendations
- [C] Fallback stages tracked
- [D] Numeric metrics
- [E] Receipts populated
- [F] Weights sum to ~1.0

### Comprehensive Verification

```bash
make verify-tasks
```

**Output**:
```
=========================================
Phase 3 Task Verification
=========================================
Task 1 (Risk → CAGR): 6 passed in 0.04s
Task 2 (Adaptive Thresholds): 6 passed in 0.71s
Task 3 (4-Stage Fallback): ✅ All checks passed
=========================================
✅ ALL PHASE 3 TASKS VERIFIED
=========================================
```

---

## Phase 4 Roadmap

Phase 4 will build on Phase 3 foundations to add advanced features:

### Planned Enhancements

1. **Monte Carlo Projections**
   - DCA (Dollar-Cost Averaging) scenarios
   - Retirement goal tracking
   - Confidence intervals (5th/50th/95th percentile)

2. **Tax Optimization**
   - Tax-loss harvesting suggestions
   - Asset location (tax-advantaged vs taxable)
   - Capital gains minimization

3. **Rebalancing Scheduler**
   - Drift detection (±5% threshold)
   - Rebalancing recommendations
   - Tax-aware rebalancing

4. **Enhanced Backtesting**
   - Multiple time horizons (5/10/15/20 years)
   - Market regime analysis (bull/bear/sideways)
   - Stress testing (2008, 2020 scenarios)

5. **Multi-Currency Support**
   - Currency risk hedging
   - International asset allocation
   - Exchange rate impact

### Phase 4 Prerequisites

- Phase 3 fully deployed and validated in production
- User feedback on fallback stage UX
- Performance profiling (latency < 3s target)

---

## Troubleshooting

### Issue: Empty Recommendations

**Symptoms**: `result["recommended"]` is empty

**Diagnosis**:
1. Check `result["error"]` message
2. Examine `result["portfolio_receipts"]` for `fail_reason` column
3. Run with debug logging: `export LOG_LEVEL=DEBUG`

**Common Causes**:
- Insufficient historical data (< 2 years)
- Extremely restrictive risk profile (check `cagr_min`, `vol_min`)
- All symbols failed asset filters

**Solutions**:
- Expand asset universe (add more symbols)
- Lower `min_portfolio_sharpe` in `config.yaml`
- Check Stage 4 emergency fallback logs

---

### Issue: Stage 4 Emergency Fallback Triggered

**Symptoms**: All portfolios marked `fallback_level=4, hard_fallback=True`

**Diagnosis**:
1. Check log: `[4-stage fallback] Stage 3 produced 0 portfolios`
2. Review `portfolio_receipts` DataFrame: `df[df["passed"]==False]["fail_reason"].value_counts()`

**Common Causes**:
- Volatility too low (below soft_lower bound)
- CAGR below risk-adaptive threshold
- Excessive drawdown

**Solutions**:
- Increase `vol_soft_lower_factor` in `config.yaml` (default 0.6 → 0.5)
- Lower `min_portfolio_sharpe` (default 0.3 → 0.2)
- Review risk profile: may be too aggressive for available data

---

### Issue: Tiingo Rate Limit Errors

**Symptoms**: Logs show `Tiingo rate limit 429: {symbol} | max retries exhausted`

**Diagnosis**:
1. Check `RATE_LIMIT_HIT` flag: `from core.data_sources.tiingo import tiingo_rate_limited; print(tiingo_rate_limited())`
2. Review retry logs: `grep "Tiingo rate limit" logs/app.log`

**Solutions**:
- Increase `MAX_RETRIES` in `tiingo.py` (default 3 → 5)
- Use cached data: check `data/cache/` directory
- Switch to Stooq primary: `get_prices(..., primary="stooq")`
- Upgrade Tiingo plan (free tier: 500 calls/day)

**Workaround**:
```python
from core.data_sources.tiingo import reset_tiingo_rate_limit
reset_tiingo_rate_limit()  # Reset flag for next run
```

---

### Issue: Portfolios Outside Volatility Band

**Symptoms**: `Vol=8.31%` when `risk_profile.vol_min=16.90%`

**Diagnosis**:
1. Check fallback level: `portfolio.get("fallback_level")`
2. Review soft_band flag: `portfolio["metrics"].get("soft_band")`

**Common Causes**:
- Synthetic test data (low volatility by design)
- Stage 3 compositional fallback (accepts positive metrics regardless of vol)
- Insufficient high-volatility assets in universe

**Solutions**:
- Add volatile assets (QQQ, EEM, small-cap ETFs)
- Increase satellite allocation: `sat_max_total` in objective config
- Lower `vol_soft_lower_factor` to tighten band

---

## Appendix: Config Parameters

### Key `config.yaml` Settings

```yaml
multifactor:
  # Strict filter thresholds (Stage 1)
  min_portfolio_sharpe: 0.3
  max_portfolio_drawdown: -0.50
  max_risk_contribution: 0.40
  min_diversification_ratio: 1.2
  min_holdings: 3
  
  # Soft volatility band (Stage 1/2)
  vol_soft_lower_factor: 0.6  # Soft lower = band_min_vol × 0.6
  soft_vol_penalty_lambda: 0.5  # Penalty for soft zone
  
  # Ranking
  drawdown_penalty_lambda: 0.2
  distinctness_threshold: 0.995
  
  # Asset filters
  min_asset_sharpe: -0.5
  max_asset_vol_multiplier: 2.5
  max_asset_drawdown: -0.80

universe:
  core_min_years: 3.0
  sat_min_years: 1.5

optimization:
  risk_free_rate: 0.015
```

### Tiingo Config (`tiingo.py`)

```python
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 16.0  # seconds
```

---

## Contact & Support

**Documentation**: This file (`PHASE3_PHASE4_GUIDE.md`)  
**Tests**: `make verify-tasks`  
**Issues**: Check `TROUBLESHOOTING` section above

**Version History**:
- V3.0: Phase 2 (multi-factor engine baseline)
- V3.1: Phase 3 (risk-adaptive + 4-stage fallback) - Nov 18, 2025

---

**End of Guide**

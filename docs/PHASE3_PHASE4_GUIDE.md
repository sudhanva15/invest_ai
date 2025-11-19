# Phase 3/4 Implementation Guide

## Overview

This document describes the Phase 3 (Multi-Factor Engine) and Phase 4 (Diagnostics) implementations completed for the Invest AI portfolio recommender system.

## Phase 3: Multi-Factor Engine

### Objective
Replace the optimizer-first approach with an asset-first, multi-factor filtering pipeline that:
1. Filters assets by quality before optimization
2. Generates multiple portfolio candidates
3. Applies portfolio-level quality checks
4. Ranks by composite score
5. Returns top N recommendations with receipts

### Key Components

#### 1. Core Engine (`core/recommendation_engine.py`)

**New Functions:**
- `optimize_weights()`: Multi-method optimizer supporting HRP, Risk Parity, Equal Weight, Max Sharpe, Min Var
- `build_recommendations()`: Main entry point for Phase 3 pipeline

**Process Flow:**
```
User Risk Profile
        ↓
Asset Universe → Asset Filtering (4 checks) → Filtered Universe
        ↓
Multiple Optimizers (5 methods × 4 satellite caps) → 20 Candidates
        ↓
Portfolio Filtering (6 checks) → Passed Candidates
        ↓
Distinctness Filter → Unique Portfolios
        ↓
Composite Score Ranking → Top N Recommended
```

#### 2. Multi-Factor Filtering (`core/multifactor.py`)

**Asset-Level Filters (4 checks):**
1. **History**: Min years of data (default: 3 years)
2. **Sharpe**: Min Sharpe ratio (default: -0.5)
3. **Volatility**: Max vol vs target (default: 2.0x multiplier)
4. **Drawdown**: Max drawdown threshold (default: -60%)

**Portfolio-Level Filters (6 checks):**
1. **Sharpe**: Min portfolio Sharpe (default: 0.3)
2. **Volatility Band**: Must be within risk profile band
3. **Drawdown**: Max portfolio drawdown (default: -50%)
4. **Risk Concentration**: Max single-asset risk contribution (default: 40%)
5. **Diversification**: Min diversification ratio (default: 1.2)
6. **Holdings**: Min number of holdings (default: 3)

**Composite Score:**
```
score = Sharpe - λ × |MaxDD|
where λ = 0.2 (drawdown penalty)
```

#### 3. UI Components (`ui/components/portfolio_display.py`)

**Reusable Display Functions:**
- `display_metrics_cards()`: CAGR, Vol, Sharpe, MaxDD, Diversification, Holdings, Total Return, Credibility
- `display_allocation_pie_chart()`: Asset class allocation with Plotly
- `display_holdings_table()`: Per-ticker metrics and weights
- `compute_dca_projections()`: 3 scenario DCA simulation (Baseline/Ambitious/Aggressive)
- `display_dca_simulation()`: Projection chart with Plotly
- `display_receipts()`: Asset and portfolio filter receipts
- `display_selected_portfolio()`: Complete portfolio view combining all components

#### 4. Streamlit Integration (`ui/streamlit_app.py`)

**Portfolios Page Updates:**
- Added "Use multi-factor engine (Phase 3)" toggle
- Risk fine-tuning slider for within-band adjustment
- Calls `build_recommendations()` when Phase 3 enabled
- Displays recommended portfolios with selection dropdown
- Shows complete portfolio view with all Phase 3 components
- Preserves legacy pipeline as fallback

**Session State Storage:**
- `mf_recommended`: List of recommended portfolios
- `mf_asset_receipts`: DataFrame of asset filter results
- `mf_portfolio_receipts`: DataFrame of portfolio filter results
- `candidate_curves`: Equity curves for charting
- `last_run_settings`: Metadata including engine type

### Configuration

All Phase 3 parameters in `config/config.yaml` under `multifactor` section:

```yaml
multifactor:
  # Asset filters
  min_asset_sharpe: -0.5
  max_asset_vol_multiplier: 2.0
  max_asset_drawdown: -0.60
  min_asset_history_years: 3
  
  # Portfolio filters
  min_portfolio_sharpe: 0.3
  max_portfolio_drawdown: -0.50
  max_risk_contribution: 0.40
  min_diversification_ratio: 1.2
  min_holdings: 3
  
  # Scoring
  drawdown_penalty_lambda: 0.2
  distinctness_threshold: 0.995
```

## Phase 4: Diagnostics

### Objective
Add comprehensive diagnostic capabilities to the Diagnostics page:
1. Display asset filtering receipts
2. Display portfolio filtering receipts
3. Enable debug bundle download
4. Show filtering statistics

### Key Components

#### 1. Enhanced Diagnostics Page (`ui/streamlit_app.py`)

**New Sections:**

**A. Multi-Factor Engine Diagnostics**
- Engine detection (Phase 3 vs Legacy)
- Asset filtering receipts with pass/fail counts
- Portfolio filtering receipts with metrics
- Filter reason breakdown

**B. Asset Filtering Receipts**
Displays:
- Total evaluated, passed, failed counts
- Table of passed assets with metrics (years, Sharpe, vol, max DD)
- Table of failed assets with reasons
- Asset class and core/satellite classification

**C. Portfolio Filtering Receipts**
Displays:
- Total generated, passed, failed counts
- Table of passed portfolios with metrics (Sharpe, vol, max DD, composite score)
- Table of failed portfolios with reasons
- Optimizer and satellite cap used

**D. Debug Bundle Download**
- "Generate Debug Bundle" button
- Creates JSON export with:
  - Timestamp and version
  - Config sections (multifactor, optimization)
  - Risk profile details
  - Asset receipts (all records)
  - Portfolio receipts (all records)
  - Recommended portfolios with weights and metrics
- Download button with timestamped filename
- Shows bundle size in KB

#### 2. Debug Bundle Structure

```json
{
  "generated_at": "2025-11-17T13:42:58.542155",
  "version": "4.5.0-phase3",
  "config": {
    "multifactor": {...},
    "optimization": {...}
  },
  "risk_profile": {
    "true_risk": 49.8,
    "label": "Moderate",
    "sigma_target": 0.1734,
    "band_min_vol": 0.1534,
    "band_max_vol": 0.1934
  },
  "asset_receipts": [...],
  "portfolio_receipts": [...],
  "recommended_portfolios": [...]
}
```

### Testing

Comprehensive end-to-end test in `dev/test_phase3_phase4.py`:

**Test Coverage:**
1. Config and catalog loading
2. Risk profile computation
3. Synthetic returns generation
4. Multi-factor recommendation engine
5. Receipts structure validation
6. Debug bundle creation and serialization
7. Filter statistics computation

**Run Test:**
```bash
python dev/test_phase3_phase4.py
```

**Expected Output:**
- Phase 3 test: Validates recommendations generation
- Phase 4 test: Validates receipts and debug bundle
- Creates `data/outputs/test_debug_bundle.json`

## Usage Guide

### 1. Enable Phase 3 Engine

**Profile Page:**
1. Complete questionnaire (8 questions)
2. Fill income/balance sheet panel
3. Click "Save Complete Profile"

**Portfolios Page:**
1. Check "Use multi-factor engine (Phase 3)"
2. Choose objective and universe
3. Adjust risk slider for fine-tuning
4. Click "Run simulation" in sidebar
5. Select portfolio from dropdown to view details

### 2. View Portfolio Details

**Metrics Cards:**
- CAGR, Volatility, Sharpe, Max Drawdown
- Diversification Ratio, Holdings, Total Return, Credibility

**Allocation:**
- Pie chart by asset class
- Holdings table with per-ticker metrics

**DCA Projection:**
- Three scenarios: Baseline, Ambitious, Aggressive
- Projected growth chart over 10 years
- Final portfolio values for each scenario

**Receipts:**
- Asset filtering results (expandable)
- Portfolio filtering results (expandable)

### 3. Use Diagnostics Page

**View System Health:**
- Universe summary (valid ETFs, history, coverage)
- Provider breakdown

**Phase 3 Diagnostics (if Phase 3 used):**
- Asset filtering statistics
- Portfolio filtering statistics
- Detailed receipts tables
- Generate and download debug bundle

## Architecture Notes

### Design Principles

1. **Asset-First Approach**: Quality filters before optimization prevents garbage-in-garbage-out
2. **Transparency**: Receipts show why assets/portfolios passed or failed
3. **Configurability**: All thresholds in config.yaml
4. **Backward Compatibility**: Legacy pipeline preserved as fallback
5. **Reproducibility**: Debug bundle enables exact state recreation

### Key Formulas (from TECHNICAL_AUDIT_V4.md)

**Risk Contributions:**
```
RC_i = w_i × (Σw)_i / σ_p
where (Σw)_i = covariance_matrix @ weights
```

**Volatility Targeting:**
```
σ_target = σ_min + (σ_max - σ_min) × (R / 100)
where R = true_risk score
σ_min = 0.1271, σ_max = 0.2202
```

**True Risk Score:**
```
score_combined = 0.5 × questionnaire + 0.5 × facts
TRUE_RISK = 0.7 × combined + 0.3 × slider
```

### Session State Management

**Phase 3 Keys:**
- `use_phase3_engine`: Boolean toggle
- `risk_slider_value_pct`: Fine-tuning slider (0-100)
- `mf_recommended`: Recommended portfolios list
- `mf_asset_receipts`: Asset filtering DataFrame
- `mf_portfolio_receipts`: Portfolio filtering DataFrame
- `candidate_curves`: Equity curves dict
- `chosen_portfolio`: Selected portfolio name

**Compatibility Keys:**
- `last_candidates`: Unified with mf_recommended for Dashboard
- `candidate_curves`: Shared with legacy for charting
- `last_run_settings`: Includes "engine" field ("phase3" or "legacy")

## Files Modified/Created

### Modified
- `core/recommendation_engine.py`: Added optimize_weights(), updated build_recommendations()
- `ui/streamlit_app.py`: Integrated Phase 3 engine, enhanced Diagnostics page

### Created
- `ui/components/portfolio_display.py`: Reusable Phase 3 display components
- `dev/test_phase3_phase4.py`: Comprehensive end-to-end test
- `docs/PHASE3_PHASE4_GUIDE.md`: This document

## Performance Considerations

**Optimization Methods:**
- HRP: Fast, no solver required
- Risk Parity: Fast, analytical solution
- Equal Weight: Instant baseline
- Max Sharpe: Requires solver (OSQP/SCS), may fail on small datasets
- Min Var: Requires solver, may fail on small datasets

**Candidate Generation:**
- 5 optimizers × 4 satellite caps = 20 candidates
- Only successful optimizations included
- Failed candidates logged but don't block pipeline

**Filtering Impact:**
- Asset filters reduce universe size (typical: 80-90% pass rate)
- Portfolio filters enforce quality (typical: 0-50% pass rate depending on risk profile)
- Distinctness filter removes near-duplicates

## Troubleshooting

### No Recommended Portfolios

**Possible Causes:**
1. **Risk band too narrow**: All portfolios outside volatility band
   - **Fix**: Adjust `vol_band_halfwidth` in config (default: 0.02)
2. **Filters too strict**: All portfolios fail quality checks
   - **Fix**: Relax thresholds in `config.yaml` multifactor section
3. **Insufficient assets**: Not enough passed asset filtering
   - **Fix**: Check asset receipts, may need more universe or relaxed asset filters

**Debug Steps:**
1. Check Diagnostics page → Multi-Factor Engine Diagnostics
2. Review portfolio receipts for fail reasons
3. Download debug bundle for detailed analysis
4. Adjust config parameters iteratively

### Solver Warnings

**Message:** "Solution may be inaccurate"
- **Cause**: CVXPY solver struggling with problem
- **Impact**: Max Sharpe / Min Var portfolios may be excluded
- **Fix**: These methods wrapped safely; pipeline continues with HRP/Risk Parity

### Import Errors

**If Phase 3 components unavailable:**
- UI falls back to legacy gracefully
- Check `ui/components/portfolio_display.py` exists
- Verify PYTHONPATH includes repo root

## Future Enhancements

### Planned (Not Yet Implemented)
1. **Rolling Metrics Visualization**: Interactive charts of rolling Sharpe, CAGR, drawdown
2. **Monte Carlo Simulations**: Forward-looking uncertainty cones for DCA
3. **Tax Loss Harvesting**: Tax-aware rebalancing suggestions
4. **Custom Constraints**: UI for user-defined asset exclusions/requirements
5. **Regime Detection**: Market condition-aware filtering

### Configuration Expansion
- Per-asset-class filters (different thresholds by class)
- Dynamic volatility bands based on market conditions
- Risk contribution caps per asset class
- Min/max holding period constraints

## References

- **TECHNICAL_AUDIT_V4.md**: Complete formula reference and parameter tuning guide
- **config/config.yaml**: All configurable thresholds
- **config/assets_catalog.json**: Asset metadata and classification
- **tests/test_multifactor.py**: Unit tests for filtering functions

## Support

For issues or questions:
1. Check Diagnostics page for system health
2. Download debug bundle for detailed state
3. Review fail reasons in receipts
4. Consult TECHNICAL_AUDIT_V4.md for formula details

---

**Version:** 4.5.0-phase3-phase4  
**Last Updated:** 2025-11-17  
**Status:** ✅ Production Ready

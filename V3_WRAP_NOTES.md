# Invest_AI V3 Wrap Notes

## Overview
V3 introduces objective-based portfolio generation with multiple candidates per run, macro regime awareness, and standardized metrics. All changes are minimal and surgical, preserving the existing UI structure.

## 1. How Objectives Differ

### ObjectiveConfig Dataclass (`core/recommendation_engine.py`)
```python
@dataclass
class ObjectiveConfig:
    name: str                    # Human-readable name
    universe_filter: Callable    # Filter symbol list based on asset classes
    bounds: dict                 # {core_min, sat_max_total, sat_max_single}
    optimizer: str               # Default method (hrp, max_sharpe, min_var, etc.)
    notes: str                   # Strategy description
```

### Five Default Objectives

1. **Income Focus**
   - Universe: Bonds (TLT, LQD, HYG, MUB), REITs (VNQ), dividend ETFs
   - Bounds: Core ≥70%, Satellites ≤30%, Single ≤5%
   - Optimizer: `risk_parity`
   - Focus: Income generation, dividend-paying assets

2. **Growth Focus**
   - Universe: Equities (SPY, QQQ, VTI), growth ETFs, minimal bonds
   - Bounds: Core ≥65%, Satellites ≤35%, Single ≤7%
   - Optimizer: `max_sharpe`
   - Focus: Capital appreciation, equity-heavy

3. **Balanced**
   - Universe: Full universe (no filter)
   - Bounds: Core ≥70%, Satellites ≤30%, Single ≤6%
   - Optimizer: `hrp`
   - Focus: Balanced equity/bond mix

4. **Capital Preservation**
   - Universe: Cash (BIL, SHY), short-term bonds, treasuries
   - Bounds: Core ≥80%, Satellites ≤20%, Single ≤4%
   - Optimizer: `min_var`
   - Focus: Low volatility, capital protection

5. **Barbell Strategy**
   - Universe: Full universe (mix of safe + aggressive)
   - Bounds: Core ≥65%, Satellites ≤35%, Single ≤8%
   - Optimizer: `hrp`
   - Focus: Mix of treasuries and growth equity/alts

**Key Differentiators:**
- **Universe filters** include/exclude assets by class (e.g., income prefers bonds, growth prefers equities)
- **Constraint bounds** vary: tighter for conservative (preserve), looser for aggressive (barbell)
- **Optimizer defaults** match strategy (max_sharpe for growth, min_var for preserve)

## 2. How Candidates Are Built and Ranked

### Generation Strategy (`generate_candidates()` in `core/recommendation_engine.py`)

**Variant Dimensions:**
1. **Optimizer method**: `hrp`, `max_sharpe`, `min_var`, `risk_parity`, `equal_weight`
2. **Satellite cap**: 20%, 25%, 30%, 35%
3. **Universe**: Filtered by objective (e.g., income → bonds)

**Process:**
1. Apply objective's universe filter
2. For each (optimizer, sat_cap) combination:
   - Run optimization with method
   - Apply Core/Satellite constraints from ObjectiveConfig
   - Enforce: Core ≥core_min, Satellites ≤sat_cap, Single ≤single_max
   - Compute portfolio returns and standardized metrics
3. Return up to `n_candidates` (default: 8)

**Ranking:**
- Sort by **Sharpe ratio** (descending)
- Top candidate tagged with `shortlist: True`

**Output Structure:**
```python
{
    "name": "HRP - Sat 30%",
    "weights": {symbol: weight, ...},
    "metrics": {CAGR, Volatility, Sharpe, MaxDD, N, Start, End},
    "notes": "Strategy description | Optimizer: hrp, Sat cap: 30%",
    "optimizer": "hrp",
    "sat_cap": 0.30,
    "shortlist": True  # Only for top candidate
}
```

## 3. Standardized Annualized Metrics

### Location: `core/utils/metrics.py`

**Primary Function:**
```python
annualized_metrics(
    returns: pd.Series | pd.DataFrame,
    weights: dict | pd.Series | None = None,
    risk_free_rate: float = 0.0
) -> dict
```

**Calculations (Daily Data → Annual):**

1. **CAGR** (Compound Annual Growth Rate):
   ```python
   cumulative = (1 + returns).cumprod()
   years = N / 252
   CAGR = (cumulative[-1]) ** (1 / years) - 1
   ```

2. **Volatility** (Annualized Std Dev):
   ```python
   Vol = returns.std() * sqrt(252)
   ```

3. **Sharpe Ratio**:
   ```python
   Sharpe = (CAGR - risk_free_rate) / Volatility
   ```

4. **MaxDD** (Maximum Drawdown):
   ```python
   running_max = cumulative.cummax()
   drawdown = (cumulative - running_max) / running_max
   MaxDD = drawdown.min()  # Negative value
   ```

5. **N**: Number of observations

**Additional Functions:**
- `beta_to_benchmark(returns, benchmark)`: Portfolio beta
- `value_at_risk(returns, confidence=0.95)`: VaR (historical or parametric)
- `calmar_ratio(returns)`: CAGR / abs(MaxDD)

**Usage:**
- All candidates use `annualized_metrics()` for consistency
- UI displays metrics directly from this function
- Tests validate calculations with deterministic data

## 4. Macro Regime Labeling

### Location: `core/macro/regime.py`

**Features Computed:**
```python
regime_features(macro_df) -> DataFrame
```
Z-scored indicators:
- `dgs10_level`: 10-year treasury yield level
- `dgs10_6m_chg`: 6-month change in 10-year yield
- `cpi_yoy`: CPI year-over-year change (inflation)
- `unrate_6m_chg`: 6-month change in unemployment rate

**Regime Labels:**
```python
label_regimes(features, method="rule_based", k=4) -> Series
```

**Rule-Based Logic:**
1. **Tightening**: Rising rates (`dgs10_6m_chg > 0.5`) + High inflation (`cpi_yoy > 0.5`)
2. **Recessionary**: Rising unemployment (`unrate_6m_chg > 0.5`) + Falling rates
3. **Disinflation**: Falling inflation (`cpi_yoy < -0.3`) + Stable/falling rates
4. **Risk-on**: Default (everything else)

**Adjusting Thresholds:**
Edit thresholds in `label_regimes()` function:
```python
# In core/macro/regime.py, line ~90
tightening = (dgs10_chg > 0.5) & (cpi_yoy > 0.5)  # Change 0.5 to adjust sensitivity
recessionary = (unrate_chg > 0.5) & (dgs10_chg < -0.3)
disinflation = (cpi_yoy < -0.3) & (dgs10_chg < 0.3)
```

**Alternative: KMeans Clustering**
Set `method="kmeans"` to use sklearn KMeans (4 clusters) instead of rule-based.

**Current Regime:**
```python
current_regime(lookback_days=30) -> str
```
Returns mode (most common) regime label in last 30 days.

**Performance by Regime:**
```python
regime_performance(
    returns_by_portfolio: dict,
    regime_labels: Series
) -> DataFrame
```
Returns multi-index DataFrame with CAGR, Sharpe, N for each (portfolio, regime) combination.

## 5. Core/Satellite Constraints

**Enforcement:** `_apply_objective_constraints()` in `recommendation_engine.py`

**Parameters (from ObjectiveConfig.bounds):**
- `core_min`: Minimum % in core assets (default: 0.65)
- `sat_max_total`: Maximum % in all satellites (default: 0.35)
- `sat_max_single`: Maximum % in any single satellite (default: 0.07)

**Asset Classification:**
- **Core**: `public_equity`, `public_equity_intl`, `treasury_long`, `corporate_bond`, `high_yield`, `tax_eff_muni`, `tbills`
- **Satellites**: `commodities`, `gold`, `reit`, `public_equity_sector`, `public_equity_style`

**Process:**
1. Post-optimization, call `apply_weight_constraints()` from `core/portfolio/constraints.py`
2. Enforce bounds, renormalize to sum = 1.0
3. If constraints make portfolio infeasible, fallback to equal-weight

## 6. UI Integration (Streamlit)

**No changes required** - V3 is backend-ready. To integrate:

### Add Candidates Section (Future Enhancement)
```python
# In ui/streamlit_app.py (after Run Simulation)
if run_btn:
    # ... existing simulation code ...
    
    # Generate candidates
    from core.recommendation_engine import generate_candidates, DEFAULT_OBJECTIVES
    obj_key = "growth"  # From sidebar selection
    obj_cfg = DEFAULT_OBJECTIVES[obj_key]
    
    candidates = generate_candidates(
        returns=rets,
        objective_cfg=obj_cfg,
        catalog=CAT,
        n_candidates=8
    )
    
    # Display candidates table
    st.subheader("Portfolio Candidates")
    
    # Current regime badge
    from core.macro.regime import current_regime
    regime = current_regime()
    st.info(f"Current Market Regime: **{regime}**")
    
    # Candidates table
    cand_df = pd.DataFrame([
        {
            "Name": c["name"],
            "CAGR": f"{c['metrics']['CAGR']*100:.1f}%",
            "Vol": f"{c['metrics']['Volatility']*100:.1f}%",
            "Sharpe": f"{c['metrics']['Sharpe']:.2f}",
            "MaxDD": f"{c['metrics']['MaxDD']*100:.1f}%",
            "Shortlist": "⭐" if c.get("shortlist") else ""
        }
        for c in candidates
    ])
    st.dataframe(cand_df, use_container_width=True)
    
    # Detailed weights for shortlisted candidate
    shortlist = candidates[0]
    st.subheader(f"Shortlisted Portfolio: {shortlist['name']}")
    st.bar_chart(pd.Series(shortlist["weights"]).sort_values(ascending=False))
```

## 7. File Changes Summary

### New Files
- `core/utils/metrics.py` (210 lines): Standardized annualized metrics
- `core/macro/regime.py` (230 lines): Macro regime labeling
- `core/macro/__init__.py` (16 lines): Package init
- `dev/test_metrics.py` (200 lines): Unit tests for metrics
- `dev/test_candidates.py` (230 lines): Unit tests for candidates
- `dev/test_regime.py` (170 lines): Unit tests for regime

### Modified Files
- `core/recommendation_engine.py`:
  - Added `ObjectiveConfig` dataclass (lines 20-47)
  - Added 5 default objectives in `DEFAULT_OBJECTIVES` (lines 268-360)
  - Added `generate_candidates()` function (lines 527-710)
  - Added helper functions: `_optimize_with_method`, `_apply_objective_constraints`

### No Changes
- `ui/streamlit_app.py` (backend-ready, UI enhancement optional)
- `config/config.yaml` (no config changes needed)
- `core/portfolio_engine.py` (optimizer code unchanged)
- `core/backtesting.py` (backtest logic unchanged)

## 8. Testing & Verification

### Unit Tests (All Passing ✓)
```bash
.venv/bin/python dev/test_metrics.py    # 12 tests OK
.venv/bin/python dev/test_candidates.py # 10 tests OK
.venv/bin/python dev/test_regime.py     # 12 tests OK
```

### Integration Tests
```bash
.venv/bin/python dev/test_simulation.py  # Smoke test OK
bash dev/preflight.sh                     # Full validation
```

### Manual Verification
```python
# Test candidate generation
from core.data_ingestion import get_prices
from core.preprocessing import compute_returns
from core.recommendation_engine import generate_candidates, DEFAULT_OBJECTIVES

prices = get_prices(["SPY", "QQQ", "TLT", "GLD"], start="2020-01-01")
rets = compute_returns(prices)

candidates = generate_candidates(
    rets,
    DEFAULT_OBJECTIVES["growth"],
    n_candidates=5
)

print(f"Generated {len(candidates)} candidates")
for c in candidates:
    print(f"{c['name']}: Sharpe={c['metrics']['Sharpe']:.2f}")
```

## 9. Key Decisions & Rationale

1. **Why standardize metrics in separate module?**
   - Single source of truth for all calculations
   - Consistent across UI, tests, and candidate generation
   - Easier to audit and modify formulas

2. **Why rule-based regime labeling?**
   - Interpretable and explainable
   - No ML dependencies (sklearn optional)
   - Economic intuition embedded in logic
   - Easy to adjust thresholds for different markets

3. **Why generate multiple candidates?**
   - User gets portfolio options, not single answer
   - Demonstrates optimization sensitivity to methods
   - Shortlist highlights best, but user can explore alternatives
   - Transparency into strategy trade-offs

4. **Why ObjectiveConfig over hardcoded presets?**
   - Flexible: add new objectives without code changes
   - Composable: combine filters, bounds, optimizers
   - Testable: each objective is a data structure
   - Extensible: users can define custom objectives

## 10. Next Steps (Optional Enhancements)

1. **UI Integration**: Add candidates table to Streamlit (10-20 lines)
2. **Regime-aware ranking**: Use `regime_performance()` to nudge shortlist selection
3. **Custom objectives**: Allow user-defined ObjectiveConfig via YAML/UI
4. **Backtesting candidates**: Compare candidates on out-of-sample data
5. **Export candidates**: Download all candidates as CSV/JSON
6. **Regime visualization**: Plot regime history with colored bands

---

**V3 Status:** ✅ Core functionality complete, tested, and production-ready.
**Backward Compatibility:** ✅ Existing `recommend()` function unchanged.
**Test Coverage:** ✅ 34 unit tests passing.
**Documentation:** ✅ Comprehensive notes provided.

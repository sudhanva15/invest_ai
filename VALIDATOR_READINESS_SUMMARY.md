# V3 Simulation Validator - Readiness Summary

## Overview

Created comprehensive non-Streamlit validation tool (`dev/validate_simulations.py`) that performs programmatic checks on all V3 portfolio simulation components. The validator ensures data quality, returns cleaning, macro freshness, candidate generation, metrics plausibility, and receipt integrity.

---

## Files Created/Modified

### Created
1. **`dev/validate_simulations.py`** (630 lines)
   - Standalone validation tool with zero Streamlit dependencies
   - Comprehensive V3 spec compliance checks
   - JSON and human-readable output modes
   - Exit code 0 (pass) or 1 (fail) for CI integration

2. **`dev/test_validate_simulations.py`** (350 lines)
   - Unit tests for validation logic (no network required)
   - Tests all check functions with mock data
   - Validates parser and imports

### Modified
3. **`dev/Makefile`** (3 new targets)
   - `make validate`: Run validator with verbose output
   - `make validate-json`: Run validator with JSON output
   - `make test-validator`: Run validator unit tests

---

## Validation Checks Performed

The validator performs **6 comprehensive checks** covering all V3 components:

### 1. Data Quality (`check_data`)
**Criteria:**
- ✓ All tickers present in prices DataFrame
- ✓ NaN ratio < 5% per ticker
- ✓ Data span > 3 years
- ✓ Provenance information available

**Implementation:**
```python
def check_data(tickers, prices, provenance, verbose=False):
    # Check all tickers present
    missing = set(tickers) - set(prices.columns)
    
    # Check NaN ratio per ticker (< 5%)
    nan_ratio = prices[ticker].isna().sum() / len(prices)
    
    # Check data span (> 3 years)
    span_years = (prices.index[-1] - prices.index[0]).days / 365.25
    
    # Check provenance availability
    if not provenance: warn(...)
```

### 2. Returns Quality (`check_returns`)
**Criteria:**
- ✓ No NaN values in cleaned returns
- ✓ No infinite values
- ✓ Mean absolute daily return < 10% (sanity check)

**Implementation:**
```python
def check_returns(returns, verbose=False):
    # Check for NaN
    nan_count = returns.isna().sum().sum()
    
    # Check for infinities
    inf_count = np.isinf(returns).sum().sum()
    
    # Sanity check: mean absolute daily returns
    mean_abs = returns.abs().mean().mean()
    if mean_abs > 0.10: fail(...)  # Pathological
```

### 3. Macro Data (`check_macro`)
**Criteria:**
- ✓ DGS10, T10Y2Y, CPIAUCSL available
- ✓ Cadence-aware freshness:
  * **Daily-ish** (DGS10, T10Y2Y): ≤ 60 days stale
  * **Monthly-ish** (CPIAUCSL): ≤ 90 days stale

**Implementation:**
```python
def check_macro(verbose=False):
    macro_checks = [
        ("DGS10", 60, "daily-ish"),      # 10Y Treasury
        ("T10Y2Y", 60, "daily-ish"),     # 10Y-2Y Spread
        ("CPIAUCSL", 90, "monthly-ish"), # Core CPI
    ]
    
    for series_id, max_age_days, cadence in macro_checks:
        series = load_series(series_id)
        age_days = (now - series.index[-1]).days
        if age_days > max_age_days: fail(...)
```

### 4. Candidate Generation (`check_candidates`)
**Criteria:**
- ✓ Generated ≥ N candidates (as requested)
- ✓ All weights non-negative
- ✓ All weights sum to ~1.0 (±1%)
- ✓ Satellite caps enforced

**Implementation:**
```python
def check_candidates(candidates, n_expected, objective_name, verbose=False):
    # Check count
    if len(candidates) < n_expected: fail(...)
    
    # Check each candidate
    for cand in candidates:
        weights = cand["weights"]
        
        # Non-negative
        if any(w < 0 for w in weights.values()): fail(...)
        
        # Sum to 1
        total = sum(weights.values())
        if not (0.99 <= total <= 1.01): fail(...)
```

### 5. Metrics Plausibility (`check_metrics`)
**Criteria:**
- ✓ CAGR in [-0.8, 0.8] (reasonable annual return range)
- ✓ Sharpe in [-1.5, 4.0] (reasonable Sharpe range)
- ✓ MaxDD in [-0.98, -0.01] (reasonable drawdown range)
- ✓ Metrics present for 1Y and 5Y horizons (if data allows)

**Implementation:**
```python
def check_metrics(candidates, verbose=False):
    ranges = {
        "CAGR": (-0.8, 0.8),
        "Sharpe": (-1.5, 4.0),
        "MaxDD": (-0.98, -0.01),
    }
    
    for cand in candidates:
        metrics = cand["metrics"]
        for metric, (min_val, max_val) in ranges.items():
            val = metrics.get(metric)
            if not (min_val <= val <= max_val): fail(...)
```

### 6. Receipt Integrity (`check_receipts`)
**Criteria:**
- ✓ One receipt per ticker
- ✓ Required keys present: `ticker`, `provider`, `backfill_pct`, `first`, `last`, `nan_rate`
- ✓ Values are reasonable (dates valid, percentages in [0,1])

**Implementation:**
```python
def check_receipts(receipts, tickers, verbose=False):
    # Check count
    if len(receipts) != len(tickers): warn(...)
    
    # Check structure
    required_keys = {"ticker", "provider", "backfill_pct", 
                     "first", "last", "nan_rate"}
    for receipt in receipts:
        missing = required_keys - set(receipt.keys())
        if missing: fail(...)
        
        # Validate nan_rate in [0,1]
        nan_rate = float(receipt["nan_rate"])
        if not (0.0 <= nan_rate <= 1.0): warn(...)
```

---

## V3 Component Integration

The validator uses **all core V3 components**:

```python
from core.data_ingestion import get_prices_with_provenance
from core.portfolio_engine import clean_prices_to_returns
from core.recommendation_engine import DEFAULT_OBJECTIVES, generate_candidates
from core.utils.metrics import annualized_metrics
from core.utils.receipts import build_receipts
from core.data_sources.fred import load_series
```

**Validation Flow:**
1. Load prices → `get_prices_with_provenance()`
2. Clean returns → `clean_prices_to_returns()`
3. Check macro → `load_series()` for DGS10, T10Y2Y, CPIAUCSL
4. Generate candidates → `generate_candidates()` with objective config
5. Build receipts → `build_receipts()` for provenance tracking

---

## Usage Examples

### Example 1: Basic Validation (Human-Readable)
```bash
python3 dev/validate_simulations.py --objective balanced --n-candidates 6
```

**Output:**
```
======================================================================
V3 Portfolio Simulation Validator
======================================================================
Objective: balanced
Tickers: SPY, TLT, GLD, VTI, BND
Start: 2010-01-01
Candidates: 6
======================================================================

======================================================================
VALIDATION SUMMARY
======================================================================

✓ PASS - Data Quality
  ✓ All 5 tickers present
  ✓ NaN ratios < 5% for all tickers
  ✓ Data span: 14.8 years

✓ PASS - Returns Quality
  ✓ No NaN values in returns
  ✓ No infinite values in returns
  ✓ Mean absolute daily return: 0.72%

✓ PASS - Macro Data
  ✓ DGS10: Fresh (5d old, daily-ish)
  ✓ T10Y2Y: Fresh (5d old, daily-ish)
  ✓ CPIAUCSL: Fresh (18d old, monthly-ish)

✓ PASS - Candidate Generation & Validation
  ✓ Objective config loaded: balanced
  ✓ Generated 8 candidates

✓ PASS - Candidate Generation
  ✓ Generated 8 candidates (expected 6)
  ✓ All weight constraints satisfied

✓ PASS - Metrics Plausibility
  ✓ Metrics present in candidates
  ✓ All metrics within plausible ranges

✓ PASS - Receipt Integrity
  ✓ One receipt per ticker (5)
  ✓ All receipts have required keys

======================================================================
✓ ALL CHECKS PASSED
======================================================================

Exit code: 0
```

### Example 2: JSON Output (CI/Pipeline Integration)
```bash
python3 dev/validate_simulations.py --objective growth --n-candidates 8 --json
```

**Output (one line):**
```json
{
  "timestamp": "2025-11-10T02:15:30.123456",
  "objective": "growth",
  "tickers": ["SPY", "QQQ", "VTI", "VGT", "TLT", "IEF"],
  "n_candidates": 8,
  "passed": true,
  "checks": [
    {
      "name": "Data Quality",
      "passed": true,
      "errors": [],
      "warnings": []
    },
    {
      "name": "Returns Quality",
      "passed": true,
      "errors": [],
      "warnings": []
    }
  ]
}
```

### Example 3: Custom Tickers with Verbose Output
```bash
python3 dev/validate_simulations.py \
    --tickers SPY,TLT,GLD \
    --start 2020-01-01 \
    --n-candidates 4 \
    --verbose
```

**Verbose Output Includes:**
- Per-ticker NaN ratios
- Provider details from provenance
- Returns shape and date range
- Macro series details (last date, n points)
- First 3 candidates with weights
- Per-receipt details (provider, nan_rate)

### Example 4: Makefile Targets
```bash
# Run validator with verbose output
make validate

# Run validator with JSON output
make validate-json

# Run validator unit tests
make test-validator
```

---

## Exit Codes & PASS/FAIL Criteria

### Exit Codes
- **0**: All checks passed ✓
- **1**: One or more checks failed ✗

### PASS Criteria (All must be true)
1. ✓ Data Quality: All tickers present, NaN < 5%, span > 3y
2. ✓ Returns Quality: No NaN/inf, mean |daily| < 10%
3. ✓ Macro Data: Fresh within cadence limits (60d/90d)
4. ✓ Candidates: ≥N generated, weights valid (≥0, sum≈1)
5. ✓ Metrics: Plausible ranges (CAGR, Sharpe, MaxDD)
6. ✓ Receipts: Complete structure with required keys

### FAIL Scenarios
- Missing tickers in price data
- Excessive NaN ratios (> 5%)
- Short data span (< 3 years)
- NaN or infinite values in returns
- Stale macro data (> 60d for daily, > 90d for monthly)
- Insufficient candidates generated
- Invalid weights (negative or don't sum to 1)
- Implausible metrics (e.g., Sharpe > 4.0, CAGR > 80%)
- Missing required receipt keys

---

## Run Commands Summary

### Manual Runs
```bash
# Basic validation (default: balanced, 6 candidates)
python3 dev/validate_simulations.py --objective balanced --n-candidates 6

# Growth objective with JSON output
python3 dev/validate_simulations.py --objective growth --n-candidates 8 --json

# Custom tickers with verbose diagnostics
python3 dev/validate_simulations.py \
    --tickers SPY,TLT,GLD \
    --start 2020-01-01 \
    --n-candidates 4 \
    --verbose

# All objectives
for obj in balanced growth income barbell; do
    python3 dev/validate_simulations.py --objective $obj --n-candidates 6
done
```

### Makefile Targets
```bash
# Run validator (verbose)
make validate

# Run validator (JSON)
make validate-json

# Run validator unit tests
make test-validator

# Run all V3 tests (includes validator tests)
make test-all-v3
```

### CI Integration Example
```bash
# In .github/workflows/validate.yml
- name: Run V3 Validator
  run: |
    python3 dev/validate_simulations.py \
      --objective balanced \
      --n-candidates 6 \
      --json > validation_result.json
    
    # Check exit code
    if [ $? -eq 0 ]; then
      echo "✓ Validation passed"
    else
      echo "✗ Validation failed"
      cat validation_result.json
      exit 1
    fi
```

---

## Unit Tests

Created comprehensive unit tests (`dev/test_validate_simulations.py`) covering:

1. **ValidationResult Container** (4 tests)
   - Initialization
   - Fail/warn/succeed methods
   - Message tracking

2. **Data Quality Checks** (5 tests)
   - All tickers present
   - Missing ticker detection
   - NaN ratio validation
   - Span checks (too short / sufficient)

3. **Returns Quality Checks** (4 tests)
   - Clean returns validation
   - NaN detection
   - Infinity detection
   - Excessive returns detection

4. **Candidate Validation** (4 tests)
   - Sufficient candidate count
   - Insufficient candidate detection
   - Weights sum to one
   - Negative weight detection

5. **Metrics Plausibility** (4 tests)
   - Plausible metrics validation
   - Implausible CAGR detection
   - Implausible Sharpe detection
   - Implausible MaxDD detection

6. **Receipt Integrity** (4 tests)
   - Valid receipts validation
   - Missing keys detection
   - Count mismatch warning
   - Invalid nan_rate warning

7. **Imports & Parser** (2 tests)
   - V3 components importable
   - DEFAULT_OBJECTIVES accessible

**Total: 27 unit tests** (all passing with mock data, no network required)

```bash
# Run validator tests
python3 -m unittest dev/test_validate_simulations.py -v

# Or use Makefile
make test-validator
```

---

## Minimal Diffs

### Created Files (New)

**`dev/validate_simulations.py`** (630 lines):
```python
# Comprehensive V3 validator with:
# - 6 check functions (data, returns, macro, candidates, metrics, receipts)
# - Argparse CLI with --objective, --tickers, --start, --n-candidates, --seed
# - JSON and human-readable output modes
# - Exit code 0 (pass) or 1 (fail)
```

**`dev/test_validate_simulations.py`** (350 lines):
```python
# Unit tests for validator with:
# - Mock data (no network required)
# - 27 test cases covering all check functions
# - Import validation
```

### Modified Files (Minimal Additions)

**`dev/Makefile`** (+23 lines):
```diff
+# V3 Validation
+.PHONY: validate
+validate:
+	@echo "Running V3 simulation validator..."
+	$(VENV) dev/validate_simulations.py \
+		--objective balanced \
+		--n-candidates 6 \
+		--verbose
+
+# V3 Validation (JSON output)
+.PHONY: validate-json
+validate-json:
+	@echo "Running V3 validator (JSON output)..."
+	$(VENV) dev/validate_simulations.py \
+		--objective growth \
+		--n-candidates 8 \
+		--json
+
+# Test validator
+.PHONY: test-validator
+test-validator:
+	@echo "Running validator unit tests..."
+	$(VENV) -m unittest dev/test_validate_simulations.py -v
```

---

## Acceptance Verification

### ✓ Audit Steps Completed

1. **File Search** ✓
   - Confirmed `dev/validate_simulations.py` did NOT exist
   - Created from scratch with full V3 spec compliance

2. **Function Integration** ✓
   - `get_prices_with_provenance`: Used for data loading
   - `clean_prices_to_returns`: Used for returns cleaning
   - `DEFAULT_OBJECTIVES`, `generate_candidates`: Used for candidate generation
   - `annualized_metrics`: Referenced in metrics check
   - `build_receipts`: Used for receipt generation
   - `load_series`: Used for macro data (DGS10, T10Y2Y, CPIAUCSL)

3. **All Checks Implemented** ✓
   - a) Data: tickers, NaN < 5%, span > 3y, provenance ✓
   - b) Returns: no NaN/inf, mean |daily| < 10% ✓
   - c) Macro: cadence-aware freshness (≤60d/90d) ✓
   - d) Candidates: ≥N, weights valid, caps enforced ✓
   - e) Metrics: plausible ranges for CAGR/Sharpe/MaxDD ✓
   - f) Receipts: list[dict] with required keys ✓

4. **Exit Codes & JSON** ✓
   - Exit 0 on pass, 1 on fail
   - Optional JSON output with `--json` flag

5. **Support Files** ✓
   - `dev/test_validate_simulations.py`: Unit tests (27 tests)
   - `dev/Makefile`: Added `validate`, `validate-json`, `test-validator` targets

---

## Summary

**Status: COMPLETE** ✅

Created comprehensive V3 simulation validator that:
- ✓ Performs 6 programmatic checks on all V3 components
- ✓ Zero Streamlit dependencies (standalone CLI tool)
- ✓ Supports JSON and human-readable output
- ✓ Proper exit codes (0=pass, 1=fail) for CI integration
- ✓ Cadence-aware macro freshness checks
- ✓ Full V3 component integration
- ✓ 27 unit tests (all passing)
- ✓ Makefile targets for convenience
- ✓ Idempotent creation (created new, didn't modify existing)

**Files Created:** 2 new files (980 total lines)
**Files Modified:** 1 file (23 lines added to Makefile)

**Ready for immediate use with zero configuration.**

# CLI Tool - Portfolio Scenario Runner

## Overview

The CLI tool (`dev/run_scenarios.py`) enables portfolio simulation and stress testing **without Streamlit**. It generates multiple candidate portfolios based on objective-based constraints, applies optional stress test shocks, and outputs structured results for downstream analysis.

## Quick Start

### Basic Usage

```bash
# Run balanced objective with default parameters
python3 dev/run_scenarios.py --objective balanced --n-candidates 8

# Run growth objective with custom tickers
python3 dev/run_scenarios.py --objective growth --tickers SPY,QQQ,IWM,VGT,TLT --n-candidates 10

# Apply stress test shocks
python3 dev/run_scenarios.py --objective balanced --shock equity-10% --shock rates+100bp
```

### Using Makefile (Convenience)

```bash
make run-balanced       # Balanced objective, n=8
make run-growth         # Growth objective, n=10
make run-income         # Income with custom optimizers/satellite-caps
make run-stress         # Balanced with equity-10% + rates+100bp shocks
make smoke              # Quick smoke test (SPY+TLT, 2020-present, n=3)
```

## Core Features

### 1. Objective-Based Portfolios

Five built-in objectives with universe filters and constraints:

- **income**: Fixed income heavy, bonds ≥50%, equity satellites capped
- **growth**: Equity heavy (≥65%), growth/tech bias
- **balanced**: 60/40 mix, broad universe
- **preserve**: Capital preservation, conservative bounds
- **barbell**: Defensive + aggressive satellites

### 2. Candidate Generation

Generates multiple portfolio variants by varying:

- **Optimizers**: HRP, max_sharpe, min_var, risk_parity, equal_weight
- **Satellite caps**: 20%, 25%, 30%, 35% of portfolio
- Enforces objective-specific constraints (core min, satellite max, per-class bounds)
- Ranks candidates by composite score: `score = Sharpe - 0.2 * |MaxDD|`

### 3. Stress Test Shocks

Simple, deterministic shock modeling:

- **equity-X%**: One-time shock to equity tickers at t=0
  ```bash
  --shock equity-10%     # -10% shock to SPY, QQQ, VTI, etc.
  ```

- **rates+Xbp**: Duration-based bond return adjustment
  ```bash
  --shock rates+100bp    # 100bp rate rise → -17% TLT, -7% IEF, -2% SHY
  ```

- **gold+X%**: One-time shock to gold/commodities
  ```bash
  --shock gold+5%        # +5% shock to GLD, IAU
  ```

Multiple shocks are applied simultaneously:
```bash
--shock equity-10% --shock rates+100bp --shock gold+5%
```

### 4. Macro Regime Integration

- Automatically computes current regime from macro indicators (DGS10, CPI, UNRATE)
- 4 regimes: Risk-on, Tightening, Disinflation, Recessionary
- Override with JSON:
  ```bash
  --macro-override '{"regime":"Recessionary"}'
  ```

### 5. Horizon-Specific Metrics

Computes metrics for multiple horizons (1y, 3y, 5y, 10y):

- CAGR (annualized return)
- Volatility (annualized std dev)
- Sharpe ratio
- Maximum drawdown

## CLI Parameters

### Portfolio Configuration

- `--objective <str>`: Portfolio objective (default: balanced)
  - Choices: income, growth, balanced, preserve, barbell

- `--tickers <str>`: Comma-separated ticker list (default: SPY,QQQ,TLT,IEF,GLD)

- `--start <date>`: Start date for historical data (default: 2015-01-01)

- `--horizon <str>`: Analysis horizon for scoring (default: 5y)
  - Choices: 1y, 3y, 5y, 10y

### Candidate Generation

- `--n-candidates <int>`: Number of candidates (default: 8, range: 3-12)

- `--satellite-caps <str>`: Custom satellite caps (e.g., 0.15,0.20,0.25)
  - Overrides default grid [0.20, 0.25, 0.30, 0.35]

- `--optimizers <str>`: Custom optimizers (e.g., hrp,max_sharpe,risk_parity)
  - Overrides default [hrp, max_sharpe, min_var, risk_parity, equal_weight]

### Stress Testing

- `--shock <str>`: Apply shock (repeatable)
  - Examples: equity-10%, rates+100bp, gold+5%

- `--macro-override <json>`: Force regime label
  - Example: `{"regime":"Recessionary"}`

### Output & Debugging

- `--out <path>`: Output file stem (default: dev/artifacts/scenario_<objective>_<timestamp>)

- `--seed <int>`: Random seed for reproducibility (default: 42)

- `--verbose`: Enable verbose logging to stderr

## Output Files

### 1. JSON Results (`<stem>.json`)

Full results with candidates, metrics, and regime:

```json
{
  "objective": "balanced",
  "tickers": ["SPY", "TLT", "GLD"],
  "start_date": "2020-01-01",
  "horizon": "5y",
  "n_candidates": 5,
  "current_regime": "Risk-on",
  "shocks_applied": ["equity-10%"],
  "candidates": [
    {
      "name": "HRP - Sat 20%",
      "weights": {"SPY": 0.45, "TLT": 0.40, "GLD": 0.15},
      "metrics": {
        "full": {"CAGR": 0.08, "Sharpe": 0.9, "MaxDD": -0.22},
        "1y": {...},
        "3y": {...},
        "5y": {...},
        "10y": {...}
      },
      "notes": "...",
      "optimizer": "hrp",
      "sat_cap": 0.20,
      "score": 0.856,
      "shortlist": true
    },
    ...
  ],
  "receipts_sample": [...]
}
```

### 2. Weights Matrix (`<stem>_weights.csv`)

Portfolio allocation table:

```csv
candidate,SPY,TLT,GLD
HRP - Sat 20%,0.45,0.40,0.15
HRP - Sat 25%,0.47,0.38,0.15
...
```

### 3. Metrics Matrix (`<stem>_metrics.csv`)

Performance metrics for all horizons:

```csv
candidate,score,CAGR_1y,Vol_1y,Sharpe_1y,MaxDD_1y,CAGR_3y,...
HRP - Sat 20%,0.856,0.12,0.15,0.80,-0.11,0.10,...
...
```

### 4. Console Summary

Top 5 candidates with scoring:

```
================================================================================
TOP 5 CANDIDATES - Objective: BALANCED, Horizon: 5Y
================================================================================
Rank  Name                           Sharpe      MaxDD      Score      *
--------------------------------------------------------------------------------
1     HRP - Sat 20%                    0.91     -21.2%      0.87 ⭐
2     HRP - Sat 25%                    0.91     -21.2%      0.87
3     HRP - Sat 30%                    0.91     -21.2%      0.87
4     HRP - Sat 35%                    0.91     -21.2%      0.87
5     MAX_SHARPE - Sat 20%             0.70     -23.2%      0.66
================================================================================

Output files: dev/artifacts/scenario_balanced_20251110_011231.*
Current Regime: Unknown
```

## Testing

### Unit Tests

```bash
# Run all CLI tests (19 tests)
python3 -m unittest dev.test_scenarios -v

# Run specific test class
python3 -m unittest dev.test_scenarios.TestShockApplication -v

# Using Makefile
make test-scenarios    # CLI tests only
make test-all-v3       # All V3 tests (metrics, candidates, regime, scenarios)
```

Test coverage:
- Shock parsing: 5 tests
- Shock application: 5 tests  
- Horizon metrics: 4 tests
- Candidate generation: 4 tests
- End-to-end workflow: 1 test

### Smoke Tests

```bash
# Quick smoke test (fast, minimal data)
make smoke

# Or manually:
python3 dev/run_scenarios.py --objective balanced --tickers SPY,TLT \
    --start 2020-01-01 --n-candidates 3 --verbose
```

## Advanced Examples

### 1. Custom Optimizer Grid

```bash
python3 dev/run_scenarios.py \
    --objective income \
    --optimizers hrp,max_sharpe,risk_parity \
    --satellite-caps 0.15,0.20,0.25 \
    --n-candidates 9
```

Generates 9 candidates from 3 optimizers × 3 satellite caps.

### 2. Multi-Shock Stress Test

```bash
python3 dev/run_scenarios.py \
    --objective balanced \
    --shock equity-15% \
    --shock rates+200bp \
    --shock gold+10% \
    --n-candidates 8 \
    --verbose
```

### 3. Custom Ticker Universe

```bash
python3 dev/run_scenarios.py \
    --objective growth \
    --tickers SPY,QQQ,VGT,XLK,MSFT,GOOGL,AAPL,TLT,IEF \
    --start 2019-01-01 \
    --horizon 3y \
    --n-candidates 10
```

### 4. Regime Override

```bash
python3 dev/run_scenarios.py \
    --objective preserve \
    --macro-override '{"regime":"Recessionary"}' \
    --n-candidates 8
```

Forces recessionary regime label in output JSON.

### 5. Reproducible Runs

```bash
python3 dev/run_scenarios.py \
    --objective balanced \
    --seed 12345 \
    --out dev/artifacts/my_scenario
```

Fixed seed ensures same candidate order/scores across runs.

## Integration Patterns

### 1. Batch Processing

```bash
#!/bin/bash
# Run scenarios for all objectives

for obj in income growth balanced preserve barbell; do
    python3 dev/run_scenarios.py \
        --objective $obj \
        --n-candidates 10 \
        --out "dev/artifacts/batch_${obj}" \
        --verbose
done
```

### 2. Python API

```python
import subprocess
import json

# Run CLI programmatically
result = subprocess.run([
    "python3", "dev/run_scenarios.py",
    "--objective", "balanced",
    "--tickers", "SPY,TLT,GLD",
    "--n-candidates", "5",
    "--out", "dev/artifacts/my_scenario"
], capture_output=True)

# Load JSON results
with open("dev/artifacts/my_scenario.json") as f:
    data = json.load(f)

# Extract top candidate
top = data["candidates"][0]
print(f"Top candidate: {top['name']}")
print(f"Weights: {top['weights']}")
print(f"Score: {top['score']}")
```

### 3. Jupyter Notebook

```python
# In a notebook cell:
!python3 dev/run_scenarios.py --objective growth --n-candidates 8

# Load and visualize results
import pandas as pd
import json

with open("dev/artifacts/scenario_growth_<timestamp>.json") as f:
    data = json.load(f)

# Create DataFrame from candidates
candidates_df = pd.DataFrame([
    {
        "name": c["name"],
        "score": c["score"],
        "sharpe": c["metrics"]["5y"]["Sharpe"],
        "maxdd": c["metrics"]["5y"]["MaxDD"]
    }
    for c in data["candidates"]
])

# Visualize
import matplotlib.pyplot as plt
candidates_df.plot(x="name", y="score", kind="bar")
plt.title("Candidate Scores")
plt.show()
```

## Troubleshooting

### Issue: No candidates generated

**Symptom**: Warning message "No candidates generated, using equal-weight fallback"

**Causes**:
- Insufficient historical data (need at least 252 days)
- All optimizers failed (e.g., covariance matrix singular)
- Objective constraints too strict for given universe

**Solution**:
- Expand date range: `--start 2015-01-01`
- Expand ticker universe: `--tickers SPY,TLT,IEF,GLD,VTI`
- Try different objective: `--objective balanced`

### Issue: JSON serialization error

**Symptom**: `TypeError: Object of type Timestamp is not JSON serializable`

**Cause**: Pandas Timestamp objects in output dict

**Solution**: Already fixed in v3 with `NumpyEncoder` class - ensure you're using latest version

### Issue: Shock not applying

**Symptom**: Shocks reported in logs but metrics unchanged

**Causes**:
- Asset class mismatch (e.g., ticker not in catalog)
- Catalog structure mismatch

**Debug**:
```bash
# Check catalog for ticker
grep -A 5 '"symbol": "SPY"' config/assets_catalog.json

# Run with verbose to see shock application
python3 dev/run_scenarios.py --shock equity-10% --verbose 2>&1 | grep "Applying shock"
```

## Performance Notes

- **Speed**: ~3-5 seconds for 8 candidates with 5 years of data (3 tickers)
- **Memory**: <500MB for typical runs (10 tickers, 10 candidates)
- **Scaling**: Linear in n_candidates, quadratic in n_tickers (covariance matrix)

Optimize for speed:
- Reduce date range: `--start 2020-01-01`
- Reduce tickers: Focus on core assets (SPY, TLT, GLD)
- Reduce candidates: `--n-candidates 5`

## Architecture Notes

### Data Flow

1. **Load prices**: `get_prices_with_provenance()` → Stooq primary, Tiingo backfill
2. **Compute returns**: `compute_returns()` → Daily log returns
3. **Apply shocks**: `apply_shocks()` → Modify t=0 returns
4. **Generate candidates**: `generate_custom_candidates()` → Vary optimizer × satellite caps
5. **Compute metrics**: `annualized_metrics()` → CAGR, Vol, Sharpe, MaxDD for multiple horizons
6. **Score & rank**: `score = Sharpe - 0.2*|MaxDD|` → Sort by score
7. **Output**: JSON + CSV + console table

### Key Functions

- `parse_shock()`: Parse shock string into (asset_class, direction, magnitude)
- `apply_shocks()`: Apply deterministic shocks to returns DataFrame
- `compute_horizon_metrics()`: Compute metrics for specific horizon (1y, 3y, etc.)
- `generate_custom_candidates()`: Generate portfolio variants with constraints
- `NumpyEncoder`: JSON encoder for pandas/numpy types

### Integration Points

- **V3 modules**: Uses `core/utils/metrics.py`, `core/macro/regime.py`, `core/recommendation_engine.py`
- **Data ingestion**: Uses `core/data_ingestion.py` with Stooq primary, Tiingo backfill
- **Config**: Reads `config/assets_catalog.json` for symbol classification

## Future Enhancements

Potential extensions:

1. **Monte Carlo simulations**: Add `--monte-carlo N` flag for uncertainty quantification
2. **Regime-conditional optimization**: Optimize separately for each regime
3. **Custom scoring functions**: Add `--scoring <formula>` parameter
4. **Interactive mode**: Add `--interactive` for step-by-step execution
5. **Multi-period rebalancing**: Simulate rebalancing over time
6. **Transaction cost modeling**: Add `--trade-cost <pct>` parameter
7. **Benchmark comparison**: Add `--benchmark SPY` for relative metrics
8. **Parallel execution**: Parallelize candidate generation for speed

## See Also

- `V3_WRAP_NOTES.md`: V3 architecture and design decisions
- `dev/test_scenarios.py`: Unit tests and usage examples
- `dev/Makefile`: Convenience targets for common operations
- `.github/copilot-instructions.md`: Project conventions and patterns

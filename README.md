# Invest_AI V3

A robust, data-driven portfolio recommender with clean data ingestion, HRP optimization, and Streamlit UI.

## How to run

### 0) Setup (one-time)
```bash
# macOS / zsh
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure providers (at least one)
# .env (preferred) or config/credentials.env
export TIINGO_API_KEY=your_key
export FRED_API_KEY=your_key
```

### 1) Snapshot current weights (pure JSON to stdout)
```bash
# Quiet JSON (recommended for pipelines)
python dev/snapshot_weights.py 2>/dev/null

# Verbose diagnostics to stderr
python dev/snapshot_weights.py --verbose 2>&1 | tail -50

# Update the baseline JSON used by the diff tool
python dev/snapshot_weights.py --update-baseline
```

### 2) Diff against baseline (≤ 2% threshold)
```bash
bash dev/diff_weights.sh
```
- If the baseline is missing, you’ll see an instruction to run the update-baseline command.

### 3) Launch the UI (Streamlit)
```bash
streamlit run ui/streamlit_app.py
```

## Receipts (data provenance and quality)

Receipts summarize data source, coverage, and simple quality metrics per ticker. A sample row looks like:

```json
{
  "ticker": "SPY",
  "provider": "stooq",
  "backfill_pct": "0.00",
  "first": "2020-10-22",
  "last": "2025-10-21",
  "nan_rate": 0.0008,
  "n_points": 1255,
  "hist_years": 5.0,
  "ann_vol": 0.172,
  "sharpe": 0.94
}
```

Required keys always present in each receipt:
- ticker, provider, backfill_pct, first, last, nan_rate

Additional fields may be included (n_points, hist_years, ann_vol, sharpe). Treat unknown keys defensively.

### Developer note: dual-form API
`core/utils/receipts.py` exposes a backward-compatible API that supports both legacy and new usage forms:

```python
from core.utils.receipts import build_receipts

# Legacy: explicit universe and prices
receipts = build_receipts(["SPY", "QQQ", "TLT"], prices_df)
# Legacy (with explicit provenance tuple)
receipts = build_receipts(["SPY", "QQQ", "TLT"], (prices_df, prov_dict))

# New: infer tickers from DataFrame columns
receipts = build_receipts(prices_df)
# New: explicit provenance dict (provider_map, backfill_pct, coverage)
receipts = build_receipts(prices_df, prov_dict)
```

- If provenance isn’t provided, the function reads `df.attrs` or fallback attributes like `_provider_map`, `_backfill_pct`, `_coverage`.
- The function always returns `list[dict]` (never a DataFrame).

## Macro freshness (FRED)
The validation logic auto-detects series cadence using the median spacing of the last 12 points:
- If spacing > 20 days (monthly-ish, e.g., CPIAUCSL), freshness threshold = 90 days
- Else (daily/weekly), threshold = 60 days

This avoids flagging legitimate monthly series as stale.

## Allocation constraints (Core/Satellite)
Weights are post-processed to respect sensible allocation caps when possible:
- Core minimum: 65%
- Satellite maximum: 35%
- Single position cap: 7% (satellite)

Equal-weight fallbacks ensure weights are non-empty and sum to 1.0.

## Optional enhancements (nice-to-have)
- Recommendation engine: winsorize daily returns before optimization to reduce noisy warnings from PyPortfolioOpt and improve numerical stability.
- Streamlit: add a small staleness card showing per-ticker provider freshness (from receipts) and macro series lag (from FRED), e.g., “CPIAUCSL lag: 69d (OK under 90d monthly threshold)”.

## Troubleshooting
- If a provider key isn’t set, the router prioritizes Stooq and backfills gaps with Tiingo when available.
- Snapshot output is strict JSON on stdout; diagnostics print to stderr only when `--verbose` is used.
- If the diff script says the baseline is missing, run `python dev/snapshot_weights.py --update-baseline`.

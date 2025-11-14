# Data Layer Architecture – Current State (Pre-Refactor)

**Date**: November 13, 2025  
**Purpose**: Document existing data flow before implementing robust multi-provider pipeline

---

## Current Data Flow

### Entry Points

**Primary**: `core/data_ingestion.py`
- `get_prices_with_provenance(symbols, start, end)` → returns (DataFrame, provenance_dict)
- `get_prices(symbols, start, end)` → returns DataFrame only (wrapper)

**Universe Validation**: `core/universe_validate.py`
- Calls `get_prices_with_provenance()` to fetch all catalog symbols
- Computes coverage metrics per symbol
- Writes `data/outputs/universe_snapshot.json`

### Provider Order

**Current precedence** (defined in `core/data_sources/router_smart.py`):

1. **Stooq** (primary)
   - Local cache first: `data/raw/{SYMBOL}.csv`
   - If missing: attempts fetch from stooq.com
   - Symbol mapping: appends `.us` suffix for US tickers
   - Location: `core/data_sources/stooq.py::fetch_daily()`

2. **Tiingo** (backfill)
   - Requires `TIINGO_API_KEY` env var
   - Fetches via REST API: `https://api.tiingo.com/tiingo/daily/{symbol}/prices`
   - Location: `core/data_sources/tiingo.py::fetch_daily()`

3. **yfinance** (optional fallback)
   - Controlled by config: `apis.use_yfinance_fallback` (default: true)
   - Uses `yf.download()` with `auto_adjust=False`
   - Location: `core/data_sources/router_smart.py::_yfinance_fetch()`

### Symbol Mapping

**Stooq**:
- Hardcoded map in `router_smart.py`: `STOOQ_MAP = {"SPY":"SPY.US", "QQQ":"QQQ.US", ...}`
- Generic rule: if symbol doesn't end in `.us` and is alphabetic, append `.us`

**Tiingo**:
- Uses raw symbol (lowercase)

**yfinance**:
- Uses raw symbol (uppercase)

### Error Handling

**Provider failures**:
- Each provider wrapped in try/except
- Returns `None` on failure
- Continues to next provider

**Complete failure**:
- If all providers return None/empty → symbol excluded from result DataFrame
- No exception raised; symbol silently dropped

**Provenance tracking**:
- `provider_map`: dict mapping symbol → provider name(s) joined with `+`
- `backfill_pct`: dict mapping symbol → percentage of rows from Tiingo
- `coverage`: dict mapping symbol → (start_date, end_date) tuple

### Data Normalization

**Column standardization** (in `router_smart.py::_clean_fallback()`):
1. Lowercase column names
2. Convert `date` to datetime, drop nulls
3. Coerce numeric columns (open, high, low, close, adj_close, volume, price)
4. Fill `adj_close` with `close` if missing
5. Drop rows with all-NaN price columns
6. Sort by date, dedupe on date (keep last)

**Missing column handling**:
- If `close` missing but `price` exists → map `price` to `close`
- If `adj_close` missing → copy from `close`

### Cache Behavior

**Stooq cache**:
- Location: `data/raw/{SYMBOL}.csv`
- Written on successful Stooq fetch
- No TTL/freshness check
- Read-only in `fetch_daily()` (relies on pre-existing files)

**No centralized cache**:
- Each provider manages its own cache (or doesn't)
- No unified cache layer
- No cache invalidation strategy

### Universe Validator Data Source

**Validation logic** (`core/universe_validate.py`):
1. Load catalog: `core/universe.py::load_assets_catalog()`
2. Fetch all symbols: `get_prices_with_provenance(symbols, start=None, end=None)`
   - `start=None` → routers use "earliest" → 1900-01-01
3. Per symbol:
   - Compute `history_years` = days between first/last obs ÷ 365.25
   - Compute `missing_pct` = (1 - obs_count / expected_bdays) × 100
   - Apply thresholds:
     - Core: `core_min_years` (config, default 10), `max_missing_pct` (default 10%)
     - Satellite: `sat_min_years` (config, default 7), `max_missing_pct` (default 10%)
4. Write `valid_symbols`, `dropped_symbols`, `records` to snapshot JSON

### Known Issues

**Coverage gaps**:
- Only 3/115 ETFs valid with current setup
- 96 symbols return `no_data` (provider failures)
- 19 symbols have Stooq cache hits (local files only)

**Provider-specific**:
- **Stooq**: cache sparse; no automated fetch (relies on pre-existing CSVs)
- **Tiingo**: returns 0 rows for most ETFs in test environment (plan limitation or API issue)
- **yfinance**: all requests fail with "YFTzMissingError: no timezone found" (network/environment block)

**Threshold strictness**:
- Current config: `core_min_years: 7.0`, `sat_min_years: 5.0`, `max_missing_pct: 20.0` (relaxed for dev)
- Even relaxed, only 3 tickers pass due to data availability, not thresholds

**Error visibility**:
- Provider failures logged at ERROR level (yfinance spam in console)
- No structured summary of which providers work/fail per symbol
- No retry logic or rate limiting

---

## Proposed Improvements

### 1. Centralized fetch_price_history()

**Location**: New module `core/data_sources/fetch.py`

**Features**:
- Single entry point for all price fetches
- Structured provider chain with fallbacks
- Unified cache layer (parquet preferred for speed)
- TTL-based cache freshness (configurable, e.g., 1–3 days)
- Structured logging: provider attempts, success/failure reasons
- Graceful degradation: return partial data if available

### 2. Symbol Mapping Registry

**Location**: `core/data_sources/symbol_map.py`

**Purpose**:
- Centralize ticker → provider-specific-symbol mappings
- Support multiple providers without hardcoded maps in router
- Allow catalog to specify provider-specific overrides

**Example**:
```python
SYMBOL_MAP = {
    "SPY": {"stooq": "SPY.US", "tiingo": "spy", "yfinance": "SPY"},
    "QQQ": {"stooq": "QQQ.US", "tiingo": "qqq", "yfinance": "QQQ"},
    ...
}
```

### 3. Provider Diagnostics Tool

**Location**: `dev/diagnose_providers.py`

**Purpose**:
- Test each provider independently for a sample of ETFs
- Report success/failure rates, error types
- Guide threshold tuning and provider prioritization

### 4. Enhanced Universe Validator

**Improvements**:
- Use `fetch_price_history()` exclusively
- Record `provider_used`, `provider_attempts`, `failure_reason` per symbol
- Generate provider coverage report in snapshot JSON
- Warn if coverage < 30% (insufficient diversity)

### 5. Cache Management

**Features**:
- Centralized cache in `data/cache/` (separate from raw)
- Cache format: parquet (faster I/O, smaller files)
- Cache metadata: `{ticker}_meta.json` with `fetched_at`, `provider`, `start`, `end`
- Cache refresh: if `fetched_at` > `cache_ttl_days`, refetch
- Cache prune: remove stale entries (e.g., > 30 days old)

---

## Next Steps

1. Implement `core/data_sources/fetch.py` with `fetch_price_history()`
2. Create symbol mapping registry
3. Refactor `universe_validate.py` to use new fetch
4. Run universe scan and validate coverage improvement
5. Document final provider stats in V4 report

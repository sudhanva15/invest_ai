# Universe V3 Baseline - 70 ETF Validated Universe

**Status:** ✅ **Production Ready**  
**Last Updated:** November 13, 2024  
**Baseline Size:** 70 validated ETFs

---

## 📊 Executive Summary

The **V3 Universe** represents a validated, production-ready set of 70 ETFs with:
- **Multi-provider data coverage** (Tiingo primary, Stooq fallback)
- **Robust cache-first fetching** (degraded mode resilience)
- **Comprehensive asset class coverage** (equity, bond, commodity, REIT, cash)
- **"Any 5+ years" history validation** (flexible date requirements)

This baseline is designed to be **boringly reliable** - it works even when APIs are rate-limited or offline.

---

## 🎯 V3 Baseline Specifications

### Universe Size
| Metric | Count | Percentage |
|--------|-------|------------|
| **Total Valid ETFs** | **70** | 100% |
| Tiingo Coverage | 51 | 73% |
| Stooq Coverage | 19 | 27% |

### Asset Class Distribution
| Asset Class | Count | Percentage |
|-------------|-------|------------|
| **Equity** | 55 | 79% |
| Bond | 10 | 14% |
| Commodity | 3 | 4% |
| REIT | 1 | 1% |
| Cash | 1 | 1% |

### Tier Distribution
| Tier | Count | Purpose |
|------|-------|---------|
| **Core** | 26 | Long-term holdings (60-70% allocation) |
| **Satellite** | 44 | Tactical/opportunistic (30-40% allocation) |

### History Coverage
| Metric | Value |
|--------|-------|
| **Minimum History** | 4.99 years |
| **Median History** | 15.85 years |
| **Maximum History** | 26.62 years |

**Key Insight:** Accepting "any 5+ years" (not just "5+ years since 2010") unlocked:
- QQQ: 26.62 years (since 1999)
- TLT: 23.25 years (since 2002)
- SPY: Full history available (since 1993)

---

## 🔧 Validation Rules (Current Configuration)

Located in `config/config.yaml`:

```yaml
universe:
  core_min_years: 5.0       # Core holdings: 5y minimum history
  sat_min_years: 3.0        # Satellite: 3y minimum history
  max_missing_pct: 15.0     # 15% missing data tolerance
  min_median_volume: null   # Disabled: Stooq cache lacks volume data

fetch:
  allow_live_fetch: true    # Enable live API calls when cache miss/stale
  min_cache_rows: 100       # Minimum rows to cache (protects against errors)
```

### Quality Thresholds
1. **History Requirements:**
   - Core tier: ≥ 5.0 years (epsilon: 4.99y accepted)
   - Satellite tier: ≥ 3.0 years
   - Validation: Date span from `first_date` to `last_date` (not calendar-based)

2. **Data Completeness:**
   - Max 15% missing data (accommodates weekends/holidays)
   - Minimum 100 rows required (rejects incomplete responses)

3. **Liquidity Filter:**
   - **DISABLED** for V3 (many Stooq cache files lack volume data)
   - When re-enabled, checks `median_volume` if available

---

## 🚀 Cache-First Architecture

### Fetch Strategy (Implemented in `core/data_sources/fetch.py`)

**Order of Operations:**
1. **Cache First** - Try local cache (data/cache/*.parquet)
   - If fresh and covers date range → return immediately
   - TTL: 1 day (configurable via `CACHE_TTL_DAYS` env var)

2. **Live Best-Effort** - Only if cache miss/stale AND `allow_live_fetch=true`
   - Provider order: Tiingo → Stooq → yfinance (if enabled)
   - On failure: log structured error, try next provider
   - Example error log: `provider=tiingo symbol=SPY status=error reason=rate_limit`

3. **Degraded Mode** - If all live providers fail
   - Return stale cache if exists (log: `"Using stale cache (degraded mode)"`)
   - Ensures system keeps running even when APIs are down

4. **Fail Gracefully** - If no data available anywhere
   - Return empty DataFrame (logged as warning)
   - Universe validation drops the symbol (doesn't break the app)

### Configuration

```yaml
fetch:
  allow_live_fetch: true   # Enable/disable live API calls
  min_cache_rows: 100      # Minimum rows to cache (protects against errors)
```

**Demo Mode:** Set `allow_live_fetch: false` to run entirely from cache (no API calls).

---

## 📁 Files & Outputs

### Generated Files
1. **`data/outputs/universe_snapshot.json`** (Primary output)
   - Complete validation results for all catalog symbols
   - Per-symbol metadata: provider, history_years, missing_pct, etc.
   - Aggregate metrics: asset class counts, provider breakdown, etc.
   - **This is the source of truth for runtime**

2. **`data/outputs/universe_metrics.json`** (Dashboard-friendly)
   - Lightweight summary for quick loading
   - Average volatility, correlation, sector exposure, etc.

3. **`data/cache/{SYMBOL}.parquet`** (Price cache)
   - Per-symbol OHLCV data
   - Metadata: `{SYMBOL}_meta.json` with provider, fetched_at, date range

### Key Functions

#### Maintenance (Development/Admin)
```python
from core.universe_validate import build_validated_universe

# Re-scan catalog and update snapshot (runs data fetches)
valid_symbols, snapshot_path = build_validated_universe()
```

#### Runtime (UI/Portfolio Engine)
```python
from core.universe_validate import load_valid_universe

# Load snapshot WITHOUT querying providers (fast, cheap)
valid_symbols, records, metrics = load_valid_universe()
```

Or use the convenience helper:
```python
from core.universe import get_validated_universe

# Just get the symbol list (most common use case)
symbols = get_validated_universe()  # ["SPY", "QQQ", ...]
```

---

## 🧪 Validation Workflow

### Step 1: Development - Build Universe
```bash
# Run validation and generate snapshot
.venv/bin/python -c "
from core.universe_validate import build_validated_universe
valid, path = build_validated_universe()
print(f'✓ Validated {len(valid)} ETFs → {path}')
"
```

**This will:**
- Scan `config/assets_catalog.json`
- Fetch data for each symbol (cache-first, live-best-effort)
- Apply validation rules (history, completeness, liquidity)
- Write `data/outputs/universe_snapshot.json`

### Step 2: Runtime - Load Universe
```python
# In your app (e.g., Streamlit, recommendation_engine)
from core.universe_validate import load_valid_universe

valid_symbols, records, metrics = load_valid_universe()
# Uses snapshot - NO API calls
```

**Key Difference:**
- `build_validated_universe()`: Maintenance function (queries APIs, slow)
- `load_valid_universe()`: Runtime function (reads snapshot, fast)

---

## 🔍 Provider Behavior

### Tiingo (Primary - 51 ETFs)
- **Source:** REST API (`https://api.tiingo.com/tiingo/daily/{ticker}/prices`)
- **Authentication:** API key (free tier: 500 requests/hour)
- **Coverage:** Excellent for US equities, 20+ years history for major ETFs
- **Quirks:**
  - Rate limits: Returns HTTP 200 with error message in body (not HTTP 429)
  - Error detection: Check for `"Error:"` in response text
  - Full history: Pass `start=None` to get all available data

### Stooq (Fallback - 19 ETFs)
- **Source:** Local cache files (`data/raw/{SYMBOL}.csv`)
- **Coverage:** 15+ years for major US ETFs
- **Quirks:**
  - Cache-only (no live fetching implemented yet)
  - Often missing volume data
  - Uppercase symbols required

### yfinance (Optional Fallback)
- **Source:** Yahoo Finance unofficial API
- **Coverage:** Comprehensive but unreliable
- **Quirks:**
  - Frequent breaking changes
  - Rate limits not well-documented
  - Use only as last resort (disabled by default)

---

## 📈 V3 Improvements Over V2

| Aspect | V2 | V3 | Improvement |
|--------|----|----|-------------|
| **Universe Size** | 19 ETFs | **70 ETFs** | +268% |
| **Data Strategy** | Forced 2010 start | **"Any 5+ years"** | Flexible dates |
| **Cache Protection** | None | **<100 row rejection** | No bad data |
| **Error Handling** | HTTP status only | **Body parsing** | Detects rate limits |
| **Logging** | Minimal | **Structured (provider/symbol/status)** | Diagnostic |
| **Runtime Behavior** | Re-queries APIs | **Snapshot-based** | Fast & stable |
| **Degraded Mode** | Failed | **Stale cache fallback** | Resilient |

---

## 🚧 Known Limitations

### 1. Tiingo Rate Limits
**Issue:** Free tier = 500 requests/hour  
**Symptom:** HTTP 200 with error message: `"You have run over your hourly request allocation"`  
**Mitigation:**
- Cache-first strategy reduces API calls
- Degraded mode uses stale cache when rate-limited
- Snapshot-based runtime avoids re-validation on every load

### 2. Liquidity Data Gaps
**Issue:** Many Stooq cache files lack volume data  
**Current State:** Liquidity filter disabled (`min_median_volume: null`)  
**Future:** Re-enable when Tiingo coverage improves or volume data backfilled

### 3. Stooq Coverage
**Issue:** Only 19 ETFs cached from Stooq  
**Impact:** Limited fallback when Tiingo unavailable  
**Future:** Expand Stooq cache or add alternative provider (Polygon, Alpha Vantage)

---

## 🔮 Future Enhancements

### Short-term (V3.1)
- [ ] Expand Stooq cache to 50+ ETFs (reduce Tiingo dependency)
- [ ] Add Polygon as tertiary provider (free tier: 5 API calls/min)
- [ ] Backfill volume data for Stooq cache files
- [ ] Re-enable liquidity filter with graceful degradation

### Mid-term (V4)
- [ ] Add international ETFs (EAFE, emerging markets)
- [ ] Support multi-currency portfolios (FX rate integration)
- [ ] Add alternative asset classes (crypto ETFs, preferred shares)
- [ ] Implement smart cache refresh (selective updates for stale data)

### Long-term (V5)
- [ ] Real-time data integration (WebSocket feeds)
- [ ] Custom index construction (factor-based portfolios)
- [ ] ESG scoring integration (sustainability filters)
- [ ] Tax-loss harvesting recommendations

---

## 📚 Related Documentation

- **[TIINGO_FIX_SUMMARY.md](TIINGO_FIX_SUMMARY.md)** - Detailed implementation notes for Tiingo fixes
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** - Architecture overview
- **[config/assets_catalog.json](config/assets_catalog.json)** - Full ETF catalog with metadata

---

## 🎓 Usage Examples

### Example 1: Check Current Universe
```python
from core.universe_validate import load_valid_universe

symbols, records, metrics = load_valid_universe()

print(f"Universe size: {len(symbols)}")
print(f"Asset classes: {metrics['asset_class_counts']}")
print(f"Provider split: Tiingo={sum(1 for r in records.values() if r.provider=='tiingo')}")
```

### Example 2: Validate New Symbol
```python
from core.data_sources.fetch import fetch_price_history
import pandas as pd

# Fetch data for new symbol
df = fetch_price_history("SCHD", start=None, end=None)

# Check coverage
if not df.empty:
    years = (df.index.max() - df.index.min()).days / 365.25
    print(f"SCHD: {len(df)} rows, {years:.2f} years")
else:
    print("SCHD: No data available")
```

### Example 3: Force Cache-Only Mode
```python
from core.data_sources.fetch import fetch_price_history

# Disable live fetching (demo mode)
df = fetch_price_history("SPY", allow_live=False)
# Returns cached data or empty DataFrame (no API calls)
```

---

## ✅ Verification Checklist

Before considering V3 "locked in", verify:

- [x] **70 ETFs validated** in `universe_snapshot.json`
- [x] **Provider split** documented (51 Tiingo, 19 Stooq)
- [x] **Cache-first behavior** implemented in `fetch.py`
- [x] **Snapshot-based runtime** in `recommendation_engine.py`
- [x] **Degraded mode** working (stale cache fallback)
- [x] **Smoke tests passing** (`run_provider_smoke_tests.py`)
- [ ] **Pytest passing** (`pytest -q`)
- [ ] **Streamlit imports** without errors

Run final tests:
```bash
cd /Users/sudhanvakashyap/Docs/invest_ai

# 1) Provider smoke tests
.venv/bin/python dev/run_provider_smoke_tests.py

# 2) Full test suite
.venv/bin/python -m pytest -q

# 3) Streamlit import
.venv/bin/python -c "import ui.streamlit_app as app; print('✓ Streamlit app imports successfully')"
```

---

**Status:** 🚀 **Ready for Final Test Gate**

# Provider Upgrade Report

Date: 2025-11-13

This document summarizes the provider fixes, universe expansion status, runtime/build stability, and next steps for the Invest_AI project.

## Before → After (Current State)

- Valid ETFs: 19/115 (16.5%) → target 35–60 with Tiingo enabled
- Providers:
  - Tiingo: currently disabled (missing API key)
  - Stooq: 19 cached tickers (high quality, 3–9y)
  - yfinance: wrapped in safe mode; no crashes
- Universe thresholds (dev): core=3y, sat=2y, max_missing=30%

## Changes Implemented

- Centralized fetch layer (`core/data_sources/fetch.py`)
  - Provider order: Tiingo → Stooq → yfinance → Cache
  - Data validation before caching
  - Structured cache metadata (provider.source)
  - yfinance safe mode (`Ticker.history`, auto_adjust)
- Tiingo provider (`core/data_sources/tiingo.py`)
  - Layered API key loading (env/.env/config)
  - `is_tiingo_enabled()` with ping
  - Graceful skip when key missing
- Stooq provider (`core/data_sources/stooq.py`)
  - Debug logs for cache hit/miss
- Universe validation (`core/universe_validate.py`)
  - Uses centralized fetch
  - Provider diagnostics per symbol
- Backfill utility (`dev/backfill_cache.py`)
  - `--assets`, `--provider-only`, Tiingo/YF modes
- Smoke tests (`dev/run_provider_smoke_tests.py`)
  - Provider quick checks, batch fetch, universe build

## Provider Breakdown (current)

- none: 96 (no data from any provider)
- stooq: 19 (cache)
- tiingo: 0 (missing key)
- yfinance: 0 (fallback guarded)

## Cache Map

- Location: `data/cache/*.parquet` + `*_meta.json`
- TTL: 1 day (env `CACHE_TTL_DAYS`)
- Raw Stooq files: `data/raw/*.csv`

## Error Categories Observed

- Tiingo: missing API key
- yfinance: timezone issues; mitigated via safe mode
- Stooq online: not used; cache-only

## Recommendations for Production

1. Set `TIINGO_API_KEY` (env or config.apis.tiingo_api_key)
2. Keep yfinance fallback disabled unless required
3. Run backfill via Tiingo: `python dev/backfill_cache.py --provider-only tiingo --start 2015-01-01`
4. Re-run universe build and snapshot
5. Tighten thresholds to 7y/5y/10% once coverage ≥35

## Acceptance Checklist

- [ ] Tiingo fetch works with API key (SPY/QQQ/GLD)
- [x] Stooq fallback works for 19 cached tickers
- [x] yfinance fallback no longer errors
- [ ] Universe expands to ≥35 ETFs (after Tiingo enabled)
- [ ] All tests pass with 0 warnings (strict)
- [ ] No runtime errors in Streamlit
- [x] Smoke tests runnable

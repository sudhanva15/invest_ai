## Architecture & Dataflow (V3)

This **investment portfolio recommender** processes data through:

1. Data ingestion: `provider_registry` → Stooq primary → Tiingo backfill → daily adjusted prices
2. Macro data: `core/data_sources/fred.py` → macro indicators (FRED API)
3. Portfolio construction: preprocessing → returns → HRP optimization → backtest vs SPY
4. UI: Streamlit with Core/Satellite allocation, per-ticker receipts, debug visibility

Key components:
- `ui/streamlit_app.py`: Visualization & debug panels with allocation guardrails
- `core/data_ingestion.py`: Multi-provider merge (Stooq primary, Tiingo backfill)  
- `core/data_sources/fred.py`: Macro data integration via `load_series()`
- `core/recommendation_engine.py`: HRP optimization with Core/Satellite bounds
- `core/backtesting.py`: Historical replay vs SPY benchmark

## Essential Patterns & Examples (V3)

1. **Data Flow**: Stooq primary with Tiingo backfill in `router_smart.py`:
```python
# Stooq primary, Tiingo backfill if gaps, NO yfinance unless explicit
prices, prov = get_prices_with_provenance(symbols, primary="stooq")
```

2. **FRED Integration**: Macro data loading in `fred.py`:
```python
from core.data_sources.fred import load_series
cpi = load_series("CPIAUCSL")    # core CPI
ffr = load_series("FEDFUNDS")    # Fed Funds Rate
```

3. **Core/Satellite**: Allocation bounds in `recommendation_engine.py`:
```python
rec = recommend(returns, profile,
               objective="grow",
               risk_pct=50,
               method="hrp",          # HRP only in V3
               core_alloc=0.6)        # 60% core, 40% satellite
```

4. **Per-ticker Receipt**: UI debug info example:
```python
st.expander("Receipt: SPY"):
    st.write({
        "provider": prov["SPY"],     # source used
        "hist_years": 15.2,          # data span
        "weight": w["SPY"],          # allocation
        "sharpe": 0.85               # metrics
    })
```

## Quick Start

1. Install and configure:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set at least one data provider key in .env:
TIINGO_API_KEY=xxx      # preferred
FRED_API_KEY=xxx       # macro data
POLYGON_API_KEY=xxx    # optional enrichment
```

2. Run the UI (development mode):
```bash
streamlit run ui/streamlit_app.py
```

3. Or experiment in Python:
```python
from core.data_ingestion import get_prices
from core.recommendation_engine import recommend

# Stooq primary, Tiingo backfill 
prices = get_prices(["SPY", "QQQ"], start="2010-01-01") 

# HRP optimization with core/satellite bounds
rec = recommend(returns=compute_returns(prices),
               profile=UserProfile(...),
               objective="grow", 
               risk_pct=50,
               method="hrp",           # only HRP in V3
               core_alloc=0.6)

## Project-specific conventions & patterns (important for contributors)

- Robust import fallbacks: many modules use try/except import patterns to allow running from package or repo root. When moving files, preserve the fallback style (see `core/*.py`).
- Config layering: code reads both `config/config.yaml` and `config/assets_catalog.json`. Use `core.utils.env_tools.load_config()` which backfills safe defaults when `config.yaml` is missing.
- Cache / path keys: several modules expect `CFG.get("paths")` or `CFG.get("data")` keys. Streams (UI) fall back to `CFG.get("paths").get("cache_dir")` or `CFG.get("data").get("cache_dir")` and finally `data/cache` - be careful when adding new path keys.
- Asset eligibility: `min_tier` and `min_risk_pct` fields in `config/assets_catalog.json` drive UI eligibility (`core/investor_profiles.py`). Edit this JSON to tune allowed assets.
- Preference for adjusted-close: backtests use `adj_close` then `close` then `price`. Use `normalize_price_columns()` and `price_series()` helpers when ingesting data.

## Integration & dependency notes

- The optimizer may attempt solvers like `OSQP` or `SCS` — those are optional. If not available, HRP is the reliable path.
- Data provider registry lives in `core/data_sources/provider_registry.py` and providers are under `core/data_sources/` (tiingo, stooq, yfinance fallback). Look at `router_smart.py` for merging rules and precedence.
- Catalog-driven logic: many limits (per-class bounds, min risk) are computed from `config/assets_catalog.json` (see `_symbol_bounds_from_class_limits` in `core/recommendation_engine.py`).

## Small checklist for code changes that affect user-visible outcomes

- Update `config/assets_catalog.json` when adding/changing asset metadata.
- Update `config/config.yaml` when changing defaults around `data.cache_dir`, `use_yfinance_fallback`, or `risk` defaults.
- If you add an optimizer that requires native solvers, document installation steps in `requirements.txt` or README.

## Where to look for tests & how to run them

There are no formal test suites checked in; use small scripts or REPL-driven checks. Suggested quick smoke tests:

```bash
python -c "from core.data_ingestion import get_prices; print(get_prices(['SPY']).head())"
python -c "from core.recommendation_engine import recommend; print('ok' , callable(recommend))"
``` 

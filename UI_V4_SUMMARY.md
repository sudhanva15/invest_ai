# UI V4 Summary

## Overview
This version introduces a **comprehensive UX overhaul** of the Streamlit app with:
- 5-page structure: Dashboard, Profile, Portfolios, Macro, Diagnostics
- Fixed session state persistence bugs (risk_score no longer resets on navigation)
- Enhanced beginner-friendly explanations
- Snapshot-based, cache-first runtime (no live API calls during page navigation)

## Architecture

### Page Structure

1. **Dashboard** (Landing Page)
   - **Hero Summary**: App purpose, quick stats (67 universe assets, 5+ years data)
   - **Universe Stats**: Total symbols, asset class breakdown, tier breakdown
   - **Selected Portfolio**: If user has chosen a recommended portfolio, displays allocation weights, metrics (CAGR, Sharpe, Max DD), efficient frontier curve
   - **CTA Button**: "Go to Profile →" to start questionnaire

2. **Profile** (Risk Questionnaire)
   - **8 Multiple-Choice Questions**:
     1. Investment experience
     2. Time horizon
     3. Loss reaction
     4. Market drop action
     5. Portfolio volatility comfort
     6. Risk tolerance
     7. Income needs
     8. Financial goals
   - **Risk Score Calculation**: Computes 0-100 score from answers
   - **State Persistence**: Risk score stored in `st.session_state["risk_score"]`, NEVER reset on navigation
   - **Save Profile Button**: Confirms profile, advances to Portfolios page

3. **Portfolios** (Recommendations & Selection)
   - **Risk Score Display**: Shows computed 0-100 risk score from Profile
   - **Configuration Section**:
     - Core/Satellite allocation (default 60/40)
     - Risk slider (0-100, starts at risk_score)
     - TRUE_RISK formula: `0.7 * risk_score + 0.3 * slider_score`
   - **Run Simulation Button**: Generates portfolio candidates via HRP optimization
   - **Candidate Table**: Top candidates with allocation, metrics, sharpe
   - **2x2 Grid**: Visual cards for each candidate
   - **Risk Match Section**: 
     - Auto-selects portfolio matching TRUE_RISK
     - Stores in `st.session_state["chosen_portfolio"]` for Dashboard display
     - Metrics: CAGR, Sharpe, Max DD, volatility
     - Efficient frontier curve visualization

4. **Macro** (Economic Indicators)
   - **Purpose**: Plain-English explanations for 4 FRED series (beginners focus)
   - **Indicators**:
     - **CPIAUCSL** (Consumer Price Index): "Tracks the average change in prices consumers pay for goods and services. Rising CPI means inflation is increasing. A stable CPI around 2% annually is generally considered healthy."
     - **FEDFUNDS** (Fed Funds Rate): "The interest rate banks charge each other for overnight loans. The Federal Reserve adjusts this rate to control inflation and stimulate growth. Higher rates cool the economy; lower rates encourage borrowing."
     - **DGS10** (10-Year Treasury Rate): "The yield on U.S. government bonds maturing in 10 years. It's a benchmark for long-term interest rates. Rising yields often signal expectations of economic growth or inflation."
     - **UNRATE** (Unemployment Rate): "The percentage of the labor force that is unemployed and actively seeking work. Lower is generally better, but extremely low unemployment can signal overheating."
   - **Display**: Line charts with last value, last date, historical range
   - **Cache-First**: Loads from `data/macro/*.csv` (no live FRED calls)

5. **Diagnostics** (Debug & Provider Info)
   - **Universe Stats**: Total symbols, asset class breakdown, tier breakdown
   - **Provider Breakdown**: Count by provider (Tiingo, Stooq, yfinance, cache)
     - Safe handling: `isinstance(rec, dict)` check before `.get("provider")`
   - **Rolling Metrics**: Explanation of vol, sharpe, sortino with plain-English descriptions
   - **Provider Receipts**: Per-symbol data sources with first_date, last_date, row count
     - Safe attribute access for SymbolValidation objects
   - **Purpose**: Transparency into data quality, sources, calculations

### State Management

#### Session State Keys
- `page`: Current page name ("Dashboard", "Profile", "Portfolios", "Macro", "Diagnostics")
- `risk_score`: Computed 0-100 score from Profile questionnaire
- `chosen_portfolio`: Selected portfolio from Risk Match (dict with name, weights, metrics, curve)
- `simulation_results`: Last simulation output from Portfolios page
- `universe`: Valid universe snapshot (loaded once, cached in session)
- `profile_saved`: Boolean flag for Profile completion

#### Initialization Pattern
```python
# CRITICAL: Only set defaults if keys don't exist
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"
if "risk_score" not in st.session_state:
    st.session_state["risk_score"] = None
if "chosen_portfolio" not in st.session_state:
    st.session_state["chosen_portfolio"] = None
# ... etc.
```

#### Navigation Pattern
```python
# Sidebar radio for page selection
nav_selection = st.sidebar.radio("Navigation", PAGE_OPTIONS, index=current_index)
if nav_selection != st.session_state["page"]:
    st.session_state["page"] = nav_selection
    st.rerun()

# Button handlers for CTAs
if st.button("Go to Profile →"):
    st.session_state["page"] = "Profile"
    st.rerun()
```

#### Reset Behavior
- **Navigation**: Does NOT reset any state (risk_score, chosen_portfolio persist)
- **Reset Session Button**: Clears all session state keys, returns to Dashboard
- **Run Simulation Button**: Only updates simulation_results, preserves risk_score

### TRUE_RISK Calculation

**Formula**: `TRUE_RISK = 0.7 * risk_score + 0.3 * slider_score`

**Purpose**: Blend questionnaire-derived risk profile with user's explicit risk preference slider

**Example**:
- User Profile: risk_score = 60
- User Slider: slider_score = 80
- TRUE_RISK = 0.7 * 60 + 0.3 * 80 = 42 + 24 = 66

**Usage**: Auto-select portfolio from candidates that matches TRUE_RISK most closely

### Snapshot-Based Runtime

**Philosophy**: Cache-first, snapshot-based, no live API calls during page navigation

**Implementation**:
- **Universe**: `load_valid_universe()` reads `data/outputs/universe_snapshot.json` (pre-validated)
- **Prices**: `get_prices_with_provenance()` checks `data/cache/prices:*.csv` first, falls back to Stooq → Tiingo → yfinance
- **Macro**: FRED series loaded from `data/macro/*.csv` (updated via separate job)
- **Receipts**: Provider info embedded in cached files, no live metadata lookups

**Benefits**:
- Fast page loads (no API latency)
- Deterministic behavior (snapshot doesn't change mid-session)
- Respects API rate limits (Tiingo free tier: 50 req/hour)

## Key Features

### Fixed Bugs
1. **Risk Score Persistence**: Session state initialization now uses `if "key" not in st.session_state` guards, prevents overwrites
2. **Go to Profile Buttons**: Properly set `st.session_state["page"]` and call `st.rerun()`, no longer stuck on same page
3. **Run Simulation Visibility**: Button only appears after Profile saved, results display properly
4. **Redundant UI**: Removed duplicate questionnaire from Home page, consolidated to Profile page only

### Enhanced UX
1. **Dashboard Landing Page**: Clear hero summary, universe stats, selected portfolio preview (if any)
2. **Beginner-Friendly Macro**: Plain-English explanations for CPI, Fed Funds, 10Y Treasury, Unemployment
3. **Safe Provider Handling**: Diagnostics page handles both dict and object types for provider receipts
4. **Chosen Portfolio Storage**: Stores full context (name, weights, metrics, curve) for Dashboard display

## Data Flow

### 1. Profile → Portfolios
```python
# Profile page
risk_score = compute_risk_score(answers)
st.session_state["risk_score"] = risk_score
st.session_state["profile_saved"] = True

# Portfolios page
if st.session_state.get("profile_saved"):
    risk_pct = st.slider("Risk", 0, 100, value=st.session_state["risk_score"])
    TRUE_RISK = 0.7 * risk_score + 0.3 * risk_pct
```

### 2. Portfolios → Dashboard
```python
# Portfolios page (Risk Match section)
st.session_state["chosen_portfolio"] = {
    "name": rec["name"],
    "weights": rec["weights"],
    "metrics": rec["metrics"],
    "curve": rec["efficient_frontier"]
}

# Dashboard page
if st.session_state.get("chosen_portfolio"):
    show_selected_portfolio(st.session_state["chosen_portfolio"])
```

### 3. Universe Loading
```python
# One-time snapshot load
universe = load_valid_universe()  # reads data/outputs/universe_snapshot.json
st.session_state["universe"] = universe

# Usage in Diagnostics
for sym in universe["valid"]:
    rec = universe["receipts"].get(sym)
    if isinstance(rec, dict):
        provider = rec.get("provider", "unknown")
    else:
        provider = getattr(rec, "provider", "unknown")
```

## Testing

### Verification Commands
```bash
# Backend tests
python -m pytest tests/ -q
# Result: 29 passed, 4 warnings (test_volatility_scaling return value warnings - non-blocking)

# Import check
python -c "import ui.streamlit_app as app; print('✓ Streamlit app imports successfully')"
# Result: ✓ Streamlit app imports successfully

# Provider smoke tests
python dev/run_provider_smoke_tests.py
# Result: ✓ Smoke tests PASS (coverage target met)
#   - 67 valid symbols
#   - Tiingo rate limit hit (expected with free tier)
#   - Cache coverage verified
```

### Known Warnings
1. **Bare Mode Warnings**: Expected Streamlit warnings when importing without `streamlit run` context
2. **Test Return Value Warnings**: 4 warnings in `test_volatility_scaling` about non-None return values (not blocking)

## Deployment

### Running the App
```bash
# Development mode
streamlit run ui/streamlit_app.py

# Production mode (with config)
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Environment Variables
```bash
# Required for data refresh (separate job, not runtime)
TIINGO_API_KEY=xxx
FRED_API_KEY=xxx

# Optional enrichment
POLYGON_API_KEY=xxx
```

### Cache Refresh
```bash
# Update universe snapshot (run periodically, not on every app load)
python -c "from core.universe_validate import build_valid_universe; build_valid_universe()"

# Update macro data
python -c "from core.data_sources.fred import refresh_all_series; refresh_all_series()"
```

## File Structure

```
ui/
  streamlit_app.py         # Main app entry point (650+ lines)
  components/              # Reusable UI components (future)

core/
  risk_profile.py          # Risk questionnaire logic
  recommendation_engine.py # HRP optimization
  universe_validate.py     # Snapshot builder
  data_sources/
    fred.py                # FRED macro data loader
    router_smart.py        # Multi-provider price router

data/
  outputs/
    universe_snapshot.json # Pre-validated universe (67 symbols)
  cache/
    prices:*.csv           # Cached price data
  macro/
    *.csv                  # FRED series (CPIAUCSL, FEDFUNDS, DGS10, UNRATE)

config/
  assets_catalog.json      # Asset metadata (tier, class, min_risk_pct)
  config.yaml              # App configuration
```

## Maintenance Notes

### Adding New Pages
1. Add page name to `PAGE_OPTIONS` list
2. Define page function with session state logic
3. Update sidebar navigation radio
4. Add documentation to this file

### Modifying Risk Questionnaire
1. Update questions in Profile page
2. Adjust `compute_risk_score()` scoring logic
3. Test TRUE_RISK calculation with new ranges
4. Update documentation

### Changing TRUE_RISK Formula
1. Update formula in Portfolios page (Risk Match section)
2. Update formula in this documentation
3. Consider impact on portfolio selection logic
4. Add migration notes if persisted state affected

### Provider Receipts
- **Dict Type**: Older cached files use `{"provider": "tiingo", ...}`
- **Object Type**: New `SymbolValidation` dataclass from `core.universe_validate`
- **Safe Access**: Always check `isinstance(rec, dict)` before using `.get()`

## Troubleshooting

### Risk Score Resets on Navigation
- **Symptom**: risk_score returns to None after visiting other pages
- **Fix**: Check session state initialization uses `if "key" not in st.session_state` guards

### Go to Profile Buttons Not Working
- **Symptom**: Button click doesn't change page
- **Fix**: Ensure `st.session_state["page"] = "Profile"` followed by `st.rerun()`

### Provider Receipt Errors
- **Symptom**: AttributeError: 'dict' object has no attribute 'get'
- **Fix**: Use `isinstance(rec, dict)` check before accessing provider field

### Slow Page Loads
- **Symptom**: Navigation takes >2s per page
- **Cause**: Live API calls during page render
- **Fix**: Use snapshot-based universe, cache-first price loading

## Future Enhancements

1. **Advanced Filters**: Asset class, sector, geography filters on Portfolios page
2. **Historical Simulation**: Run portfolio on past data, show performance over time
3. **Rebalancing**: Quarterly/annual rebalance simulation with transaction costs
4. **Tax Optimization**: Tax-loss harvesting, capital gains management
5. **Custom Assets**: Allow user to add custom tickers to universe
6. **Export**: Download portfolio weights, allocation reports as CSV/PDF
7. **Alerts**: Email/SMS notifications when portfolio drift exceeds threshold

# Phase 3 Runtime Verification Notes

_Last updated: 2025-11-18_

These notes document the most recent end-to-end execution of the Phase 3 multi-factor engine to confirm that typical Balanced users receive non-emergency (non-hard-fallback) portfolios with clear visibility in the UI and supporting instrumentation.

## Execution Context

- **Command**: `PYTHONPATH=. .venv/bin/python dev/smoke_phase3.py`
- **Environment**: macOS, Python 3.11 virtualenv (`.venv`)
- **Universe Size**: 28 ETFs after catalog filters
- **Risk Profile Under Test**: Balanced (TRUE_RISK ≈ 66.5 / volatility target 18.9%)

## Key Observations

| Dimension | Details | Outcome |
| --- | --- | --- |
| Candidate generation | 20 total candidate portfolios constructed before quality filters | ✅
| Strict (Stage 1) filters | 0 candidates met the strict thresholds (expected for Balanced users after raising CAGR floor) | ⚠️ expected
| Stage 2 relaxed filters | 2 candidates (`MAX_SHARPE - Sat 20%`, `HRP - Sat 20%`) passed with CAGR 12.77% / 11.14% and volatility under 9% | ✅ delivers non-hard fallback
| Later stages | Stage 3/4 were not needed; no emergency fallback triggered | ✅
| Portfolio quality labels | Both surfaced as "Relaxed filters" in the debug payload (quality computed from fallback stage) | ✅
| Receipts | Per-ticker and per-portfolio receipts populated (20 receipt rows) | ✅
| Risk–objective banner | Shows green success message because objective fit scores remained high under relaxed filters | ✅ (verified via Streamlit logic trace)

## Interpretation

1. **Relaxed Stage-2 thresholds are sufficient**: Increasing the volatility tolerance (soft floor now scales with profile risk) kept both MAX_SHARPE and HRP candidates eligible without touching emergency fallbacks.
2. **Stats payload matches UI**: `recommendation_engine` reports `stats={"total":20,"strict":0,"recommended":2,"fallback":2,"hard":0}` which powers both smoke output and Streamlit debug expander, ensuring transparency.
3. **User messaging is coherent**: Balanced users see the objective-fit banner before the table plus the Quality column labeling these portfolios as "Relaxed filters" instead of "Emergency fallback".
4. **Receipts remain intact**: Asset and portfolio receipts include provider provenance, span, and optimizer metadata—matching the UI per-ticker receipt expander requirement.

## Next Checks

- Spot-check Streamlit in a browser to visually confirm the Quality tags align with the stats above (already validated in code paths but worth a UX glance).
- Re-run `dev/smoke_final_integration.py` whenever catalog limits or relaxed multipliers change to ensure end-to-end resiliency.
- Monitor Tiingo rate-limit skip logs to ensure the provider ladder remains healthy when running the UI live.

These notes should be updated whenever we materially adjust Stage-2 thresholds, change catalog allocations, or observe the engine falling back to Stage 3/4 for mainstream profiles.

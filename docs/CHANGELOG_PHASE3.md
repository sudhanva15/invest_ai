# Phase 3 Changelog (Multifactor & UX Enhancements)

Date: 2025-11-17

## Engine & Filtering
- Soft volatility band implemented (`portfolio_passes_filters`): portfolios slightly below target vol now pass with a score penalty instead of full rejection.
- Composite scoring updated: applies soft-band gap penalty (`soft_vol_penalty_lambda`) after base `Sharpe - λ*|MaxDD|`.
- Refined fallback hierarchy in `build_recommendations`: relaxed Level 1 acceptance before legacy fallback; emergency equal-weight only as final resort.
- Asset-class injection: `asset_classes` mapping added to each recommended portfolio; receipts now include normalized `asset_class`.
- Centralized asset-class mapping helper (`core/utils/asset_classes.py`).

## Provider Resilience
- Tiingo rate-limit guard: module-level flag prevents repeated HTTP calls after first 429 / textual limit detection; subsequent fetches return empty until reset.
- Added skip behavior test (`test_tiingo_rate_limit.py`).

## UI / UX Improvements
- Dark themed Recommended Portfolios table with fallback row shading & risk band indicators (✅ in-band / ⚠️ out-of-band).
- Allocation view toggle (ticker vs asset class) with prioritized symbol classification.
- Beginner Explanations expander: strategy glossary, metric definitions, soft-band rationale, DCA disclaimer.
- Risk Match volatility display unified (eliminated prior 0.0% bug).

## Testing Additions
- Soft band behavior test (`test_phase3_soft_band.py`) exercising soft zone penalty logic.
- Diversification and asset-class mapping test (`test_diversification_asset_classes.py`).
- Tiingo rate-limit skip test verifying no second network call post 429.

## Configuration
- Added `vol_soft_lower_factor` and `soft_vol_penalty_lambda` defaults to `config/config.yaml` under `multifactor`.

## Quality Gates
- Full pytest suite passing (warnings from `pypfopt` and `cvxpy` acknowledged; non-fatal).

## Follow-ups / Next Considerations
- Optional: Store risk profile object directly in session state for more precise band rendering.
- Potential suppression or styling of optimization warnings in UI debug pane.
- Evaluate dynamic soft-factor adaptation to market regime volatility.

---
Generated automatically; edit manually for release notes formatting if publishing externally.

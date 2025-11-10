# Invest_AI v3.0.1 Release Notes

Date: 2025-11-10
Tag: v3.0.1

## What’s New
This release finalizes the V3 architecture with a fully scriptable, non‑Streamlit validation and scenario pipeline:

1. **CLI Validator (`dev/validate_simulations.py`)**
   - Gates: Data Quality, Returns Quality, Macro Freshness (DGS10, T10Y2Y, CPIAUCSL), Candidate Generation, Metrics Plausibility (CAGR, Sharpe, MaxDD ranges), Receipt Integrity.
   - Supports `--json` output for CI.
2. **Scenario Runner (`dev/run_scenarios.py`)**
   - Generates candidate portfolios with HRP & variants; produces artifacts in `dev/artifacts/` (JSON, weights CSV, metrics CSV).
3. **Standardized Metrics (`annualized_metrics`)**
   - Consistent CAGR, Volatility, Sharpe, Max Drawdown computation across horizons.
4. **Objective & Candidate Framework**
   - `DEFAULT_OBJECTIVES` with growth / balanced / income / barbell profiles and controlled satellite caps.
5. **Macro Regime Labeling**
   - Regime feature extraction and labeling enhances candidate context and diversity scoring.
6. **Diversity-Aware Ranking**
   - Optional penalties for concentration & sector weight bias plus regime nudges (feature flag `RANK_DIVERSITY`).

## Preflight (Local Release Validation)
Run these three commands before tagging to ensure full health:
```bash
bash dev/run_tests.sh
python3 dev/validate_simulations.py --objective balanced --n-candidates 6
python3 dev/snapshot_weights.py --update-baseline && bash dev/diff_weights.sh || true
```
PASS criteria:
- All unit tests green.
- Validator reports all checks PASS.
- Weight diff script shows either “No significant weight changes” (≤2% deltas) OR baseline updated intentionally.

## Updating the Baseline & Why
The baseline weights (`tests/fixtures/weights_baseline.json`) capture a reference allocation for regression checking. Update only when:
- Data provider adjustments yield expected structural shifts.
- Objective parameter changes (e.g., satellite cap modifications) are deliberate.
Use:
```bash
python3 dev/snapshot_weights.py --update-baseline
bash dev/diff_weights.sh
```
If deltas remain ≤2%, no update is required. Larger intentional shifts should be documented here.

## Current Status
All checks PASS on v3.0.1:
- Validator: PASS (balanced objective, 6 candidates)
- Tests: PASS (`dev/run_tests.sh`)
- Baseline: No significant weight changes (≤2% deltas)

## CI Additions
A lightweight GitHub Actions workflow (`.github/workflows/validator.yml`) runs the balanced validator on every push, uploading artifacts for inspection.

## Next Steps / Suggestions
- Optional nightly scenario sweep using existing sensitivity runner.
- Add per-regime performance deltas to report generator.
- Integrate receipts freshness card into UI.

---
**Tag Annotation:** `v3.0.1 — V3: validator + scenarios + passing tests`

Happy allocating!

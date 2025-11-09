# V3 Release Changes

## Feature Scope
- [ ] Portfolio receipts with provider transparency
- [ ] Macro indicators integration (FRED)
- [ ] Enhanced performance metrics (CAGR, Sharpe, MaxDD)
- [ ] Core/Satellite constraints (≥65% core, ≤35% sats, ≤7% per sat)

## Test Plan
1. Unit Tests:
   - [ ] `dev/test_allocation.py` - Core/Satellite constraints
   - [ ] `dev/test_metrics.py` - Performance calculations  
   - [ ] `dev/test_macro.py` - FRED integration
   - [ ] `dev/test_receipts.py` - Receipt generation

2. Weight Stability:
   - [ ] `./dev/snapshot_weights.py` baseline comparison
   - [ ] No significant weight changes (>2%) detected

3. UI Walkthrough:
   - [ ] Receipts downloadable as CSV
   - [ ] Macro tab shows key indicators
   - [ ] Performance metrics displayed correctly
   - [ ] Core/Satellite constraints enforced

## Screenshots
Please attach:
1. Receipts table with per-asset details
2. Macro indicators tab 
3. Portfolio metrics panel

## Post-Release
- [ ] Tag pushed as v3.0.0
- [ ] Documentation updated for new features
- [ ] Migration guide if needed
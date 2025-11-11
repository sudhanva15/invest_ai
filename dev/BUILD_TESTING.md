# Build Test Harness - Invest_AI V3

IDEMPOTENT build validation system for Invest_AI V3. Test any build locally or in CI without changing repository state. Multiple runs for the same build produce identical results.

## Quick Start

```bash
# Test current build
make build-test

# List tested builds
make build-list

# Compare two builds
make build-diff BUILD1=V3.0.1+42 BUILD2=V3.0.1+45

# Force re-test (override idempotency)
make build-test-force
```

## Architecture

```
dev/
├── test_build.sh           # Main test harness (bash)
├── compare_builds.py       # Build diff tool (python)
└── builds/                 # Test results (gitignored)
    └── ${BUILD}/
        ├── summary.json    # Atomic test summary
        ├── validator.log   # validate_simulations.py output
        ├── acceptance.log  # acceptance_v3.py output
        ├── scenario.log    # run_scenarios.py output
        └── artifacts/      # Copied from dev/artifacts/
```

## Features

### 1. Idempotent Execution

- **First run**: Executes all tests and creates `summary.json`
- **Subsequent runs**: Detects existing summary and exits immediately
- **Override**: Use `--force` flag to re-run tests

```bash
$ bash dev/test_build.sh
🏗️  Testing build V3.0.1+42...
✅ BUILD OK: V3.0.1+42

$ bash dev/test_build.sh
ℹ️  Build V3.0.1+42 already tested. Skipping (idempotent).
```

### 2. Build Stamping

Uses `core.utils.version.get_build_version()` for consistent versioning:

- **Tagged builds**: `V3.0.1+42` (42 commits since v3.0.1)
- **Dirty builds**: `V3.0.1+42.dirty` (uncommitted changes)
- **Dev builds**: `dev-20251110-143000` (no git)

### 3. Three-Phase Testing

**Phase 1: Validation Suite**
```bash
python3 dev/validate_simulations.py --objective balanced --n-candidates 6
```
- Data quality checks
- Returns quality validation
- Macro data validation
- Candidate generation tests
- Metrics plausibility
- Receipt integrity

**Phase 2: Acceptance Tests**
```bash
python3 dev/acceptance_v3.py
```
- Generate ≥3 candidates with unique weights
- Bootstrap-ANOVA shortlist selection
- FRED macro series loading
- Catalog validation
- Export functionality

**Phase 3: Scenario Smoke Test**
```bash
python3 dev/run_scenarios.py --objective balanced --n-candidates 8
```
- End-to-end scenario execution
- Portfolio optimization pipeline
- Artifact generation

### 4. Atomic Artifact Handling

All writes use temporary files with atomic rename:

```bash
# Create in temp location
tmp=$(mktemp)
python3 generate_summary.py > "$tmp"

# Atomic rename (all-or-nothing)
mv "$tmp" "dev/builds/${BUILD}/summary.json"
```

### 5. Build Comparison

Side-by-side diff of two builds:

```bash
$ python3 dev/compare_builds.py V3.0.1+42 V3.0.1+45

============================================================
Build Comparison
============================================================

Build 1: V3.0.1+42
  Timestamp: 2025-11-10T15:30:00
  Status: PASS

Build 2: V3.0.1+45
  Timestamp: 2025-11-10T17:45:00
  Status: PASS

Test Results:
  validator_pass       ✓ → ✓
  acceptance_pass      ✓ → ✓
  scenario_pass        ✗ → ✓ (FIXED)

Artifacts:
  Total files: 36 → 39
  Common: 36
  Only in V3.0.1+45: candidates_v2.json, weights_final.csv

Overall:
  IMPROVEMENT: PARTIAL → PASS
```

## CI Integration

GitHub Actions workflow (`.github/workflows/build-test.yml`):

```yaml
name: Build Test

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for version stamping
      
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Run build test
        run: bash dev/test_build.sh
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build-test-results
          path: dev/builds/*/
```

## Usage Examples

### Local Development

```bash
# Test current state
make build-test

# Check what changed since last PR
make build-list
make build-diff BUILD1=V3.0.1+38 BUILD2=V3.0.1+42

# Force re-test after fixing issues
make build-test-force
```

### Release Engineering

```bash
# Create clean build from tag
git checkout v3.1.0
bash dev/test_build.sh

# Verify all tests pass
cat dev/builds/V3.1.0/summary.json

# Tag successful build
git tag -a v3.1.0 -m "Release V3.1.0 - all tests pass"
```

### Debugging Failed Builds

```bash
# Run test suite
make build-test

# Check logs if tests fail
BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")

cat "dev/builds/${BUILD}/validator.log"     # Validation failures
cat "dev/builds/${BUILD}/acceptance.log"    # Acceptance issues
cat "dev/builds/${BUILD}/scenario.log"      # Scenario errors

# Check artifact output
ls -la "dev/builds/${BUILD}/artifacts/"
```

## File Reference

### test_build.sh

Main test harness script. Features:

- **Idempotency guard**: Checks for existing `summary.json`
- **Build discovery**: Uses `get_build_version()` for consistent naming
- **Atomic writes**: All outputs use `tmp → rename` pattern
- **Error handling**: Trap on ERR with cleanup
- **Exit code 0**: Always succeeds (for CI artifact capture)

Usage:
```bash
bash dev/test_build.sh           # Normal run
bash dev/test_build.sh --force   # Override idempotency
```

### compare_builds.py

Build comparison utility. Features:

- **List builds**: `--list` flag shows all tested builds
- **Side-by-side diff**: Compare test results, artifacts, logs
- **Regression detection**: Highlights PASS→FAIL transitions
- **Colored output**: ANSI colors (auto-disabled for non-TTY)

Usage:
```bash
python3 dev/compare_builds.py --list
python3 dev/compare_builds.py V3.0.1+42 V3.0.1+45
python3 dev/compare_builds.py --no-color V3.0.1+42 V3.0.1+45  # CI-friendly
```

### Makefile Targets

**Root Makefile** (`Makefile`):
- `build-test`: Run test harness (idempotent)
- `build-test-force`: Force re-run
- `build-list`: List all builds
- `build-diff`: Compare two builds

**Dev Makefile** (`dev/Makefile`):
- Scenario runner targets unchanged
- Build targets use relative paths from dev/

## Summary JSON Schema

```json
{
  "build": "V3.0.1+42.dirty",
  "timestamp": "2025-11-10T17:55:34",
  "tests": {
    "validator_pass": true,
    "acceptance_pass": true,
    "scenario_pass": false
  },
  "logs": {
    "validator": "dev/builds/V3.0.1+42.dirty/validator.log",
    "acceptance": "dev/builds/V3.0.1+42.dirty/acceptance.log",
    "scenario": "dev/builds/V3.0.1+42.dirty/scenario.log"
  },
  "artifacts": {
    "count": 36,
    "directory": "dev/builds/V3.0.1+42.dirty/artifacts"
  },
  "overall_status": "PASS"
}
```

**Status Logic**:
- `PASS`: Both `validator_pass` and `acceptance_pass` are true
- `FAIL`: Either critical test failed
- Scenario test is informational only (doesn't affect overall status)

## Best Practices

### 1. Always Test Before Commit

```bash
make build-test
# Review summary
# Only commit if PASS
```

### 2. Compare Against Main Before PR

```bash
# Get main's latest build
git checkout main
make build-test
MAIN_BUILD=$(python3 -c "...")

# Switch back to feature branch
git checkout feature/my-change
make build-test
FEATURE_BUILD=$(python3 -c "...")

# Compare
make build-diff BUILD1=$MAIN_BUILD BUILD2=$FEATURE_BUILD
```

### 3. Clean Old Builds Periodically

```bash
# Keep last 10 builds
cd dev/builds
ls -t | tail -n +11 | xargs rm -rf
```

### 4. Use in Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

make build-test
BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")

STATUS=$(python3 -c "import json; print(json.load(open('dev/builds/${BUILD}/summary.json'))['overall_status'])")

if [ "$STATUS" != "PASS" ]; then
  echo "❌ Build tests failed - commit blocked"
  exit 1
fi
```

## Troubleshooting

### Issue: "Build already tested" but I want to re-run

**Solution**: Use `--force` flag
```bash
make build-test-force
```

### Issue: Python not found

**Solution**: Activate virtualenv first
```bash
source .venv/bin/activate
make build-test
```

### Issue: Tests pass locally but fail in CI

**Solution**: Check API keys in GitHub Secrets
```bash
# Required secrets:
# - FRED_API_KEY (macro data)
# - TIINGO_API_KEY (price data)
```

### Issue: build-list shows no builds

**Solution**: Run test harness first
```bash
make build-test
make build-list  # Should now show builds
```

## Dependencies

**Runtime**:
- Python 3.11+ (already required by project)
- Bash 4.0+ (macOS/Linux default)
- Git (for version stamping)

**No new Python packages required** - uses existing project dependencies.

## Future Enhancements

Potential additions (not currently implemented):

1. **Performance tracking**: Store runtime metrics in summary
2. **Regression alerts**: Automated email/Slack on PASS→FAIL
3. **Artifact diffing**: Binary diff of CSV/JSON outputs
4. **Historical trending**: Plot test success rate over time
5. **Parallel testing**: Run validator/acceptance concurrently

## License

Same as parent project (Invest_AI V3).

## Maintainers

- Primary: Release Engineering team
- Issues: File under `build-testing` label

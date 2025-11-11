#!/usr/bin/env bash
# =============================================================================
# BUILD TEST QUICK REFERENCE
# =============================================================================
# Invest_AI V3 - Idempotent Build Testing Cheat Sheet
#
# All commands assume you're in the repository root directory.
# =============================================================================

# ── BASIC USAGE ──────────────────────────────────────────────────────────────

# Test current build (idempotent - safe to run multiple times)
make build-test

# Force re-run tests (override idempotency guard)
make build-test-force

# List all tested builds
make build-list

# Compare two builds
make build-diff BUILD1=V3.0.1+42 BUILD2=V3.0.1+45

# ── DIRECT SCRIPT USAGE ──────────────────────────────────────────────────────

# Run test harness directly
bash dev/test_build.sh

# Force re-test
bash dev/test_build.sh --force

# List builds
python3 dev/compare_builds.py --list

# Compare builds
python3 dev/compare_builds.py V3.0.1+42 V3.0.1+45

# ── BUILD INSPECTION ─────────────────────────────────────────────────────────

# Get current build version
python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())"

# View build summary
BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")
cat "dev/builds/${BUILD}/summary.json"

# Check test logs
cat "dev/builds/${BUILD}/validator.log"
cat "dev/builds/${BUILD}/acceptance.log"
cat "dev/builds/${BUILD}/scenario.log"

# List artifacts
ls -la "dev/builds/${BUILD}/artifacts/"

# ── WORKFLOW PATTERNS ────────────────────────────────────────────────────────

# Pre-commit check
make build-test && git commit -m "feat: my change"

# Compare feature branch against main
git checkout main && make build-test
MAIN_BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")
git checkout feature/my-branch && make build-test
FEAT_BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")
make build-diff BUILD1=$MAIN_BUILD BUILD2=$FEAT_BUILD

# Clean old builds (keep last 10)
cd dev/builds && ls -t | tail -n +11 | xargs rm -rf && cd ../..

# ── CI/CD ────────────────────────────────────────────────────────────────────

# Simulate CI run locally
export PYTHON=.venv/bin/python3
bash dev/test_build.sh
echo "Exit code: $?"

# Check if build passed (for CI scripts)
BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")
STATUS=$(python3 -c "import json; print(json.load(open('dev/builds/${BUILD}/summary.json'))['overall_status'])")
if [ "$STATUS" = "PASS" ]; then echo "✅ PASS"; else echo "❌ FAIL"; exit 1; fi

# ── DEBUGGING ────────────────────────────────────────────────────────────────

# Run individual test phases manually
python3 dev/validate_simulations.py --objective balanced --n-candidates 6
python3 dev/acceptance_v3.py
python3 dev/run_scenarios.py --objective balanced --n-candidates 8

# Check build diagnostics
python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_diagnostics; import json; print(json.dumps(get_build_diagnostics(), indent=2))"

# ── DIRECTORY STRUCTURE ──────────────────────────────────────────────────────
#
# dev/builds/${BUILD}/
#   ├── summary.json       # Test results summary (JSON)
#   ├── validator.log      # validate_simulations.py output
#   ├── acceptance.log     # acceptance_v3.py output
#   ├── scenario.log       # run_scenarios.py output
#   └── artifacts/         # Copied from dev/artifacts/
#       ├── candidates.csv
#       ├── candidates.json
#       ├── shortlist_weights.csv
#       └── ...
#
# ── STATUS CODES ─────────────────────────────────────────────────────────────
#
# overall_status values:
#   - PASS: Both validator_pass and acceptance_pass are true
#   - FAIL: Either critical test failed
#
# Scenario test is informational only (doesn't affect overall status)
#
# ── TIPS ─────────────────────────────────────────────────────────────────────
#
# 1. Always run build-test before committing major changes
# 2. Use build-diff to compare against baseline before PR
# 3. Force re-test only when changing test infrastructure
# 4. Keep build history manageable (clean old builds periodically)
# 5. Check logs if overall_status is FAIL
#
# ══════════════════════════════════════════════════════════════════════════════

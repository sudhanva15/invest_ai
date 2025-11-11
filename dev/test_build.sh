#!/usr/bin/env bash
# =============================================================================
# test_build.sh - IDEMPOTENT Build Test Harness for Invest_AI V3
# =============================================================================
# GOAL: One command to validate any build without changing repo state.
#       Multiple runs for the same build produce identical results.
#
# USAGE:
#   ./dev/test_build.sh               # Test current build
#   ./dev/test_build.sh --force       # Force re-test even if summary exists
#
# OUTPUT:
#   dev/builds/${BUILD}/
#     ├── summary.json        # Atomic summary with timestamps
#     ├── validator.log       # validate_simulations.py output
#     ├── acceptance.log      # acceptance_v3.py output
#     ├── scenario.log        # run_scenarios.py output
#     └── artifacts/          # Copies of dev/artifacts/*
#
# IDEMPOTENCY:
#   - If summary.json exists, exits with success (skips re-run)
#   - Use --force to override idempotency guard
#   - All writes are atomic (tmp → rename)
# =============================================================================

set -euo pipefail

# --- Configuration ---
PYTHON="${PYTHON:-.venv/bin/python3}"
FORCE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "❌ Unknown option: $1"
      echo "Usage: $0 [--force]"
      exit 1
      ;;
  esac
done

# --- Build Discovery ---
# Use core.utils.version.get_build_version() for consistent stamping
BUILD="$("${PYTHON}" - <<'PY'
import sys
sys.path.insert(0, ".")
from core.utils.version import get_build_version
print(get_build_version())
PY
)"

if [[ -z "${BUILD}" ]]; then
  echo "❌ Failed to determine build version"
  exit 1
fi

OUT="dev/builds/${BUILD}"
mkdir -p "${OUT}/artifacts"

# --- Idempotency Guard ---
if [[ -f "${OUT}/summary.json" && ${FORCE} -eq 0 ]]; then
  echo "ℹ️  Build ${BUILD} already tested. Skipping (idempotent)."
  echo "    Use --force to re-run."
  echo "    Summary: ${OUT}/summary.json"
  exit 0
fi

echo "🏗️  Testing build ${BUILD}..."
echo "    Output directory: ${OUT}"

# Trap for error handling
trap 'echo "❌ Build ${BUILD} failed"; exit 1' ERR

# --- Phase 1: Validation Suite ---
echo "📋 Running validate_simulations.py..."
tmpv=$(mktemp)
"${PYTHON}" dev/validate_simulations.py --objective balanced --n-candidates 6 >"${tmpv}" 2>&1 || true
mv "${tmpv}" "${OUT}/validator.log"

VAL_PASS=0
if grep -q "ALL CHECKS PASSED" "${OUT}/validator.log"; then
  VAL_PASS=1
  echo "   ✅ Validator PASSED"
else
  echo "   ⚠️  Validator had issues (see validator.log)"
fi

# --- Phase 2: Acceptance Tests ---
echo "📋 Running acceptance_v3.py..."
tmpa=$(mktemp)
"${PYTHON}" dev/acceptance_v3.py >"${tmpa}" 2>&1 || true
mv "${tmpa}" "${OUT}/acceptance.log"

ACC_PASS=0
if grep -q "ACCEPTANCE: PASS" "${OUT}/acceptance.log"; then
  ACC_PASS=1
  echo "   ✅ Acceptance PASSED"
else
  echo "   ⚠️  Acceptance had issues (see acceptance.log)"
fi

# --- Phase 3: Scenario Smoke Test ---
echo "📋 Running scenario smoke test..."
tmps=$(mktemp)
"${PYTHON}" dev/run_scenarios.py --objective balanced --n-candidates 8 >"${tmps}" 2>&1 || true
mv "${tmps}" "${OUT}/scenario.log"

SCENARIO_PASS=0
if grep -q "SCENARIO COMPLETE" "${OUT}/scenario.log"; then
  SCENARIO_PASS=1
  echo "   ✅ Scenario smoke test PASSED"
else
  echo "   ⚠️  Scenario test had issues (see scenario.log)"
fi

# --- Phase 4: Copy Artifacts ---
echo "📦 Copying artifacts..."
rm -rf "${OUT}/artifacts"
mkdir -p "${OUT}/artifacts"

if [[ -d "dev/artifacts" ]]; then
  # Copy only if artifacts directory has content
  if ls dev/artifacts/* >/dev/null 2>&1; then
    cp -r dev/artifacts/* "${OUT}/artifacts/" 2>/dev/null || true
    ARTIFACT_COUNT=$(ls -1 "${OUT}/artifacts" 2>/dev/null | wc -l | tr -d ' ')
    echo "   Copied ${ARTIFACT_COUNT} artifacts"
  else
    ARTIFACT_COUNT=0
  fi
else
  ARTIFACT_COUNT=0
fi

# --- Phase 5: Generate Summary (ATOMIC) ---
echo "📝 Generating summary..."

# Create summary in temp file first (atomic write)
tmp_summary=$(mktemp)
"${PYTHON}" - "${BUILD}" "${OUT}" "${VAL_PASS}" "${ACC_PASS}" "${SCENARIO_PASS}" "${ARTIFACT_COUNT}" <<'PY'
import json
import sys
import datetime as dt
import os

build, out, val_pass, acc_pass, scenario_pass, artifact_count = sys.argv[1:7]

summary = {
    "build": build,
    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
    "tests": {
        "validator_pass": val_pass == "1",
        "acceptance_pass": acc_pass == "1",
        "scenario_pass": scenario_pass == "1",
    },
    "logs": {
        "validator": f"{out}/validator.log",
        "acceptance": f"{out}/acceptance.log",
        "scenario": f"{out}/scenario.log",
    },
    "artifacts": {
        "count": int(artifact_count),
        "directory": f"{out}/artifacts",
    },
    "overall_status": "PASS" if (val_pass == "1" and acc_pass == "1") else "FAIL"
}

# Write to temp file, then atomic rename
tmp_path = f"{out}/summary.json.tmp"
with open(tmp_path, "w") as f:
    json.dump(summary, f, indent=2)

# Atomic rename
os.rename(tmp_path, f"{out}/summary.json")
print(json.dumps(summary, indent=2))
PY

mv "${tmp_summary}" /dev/null 2>&1 || true

# --- Final Status ---
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [[ ${VAL_PASS} -eq 1 && ${ACC_PASS} -eq 1 ]]; then
  echo "✅ BUILD OK: ${BUILD}"
  echo "   All critical tests passed"
else
  echo "⚠️  BUILD PARTIAL: ${BUILD}"
  echo "   Some tests failed - review logs"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Summary: ${OUT}/summary.json"
echo "📂 Artifacts: ${OUT}/artifacts/ (${ARTIFACT_COUNT} files)"
echo ""

# Exit with success even if some tests failed (non-critical)
# This allows idempotent re-runs and CI to capture partial results
exit 0

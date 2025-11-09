#!/usr/bin/env bash
# Preflight checks before launching Streamlit UI
# Run: bash dev/preflight.sh

set -e  # Exit on any error

# Use venv Python
PYTHON=".venv/bin/python"

echo "========================================"
echo "Invest_AI V3 Preflight Checks"
echo "========================================"

# 1) Run unit tests
echo ""
echo "[1/4] Running unit tests..."
$PYTHON dev/test_allocation.py
$PYTHON dev/test_receipts.py
echo "✓ Unit tests PASS"

# 2) Run smoke check
echo ""
echo "[2/4] Running smoke check..."
$PYTHON dev/smoke_check.py
echo "✓ Smoke check PASS"

# 3) Generate snapshot
echo ""
echo "[3/4] Generating weight snapshot..."
$PYTHON dev/snapshot_weights.py 2>/dev/null | $PYTHON -m json.tool --compact
echo "✓ Snapshot generated"

# 4) Run diff against baseline
echo ""
echo "[4/4] Checking weight stability..."
if bash dev/diff_weights.sh 2>&1 | grep -q "No significant"; then
    echo "✓ Weights stable (≤2% changes)"
else
    echo "⚠ Weights changed >2% from baseline"
    echo "  Run: python3 dev/snapshot_weights.py --update-baseline"
    echo "  if this is expected behavior"
fi

echo ""
echo "========================================"
echo "✓ All preflight checks complete!"
echo "========================================"
echo ""
echo "Launch UI:"
echo "  streamlit run core/ui/dashboard.py"
echo ""

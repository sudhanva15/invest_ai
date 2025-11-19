#!/usr/bin/env bash
# Comprehensive sanity check script for invest_ai
# Usage: ./scripts/sanity_check_all.sh

set -e  # Exit on first error

echo "======================================================================"
echo "🔍 SANITY CHECK SUITE - invest_ai"
echo "======================================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
FAILED=0

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 1: FILE-LEVEL CHECKS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check critical files exist
CRITICAL_FILES=(
    "core/recommendation_engine.py"
    "core/multifactor.py"
    "core/data_ingestion.py"
    "ui/streamlit_app.py"
    "config/config.yaml"
    "config/assets_catalog.json"
    "tests/test_phase3_soft_band.py"
    "tests/test_diversification_asset_classes.py"
    "tests/test_tiingo_rate_limit.py"
)

echo "Checking critical files..."
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 2: BUILD-LEVEL CHECKS (Syntax Compilation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Compiling all Python files..."
if python3 -m py_compile $(git ls-files '*.py') 2>&1; then
    echo -e "${GREEN}✅ All Python files compile successfully${NC}"
else
    echo -e "${RED}❌ Syntax errors detected${NC}"
    FAILED=$((FAILED + 1))
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 3: RUNTIME CHECKS (Import & Instantiation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing core module imports..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.data_ingestion import get_prices
from core.recommendation_engine import recommend
from core.multifactor import build_filtered_universe, portfolio_passes_filters
from core.utils.asset_classes import build_symbol_metadata_map
from core.investor_profiles import classify_investor, InvestorInputs
print('✅ All core modules import successfully')
" || { echo -e "${RED}❌ Import test failed${NC}"; FAILED=$((FAILED + 1)); }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 4: TEST SUITE EXECUTION"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Running Phase 3 tests..."
if python3 -m pytest tests/test_phase3_soft_band.py tests/test_diversification_asset_classes.py tests/test_tiingo_rate_limit.py -v --tb=short 2>&1 | tee /tmp/pytest_output.txt; then
    echo -e "${GREEN}✅ All tests passed${NC}"
else
    echo -e "${RED}❌ Test failures detected${NC}"
    FAILED=$((FAILED + 1))
fi

# Check for test count
TEST_COUNT=$(grep -c "passed" /tmp/pytest_output.txt || echo "0")
echo "Total tests passed: $TEST_COUNT"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "STAGE 5: INTEGRATION SMOKE TESTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing recommendation engine pipeline..."
python3 -c "
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from core.recommendation_engine import recommend

np.random.seed(42)
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
returns = pd.DataFrame({
    'SPY': np.random.randn(len(dates)) * 0.01,
    'BND': np.random.randn(len(dates)) * 0.005,
    'GLD': np.random.randn(len(dates)) * 0.008
}, index=dates)

profile = {'risk_level': 'moderate', 'horizon_years': 10}
rec = recommend(returns, profile, objective='grow', risk_pct=50, method='hrp')

if rec and 'weights' in rec and len(rec['weights']) > 0:
    print(f'✅ Integration: Portfolio generated with {len(rec[\"weights\"])} assets')
    print(f'   Sharpe: {rec[\"metrics\"].get(\"Sharpe\", \"N/A\"):.2f}')
else:
    print('❌ Integration: Empty recommendation')
    sys.exit(1)
" 2>&1 | grep -v "FutureWarning" || { echo -e "${RED}❌ Integration test failed${NC}"; FAILED=$((FAILED + 1)); }

echo ""
echo "======================================================================"
echo "📊 SUMMARY"
echo "======================================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✅ ALL SANITY CHECKS PASSED${NC}"
    echo ""
    echo "Stage 1 (File):         ✅ PASS"
    echo "Stage 2 (Build):        ✅ PASS"
    echo "Stage 3 (Runtime):      ✅ PASS"
    echo "Stage 4 (Tests):        ✅ PASS"
    echo "Stage 5 (Integration):  ✅ PASS"
    exit 0
else
    echo -e "${RED}❌ SANITY CHECK FAILURES: $FAILED${NC}"
    exit 1
fi

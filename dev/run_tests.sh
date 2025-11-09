#!/bin/bash

# Stop on first error
set -e

# Ensure PYTHONPATH includes project root
export PYTHONPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Running test suite from $(pwd)"
echo "PYTHONPATH=$PYTHONPATH"
echo

run_test() {
    local test_script=$1
    echo "Running ${test_script}..."
    if python3 "$test_script"; then
        echo -e "${GREEN}✓ Passed: ${test_script}${NC}"
        return 0
    else
        echo -e "${RED}✗ Failed: ${test_script}${NC}"
        return 1
    fi
}

# Run all test scripts in order
for test in test_*.py; do
    if [ -f "$test" ]; then
        run_test "$test" || exit 1
    fi
done

echo
echo -e "${GREEN}All tests passed successfully!${NC}"
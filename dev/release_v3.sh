#!/usr/bin/env bash
set -euo pipefail

# Get repo root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="v3.0.0"

echo "Running V3 release checks..."

# Step 1: Run test suite
echo "==> Running test suite..."
"${ROOT_DIR}/dev/run_tests.sh"

# Step 2: Check weight stability
echo -e "\n==> Checking weight stability..."
"${ROOT_DIR}/dev/diff_weights.sh"

# If we get here, all checks passed
echo -e "\n✓ All checks passed!"

# Create local tag
git tag -a "$TAG" -m "V3: Portfolio receipts, macro integration, enhanced metrics"
echo -e "\n✓ Local tag $TAG created"

# Instructions for pushing
echo -e "\nTo complete the release, run these commands:"
echo -e "  git push origin $TAG            # Push just the tag"
echo -e "  git push origin main $TAG       # Push commits and tag"
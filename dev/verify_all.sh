#!/usr/bin/env bash
set -euo pipefail

# Print build version
echo "[verify_all] Build version: $(python3 -c 'from core.utils.version import get_build_version; print(get_build_version())')"

# Run validator
python3 dev/validate_simulations.py --objective balanced --n-candidates 8

# Run acceptance
python3 dev/acceptance_v3.py

# Idempotent build-test
make build-test

# On success, print summary location
echo "[verify_all] PASS: All checks succeeded. See dev/builds/$(python3 -c 'from core.utils.version import get_build_version; print(get_build_version())')/summary.json"

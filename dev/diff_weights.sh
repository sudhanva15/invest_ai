#!/usr/bin/env bash
set -euo pipefail

# Get repo root directory
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Choose Python: prefer project venv if available, fallback to PYTHON env var or python3
PY_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
    PY_BIN="${PYTHON:-python3}"
fi

BASELINE_FILE="${ROOT_DIR}/tests/fixtures/weights_baseline.json"

# Check baseline exists before running snapshot
if [[ ! -f "$BASELINE_FILE" ]]; then
    echo "Error: Baseline file not found at: $BASELINE_FILE" >&2
    echo "" >&2
    echo "To create a baseline, run:" >&2
    echo "  ${PYTHON:-python3} dev/snapshot_weights.py --update-baseline" >&2
    exit 1
fi

# Generate current snapshot
SNAPSHOT=$("$PY_BIN" "${ROOT_DIR}/dev/snapshot_weights.py")

# Create temp file for current snapshot
TEMP_FILE=$(mktemp)
echo "$SNAPSHOT" > "$TEMP_FILE"

# Python script to compare weights
"$PY_BIN" - "$TEMP_FILE" "$BASELINE_FILE" <<'EOF'
import sys
import json
from pathlib import Path

def load_weights(path):
    return json.loads(Path(path).read_text())

def compare_weights(current, baseline, threshold=0.02):
    all_symbols = sorted(set(current) | set(baseline))
    max_delta = 0
    deltas = {}
    
    for sym in all_symbols:
        curr = current.get(sym, 0)
        base = baseline.get(sym, 0)
        delta = abs(curr - base)
        if delta > threshold:
            deltas[sym] = delta
        max_delta = max(max_delta, delta)
    
    return max_delta, deltas

current = load_weights(sys.argv[1])
baseline = load_weights(sys.argv[2])
max_delta, significant = compare_weights(current, baseline)

# Print differences
print(f"\nComparing against baseline: {Path(sys.argv[2]).name}")
if significant:
    print("\nSignificant weight changes (>2%):")
    for sym, delta in significant.items():
        curr = current.get(sym, 0)
        base = baseline.get(sym, 0)
        print(f"  {sym}: {base:.4f} -> {curr:.4f} (Δ {delta:.4f})")
    sys.exit(1)
else:
    print("✓ No significant weight changes detected (all Δ ≤ 2%)")
    sys.exit(0)
EOF

# Clean up temp file
rm "$TEMP_FILE"
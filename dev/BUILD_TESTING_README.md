## Build Test Harness - Release Engineering

**IDEMPOTENT** build validation for Invest_AI V3. Test any build locally or in CI without changing repository state.

### Quick Start

```bash
make build-test              # Test current build (idempotent)
make build-list              # List all tested builds  
make build-diff BUILD1=V3.0.1+42 BUILD2=V3.0.1+45  # Compare builds
make build-test-force        # Force re-test
```

### Files

- **`dev/test_build.sh`** - Main test harness (bash, executable)
- **`dev/compare_builds.py`** - Build comparison tool (python, executable)
- **`dev/BUILD_TESTING.md`** - Complete documentation (550 lines)
- **`dev/build_test_cheatsheet.sh`** - Quick reference card
- **`.github/workflows/build-test.yml`** - CI integration
- **`Makefile`** - Root-level targets (build-test, build-list, build-diff)

### Architecture

```
dev/builds/${BUILD}/
  ├── summary.json       # Test results + metadata
  ├── validator.log      # validate_simulations.py output
  ├── acceptance.log     # acceptance_v3.py output  
  ├── scenario.log       # run_scenarios.py output
  └── artifacts/         # Copied from dev/artifacts/
```

### Key Features

✅ **Idempotent** - First run executes tests, subsequent runs skip (unless `--force`)  
✅ **Build Stamping** - Uses `core.utils.version.get_build_version()` for consistent naming  
✅ **Atomic Writes** - All outputs use `tmp → rename` pattern  
✅ **Zero New Dependencies** - Uses existing Python packages only  
✅ **CI-Ready** - GitHub Actions workflow with artifact upload

### Three-Phase Testing

1. **Validator** - `dev/validate_simulations.py` (data quality, returns, macro, candidates)
2. **Acceptance** - `dev/acceptance_v3.py` (candidates, shortlist, exports)  
3. **Scenario** - `dev/run_scenarios.py` (end-to-end smoke test)

**Status Logic**: `PASS` = validator AND acceptance pass, `FAIL` = either fails, scenario is informational

### Example Output

```json
{
  "build": "V3.0.1+42.dirty",
  "timestamp": "2025-11-10T17:55:34",
  "tests": {
    "validator_pass": true,
    "acceptance_pass": true,
    "scenario_pass": false
  },
  "overall_status": "PASS"
}
```

### CI Integration

Workflow triggers on push/PR to `main`/`develop`:

1. Checkout with full history (for version stamping)
2. Setup Python 3.11 + virtualenv
3. Install dependencies from `requirements.txt`
4. Run `dev/test_build.sh`
5. Upload artifacts (30-day retention)
6. Check status (exit 1 if FAIL)

**Required Secrets** (optional): `FRED_API_KEY`, `TIINGO_API_KEY`

### Workflow Examples

**Pre-commit check:**
```bash
make build-test && git commit -m "feat: my change"
```

**Compare feature branch vs main:**
```bash
git checkout main && make build-test
MAIN=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")

git checkout feature/my-branch && make build-test  
FEAT=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")

make build-diff BUILD1=$MAIN BUILD2=$FEAT
```

**Debugging failed build:**
```bash
make build-test
BUILD=$(python3 -c "import sys; sys.path.insert(0, '.'); from core.utils.version import get_build_version; print(get_build_version())")

cat "dev/builds/${BUILD}/validator.log"    # Check validator failures
cat "dev/builds/${BUILD}/acceptance.log"   # Check acceptance issues
ls -la "dev/builds/${BUILD}/artifacts/"    # Inspect artifacts
```

### Documentation

See **`dev/BUILD_TESTING.md`** for:
- Complete architecture overview
- Detailed usage examples  
- Troubleshooting guide
- Best practices
- Summary JSON schema

See **`dev/build_test_cheatsheet.sh`** for quick reference commands.

### Status

✅ **Implementation Complete**  
✅ **Idempotency Verified**  
✅ **Makefile Targets Functional**  
✅ **CI Workflow Ready**  
✅ **Documentation Complete**

---

**Next Steps:**
1. Add GitHub secrets (FRED_API_KEY, TIINGO_API_KEY) for CI
2. Push to trigger workflow  
3. Use `make build-diff` for PR reviews

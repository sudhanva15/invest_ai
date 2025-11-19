# Invest_AI V3 Root Makefile
# Convenience wrapper for scenario runner and build testing

PYTHON := python3

.PHONY: help
help:
	@echo "Invest_AI V3 - Available Targets"
	@echo ""
	@echo "BUILD TESTING:"
	@echo "  make build-test           Test current build (idempotent)"
	@echo "  make build-test-force     Force re-test current build"
	@echo "  make build-list           List all tested builds"
	@echo "  make build-diff BUILD1=<b1> BUILD2=<b2>  Compare builds"
	@echo ""
	@echo "SCENARIO RUNNERS:"
	@echo "  make run-balanced         Run balanced objective"
	@echo "  make run-growth           Run growth objective"
	@echo "  make run-stress           Run with stress tests"
	@echo ""
	@echo "VALIDATION:"
	@echo "  make validate             Run V3 validation suite"
	@echo "  make test-all-v3          Run all V3 tests"
	@echo ""
	@echo "See dev/Makefile for more targets"
	@echo ""
	@echo "TESTS & SMOKES:"
	@echo "  make test-unit             Run full pytest suite"
	@echo "  make test-risk-profile     Run RiskProfile integration test"
	@echo "  make smoke-phase3          Run Phase 3 multi-factor smoke script"

# =============================================================================
# BUILD TESTING
# =============================================================================

.PHONY: build-test
build-test:
	@bash dev/test_build.sh

.PHONY: build-test-force
build-test-force:
	@bash dev/test_build.sh --force

.PHONY: build-list
build-list:
	@$(PYTHON) dev/compare_builds.py --list

.PHONY: build-diff
build-diff:
	@if [ -z "$(BUILD1)" ] || [ -z "$(BUILD2)" ]; then \
		echo "Usage: make build-diff BUILD1=<build1> BUILD2=<build2>"; \
		exit 1; \
	fi
	@$(PYTHON) dev/compare_builds.py "$(BUILD1)" "$(BUILD2)"

# =============================================================================
# SCENARIO RUNNERS (delegate to dev/Makefile)
# =============================================================================

.PHONY: run-balanced run-growth run-stress validate test-all-v3
run-balanced run-growth run-stress validate test-all-v3:
	@$(MAKE) -C dev -f Makefile $@

# =============================================================================
# VERIFICATION
# =============================================================================

.PHONY: verify-all
verify-all:
	bash dev/verify_all.sh

# =============================================================================
# LOCAL TESTS & SMOKES (Phase 3 enhanced)
# =============================================================================

.PHONY: test-unit test-risk-profile smoke-phase3 test-phase3 test-task1 test-task2 verify-tasks

test-unit:
	PYTHONPATH=. pytest

test-risk-profile:
	PYTHONPATH=. pytest tests/test_risk_profile_integration.py -q

smoke-phase3:
	@echo "Running Phase 3 smoke test (A-Z verification)..."
	PYTHONPATH=. .venv/bin/python dev/smoke_phase3.py

# Phase 3 Task-specific tests
test-task1:
	@echo "Testing Task 1: Risk → CAGR mapping..."
	PYTHONPATH=. .venv/bin/pytest tests/test_risk_profile_cagr_mapping.py -v

test-task2:
	@echo "Testing Task 2: Adaptive portfolio thresholds..."
	PYTHONPATH=. .venv/bin/pytest tests/test_adaptive_thresholds.py -v

test-phase3: test-task1 test-task2 smoke-phase3
	@echo "✅ All Phase 3 tests passed!"

verify-tasks:
	@echo "========================================="
	@echo "Phase 3 Task Verification"
	@echo "========================================="
	@echo "Task 1 (Risk → CAGR):"
	@PYTHONPATH=. .venv/bin/pytest tests/test_risk_profile_cagr_mapping.py -v --tb=short
	@echo ""
	@echo "Task 2 (Adaptive Thresholds):"
	@PYTHONPATH=. .venv/bin/pytest tests/test_adaptive_thresholds.py -v --tb=short
	@echo ""
	@echo "Task 3 (4-Stage Fallback):"
	@PYTHONPATH=. .venv/bin/python dev/smoke_phase3.py
	@echo ""
	@echo "========================================="
	@echo "✅ ALL PHASE 3 TASKS VERIFIED"
	@echo "========================================="

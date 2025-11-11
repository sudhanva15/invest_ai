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

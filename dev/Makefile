# Makefile for Portfolio Scenario Runner
# Convenience targets for common CLI operations

SHELL := /bin/bash
PYTHON := python3
# Use python3 directly - assumes venv is activated or python3 is in PATH
VENV := python3

# Default target
.PHONY: help
help:
	@echo "Portfolio Scenario Runner - Available Targets:"
	@echo ""
	@echo "BUILD TESTING:"
	@echo "  make build-test           Test current build (idempotent)"
	@echo "  make build-test-force     Force re-test current build"
	@echo "  make build-list           List all tested builds"
	@echo "  make build-diff BUILD1 BUILD2  Compare two builds"
	@echo ""
	@echo "SCENARIO RUNNERS:"
	@echo "  make run-balanced         Run balanced objective (n=8)"
	@echo "  make run-growth           Run growth objective (n=10)"
	@echo "  make run-income           Run income objective with custom config"
	@echo "  make run-stress           Run balanced with equity shock"
	@echo "  make run-custom           Run custom ticker universe"
	@echo ""
	@echo "TESTING:"
	@echo "  make test-scenarios       Run CLI unit tests"
	@echo "  make test-all-v3          Run all V3 tests"
	@echo ""
	@echo "CLEANUP:"
	@echo "  make clean                Clean artifacts directory"
	@echo ""

# Run balanced objective with default parameters
.PHONY: run-balanced
run-balanced:
	@echo "Running balanced objective (n=8)..."
	$(VENV) dev/run_scenarios.py \
		--objective balanced \
		--n-candidates 8 \
		--verbose

# Run growth objective with higher candidate count
.PHONY: run-growth
run-growth:
	@echo "Running growth objective (n=10)..."
	$(VENV) dev/run_scenarios.py \
		--objective growth \
		--n-candidates 10 \
		--horizon 5y \
		--verbose

# Run income objective with custom optimizers and satellite caps
.PHONY: run-income
run-income:
	@echo "Running income objective with custom config..."
	$(VENV) dev/run_scenarios.py \
		--objective income \
		--optimizers hrp,max_sharpe,risk_parity \
		--satellite-caps 0.15,0.20,0.25 \
		--n-candidates 9 \
		--verbose

# Run balanced with equity stress test
.PHONY: run-stress
run-stress:
	@echo "Running balanced with equity-10% shock..."
	$(VENV) dev/run_scenarios.py \
		--objective balanced \
		--shock equity-10% \
		--shock rates+100bp \
		--n-candidates 8 \
		--verbose

# Run custom ticker universe
.PHONY: run-custom
run-custom:
	@echo "Running custom ticker universe (tech focus)..."
	$(VENV) dev/run_scenarios.py \
		--objective growth \
		--tickers SPY,QQQ,VGT,XLK,MSFT,GOOGL,TLT \
		--start 2019-01-01 \
		--n-candidates 8 \
		--verbose

# Run barbell with macro override
.PHONY: run-barbell
run-barbell:
	@echo "Running barbell objective with recessionary regime..."
	$(VENV) dev/run_scenarios.py \
		--objective barbell \
		--macro-override '{"regime":"Recessionary"}' \
		--n-candidates 8 \
		--verbose

# Run CLI unit tests
.PHONY: test-scenarios
test-scenarios:
	@echo "Running CLI unit tests..."
	$(VENV) -m unittest dev/test_scenarios.py -v

# Run all V3 unit tests
.PHONY: test-all-v3
test-all-v3:
	@echo "Running all V3 unit tests..."
	$(VENV) -m unittest discover -s dev -p "test_*.py" -v

# Clean artifacts directory
.PHONY: clean
clean:
	@echo "Cleaning artifacts directory..."
	rm -f dev/artifacts/*.json
	rm -f dev/artifacts/*.csv
	@echo "Done."

# Quick smoke test (fast, minimal data)
.PHONY: smoke
smoke:
	@echo "Running quick smoke test..."
	$(VENV) dev/run_scenarios.py \
		--objective balanced \
		--tickers SPY,TLT \
		--start 2020-01-01 \
		--n-candidates 3 \
		--verbose

# V3 Validation
.PHONY: validate
validate:
	@echo "Running V3 simulation validator..."
	$(VENV) dev/validate_simulations.py \
		--objective balanced \
		--n-candidates 6 \
		--verbose

# V3 Validation (JSON output)
.PHONY: validate-json
validate-json:
	@echo "Running V3 validator (JSON output)..."
	$(VENV) dev/validate_simulations.py \
		--objective growth \
		--n-candidates 8 \
		--json

# Test validator
.PHONY: test-validator
test-validator:
	@echo "Running validator unit tests..."
	$(VENV) -m unittest dev/test_validate_simulations.py -v

# =============================================================================
# BUILD TESTING TARGETS (V3 Release Engineering)
# =============================================================================
# NOTE: These targets expect to be run from repo root via: make -f dev/Makefile target
# Or via the root Makefile which includes this file

# Test current build (idempotent - skips if already tested)
.PHONY: build-test
build-test:
	@echo "Testing current build..."
	@bash dev/test_build.sh

# Force re-test current build
.PHONY: build-test-force
build-test-force:
	@echo "Force re-testing current build..."
	@bash dev/test_build.sh --force

# List all available builds
.PHONY: build-list
build-list:
	@echo "Listing available builds..."
	@$(VENV) dev/compare_builds.py --list

# Compare two builds (usage: make build-diff BUILD1=V3.0.1+42 BUILD2=V3.0.1+45)
.PHONY: build-diff
build-diff:
	@if [ -z "$(BUILD1)" ] || [ -z "$(BUILD2)" ]; then \
		echo "Usage: make build-diff BUILD1=<build1> BUILD2=<build2>"; \
		echo "Example: make build-diff BUILD1=V3.0.1+42 BUILD2=V3.0.1+45"; \
		exit 1; \
	fi
	@$(VENV) dev/compare_builds.py "$(BUILD1)" "$(BUILD2)"

#!/usr/bin/env python3
"""Unit tests for V3 validation tool (dev/validate_simulations.py)

Tests the validator without requiring network access by using mock data.
Focus on parser, structure, and internal logic validation.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dev.validate_simulations import (
        ValidationResult,
        check_data,
        check_returns,
        check_candidates,
        check_metrics,
        check_receipts,
    )
except ImportError as e:
    print(f"ERROR: Cannot import validation module: {e}")
    sys.exit(1)

from core.utils.metrics import align_returns_matrix, assert_metrics_consistency, annualized_metrics



class TestValidationResult(unittest.TestCase):
    """Test ValidationResult container."""
    
    def test_init(self):
        """Test ValidationResult initialization."""
        result = ValidationResult("Test Check")
        self.assertEqual(result.name, "Test Check")
        self.assertTrue(result.passed)
        self.assertEqual(result.messages, [])
        self.assertEqual(result.warnings, [])
    
    def test_fail(self):
        """Test fail method marks check as failed."""
        result = ValidationResult("Test")
        result.fail("Something went wrong")
        self.assertFalse(result.passed)
        self.assertIn("✗ Something went wrong", result.messages)
    
    def test_warn(self):
        """Test warn doesn't fail check."""
        result = ValidationResult("Test")
        result.warn("Minor issue")
        self.assertTrue(result.passed)
        self.assertIn("⚠ Minor issue", result.warnings)
    
    def test_succeed(self):
        """Test succeed adds success message."""
        result = ValidationResult("Test")
        result.succeed("All good")
        self.assertTrue(result.passed)
        self.assertIn("✓ All good", result.messages)


class TestCheckData(unittest.TestCase):
    """Test data quality checks."""
    
    def setUp(self):
        """Create mock price data."""
        dates = pd.date_range("2015-01-01", "2020-01-01", freq="D")
        self.prices = pd.DataFrame(
            {
                "SPY": np.random.randn(len(dates)).cumsum() + 100,
                "TLT": np.random.randn(len(dates)).cumsum() + 80,
            },
            index=dates
        )
        self.provenance = {"SPY": "stooq", "TLT": "tiingo"}
    
    def test_all_tickers_present(self):
        """Test check passes when all tickers present."""
        result = check_data(["SPY", "TLT"], self.prices, self.provenance)
        self.assertTrue(result.passed)
    
    def test_missing_ticker(self):
        """Test check fails when ticker missing."""
        result = check_data(["SPY", "TLT", "GLD"], self.prices, self.provenance)
        self.assertFalse(result.passed)
    
    def test_nan_ratio(self):
        """Test NaN ratio check."""
        # Add excessive NaNs to TLT
        self.prices.loc[self.prices.index[:300], "TLT"] = np.nan
        result = check_data(["SPY", "TLT"], self.prices, self.provenance)
        self.assertFalse(result.passed)
    
    def test_span_too_short(self):
        """Test span check fails for short history."""
        short_prices = self.prices.iloc[:365]  # 1 year only
        result = check_data(["SPY", "TLT"], short_prices, self.provenance)
        self.assertFalse(result.passed)
    
    def test_span_sufficient(self):
        """Test span check passes for long history."""
        result = check_data(["SPY", "TLT"], self.prices, self.provenance)
        self.assertTrue(result.passed)


class TestCheckReturns(unittest.TestCase):
    """Test returns quality checks."""
    
    def setUp(self):
        """Create mock returns data."""
        dates = pd.date_range("2015-01-01", "2020-01-01", freq="D")
        self.returns = pd.DataFrame(
            {
                "SPY": np.random.randn(len(dates)) * 0.01,
                "TLT": np.random.randn(len(dates)) * 0.008,
            },
            index=dates
        )
    
    def test_clean_returns(self):
        """Test check passes for clean returns."""
        result = check_returns(self.returns)
        self.assertTrue(result.passed)
    
    def test_nan_detection(self):
        """Test check fails with NaN values."""
        self.returns.iloc[10, 0] = np.nan
        result = check_returns(self.returns)
        self.assertFalse(result.passed)
    
    def test_infinity_detection(self):
        """Test check fails with infinite values."""
        self.returns.iloc[20, 1] = np.inf
        result = check_returns(self.returns)
        self.assertFalse(result.passed)
    
    def test_excessive_returns(self):
        """Test check fails for pathological returns."""
        self.returns[:] = 0.15  # 15% daily returns for all columns
        result = check_returns(self.returns)
        self.assertFalse(result.passed)


class TestCheckCandidates(unittest.TestCase):
    """Test candidate validation checks."""
    
    def setUp(self):
        """Create mock candidates."""
        self.candidates = [
            {
                "name": "MAX_SHARPE - Sat 20%",
                "weights": {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1},
                "metrics": {"CAGR": 0.08, "Sharpe": 0.9, "MaxDD": -0.15}
            },
            {
                "name": "HRP - Sat 25%",
                "weights": {"SPY": 0.5, "TLT": 0.35, "GLD": 0.15},
                "metrics": {"CAGR": 0.07, "Sharpe": 0.85, "MaxDD": -0.12}
            }
        ]
    
    def test_sufficient_candidates(self):
        """Test check passes with sufficient candidates."""
        result = check_candidates(self.candidates, n_expected=2, objective_name="balanced")
        self.assertTrue(result.passed)
    
    def test_insufficient_candidates(self):
        """Test check fails with too few candidates."""
        result = check_candidates(self.candidates, n_expected=5, objective_name="balanced")
        self.assertFalse(result.passed)
    
    def test_weights_sum_to_one(self):
        """Test weights validation."""
        # Add bad candidate with weights not summing to 1
        bad_cand = {
            "name": "Bad Weights",
            "weights": {"SPY": 0.5, "TLT": 0.3},  # Sum = 0.8
            "metrics": {}
        }
        result = check_candidates(
            self.candidates + [bad_cand],
            n_expected=2,
            objective_name="balanced"
        )
        self.assertFalse(result.passed)
    
    def test_negative_weights(self):
        """Test negative weight detection."""
        bad_cand = {
            "name": "Negative Weights",
            "weights": {"SPY": 1.2, "TLT": -0.2},  # Negative weight
            "metrics": {}
        }
        result = check_candidates(
            self.candidates + [bad_cand],
            n_expected=2,
            objective_name="balanced"
        )
        self.assertFalse(result.passed)


class TestCheckMetrics(unittest.TestCase):
    """Test metrics plausibility checks."""
    
    def setUp(self):
        """Create mock candidates with metrics."""
        self.candidates = [
            {
                "name": "Candidate 1",
                "metrics": {"CAGR": 0.08, "Sharpe": 0.9, "MaxDD": -0.15}
            },
            {
                "name": "Candidate 2",
                "metrics": {"CAGR": 0.06, "Sharpe": 0.75, "MaxDD": -0.12}
            }
        ]
    
    def test_plausible_metrics(self):
        """Test check passes for plausible metrics."""
        result = check_metrics(self.candidates)
        self.assertTrue(result.passed)
    
    def test_implausible_cagr(self):
        """Test check fails for implausible CAGR."""
        self.candidates[0]["metrics"]["CAGR"] = 1.5  # 150% CAGR
        result = check_metrics(self.candidates)
        self.assertFalse(result.passed)
    
    def test_implausible_sharpe(self):
        """Test check fails for implausible Sharpe."""
        self.candidates[0]["metrics"]["Sharpe"] = 5.0  # Too high
        result = check_metrics(self.candidates)
        self.assertFalse(result.passed)
    
    def test_implausible_maxdd(self):
        """Test check fails for implausible MaxDD."""
        self.candidates[0]["metrics"]["MaxDD"] = -1.5  # > 100% drawdown
        result = check_metrics(self.candidates)
        self.assertFalse(result.passed)


class TestCheckReceipts(unittest.TestCase):
    """Test receipt integrity checks."""
    
    def setUp(self):
        """Create mock receipts."""
        self.receipts = [
            {
                "ticker": "SPY",
                "provider": "stooq",
                "backfill_pct": "0.00",
                "first": "2010-01-01",
                "last": "2020-01-01",
                "nan_rate": 0.02,
                "n_points": 2500,
                "hist_years": 10.0
            },
            {
                "ticker": "TLT",
                "provider": "tiingo",
                "backfill_pct": "5.20",
                "first": "2010-01-01",
                "last": "2020-01-01",
                "nan_rate": 0.03,
                "n_points": 2450,
                "hist_years": 10.0
            }
        ]
        self.tickers = ["SPY", "TLT"]
    
    def test_valid_receipts(self):
        """Test check passes for valid receipts."""
        result = check_receipts(self.receipts, self.tickers)
        self.assertTrue(result.passed)
    
    def test_missing_keys(self):
        """Test check fails for missing required keys."""
        incomplete = [
            {
                "ticker": "SPY",
                "provider": "stooq",
                # Missing backfill_pct, first, last, nan_rate
            }
        ]
        result = check_receipts(incomplete, ["SPY"])
        self.assertFalse(result.passed)
    
    def test_count_mismatch(self):
        """Test warning for receipt count mismatch."""
        result = check_receipts(self.receipts, ["SPY", "TLT", "GLD"])
        # Should warn but not fail
        self.assertTrue(len(result.warnings) > 0)
    
    def test_invalid_nan_rate(self):
        """Test warning for invalid nan_rate values."""
        self.receipts[0]["nan_rate"] = 1.5  # > 1.0
        result = check_receipts(self.receipts, self.tickers)
        self.assertTrue(len(result.warnings) > 0)


class TestAlignmentAndConsistency(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        self.pr = pd.DataFrame({
            "SPY": np.random.randn(len(dates))*0.01,
            "TLT": np.random.randn(len(dates))*0.008,
            "GLD": np.random.randn(len(dates))*0.009,
        }, index=dates)
        # Introduce some leading NaNs
        self.pr.loc[self.pr.index[:5], "TLT"] = np.nan
        self.pr.loc[self.pr.index[:10], "GLD"] = np.nan

    def test_single_window_alignment(self):
        aligned = align_returns_matrix(self.pr, ["SPY","TLT","GLD"])
        # First rows should have no NaNs
        self.assertFalse(aligned.isna().any().any())
        # Alignment should drop at least 10 leading rows
        self.assertEqual(aligned.index[0], self.pr.index[10])

    def test_metrics_consistency_curves_vs_table(self):
        aligned = align_returns_matrix(self.pr, ["SPY","TLT"])  # 2 asset example
        w = pd.Series({"SPY":0.6, "TLT":0.4}).reindex(aligned.columns).fillna(0.0)
        port = (aligned * w).sum(axis=1)
        curve = (1+port).cumprod()
        ok = assert_metrics_consistency(curve, port)
        self.assertTrue(ok)

    def test_projection_uses_backtest_window(self):
        aligned = align_returns_matrix(self.pr, ["SPY","TLT","GLD"])  # common window
        w = pd.Series({"SPY":0.5, "TLT":0.3, "GLD":0.2}).reindex(aligned.columns).fillna(0.0)
        port = (aligned * w).sum(axis=1)
        m = annualized_metrics(port)
        # Sanity on N matching aligned length
        self.assertEqual(m.get("N"), len(port.dropna()))


class TestParserAndImports(unittest.TestCase):
    """Test module imports and argument parser."""
    
    def test_imports_available(self):
        """Test all required V3 components importable."""
        try:
            from core.data_ingestion import get_prices_with_provenance
            from core.portfolio_engine import clean_prices_to_returns
            from core.recommendation_engine import DEFAULT_OBJECTIVES, generate_candidates
            from core.utils.metrics import annualized_metrics
            from core.utils.receipts import build_receipts
            from core.data_sources.fred import load_series
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import failed: {e}")
    
    def test_default_objectives_available(self):
        """Test DEFAULT_OBJECTIVES is accessible."""
        from core.recommendation_engine import DEFAULT_OBJECTIVES
        self.assertIsInstance(DEFAULT_OBJECTIVES, dict)
        self.assertIn("balanced", DEFAULT_OBJECTIVES)
        self.assertIn("growth", DEFAULT_OBJECTIVES)


if __name__ == "__main__":
    # Run tests
    unittest.main(argv=[""], verbosity=2, exit=False)

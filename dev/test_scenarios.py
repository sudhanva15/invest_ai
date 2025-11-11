"""
Unit tests for portfolio scenario runner CLI tool.

Run with:
    python3 -m unittest dev/test_scenarios.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path
import unittest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dev"))

import pandas as pd
import numpy as np
from run_scenarios import (
    parse_shock,
    apply_shocks,
    compute_horizon_metrics,
    generate_custom_candidates,
)

# Import V3 modules
from core.data_ingestion import get_prices
from core.preprocessing import compute_returns
from core.recommendation_engine import DEFAULT_OBJECTIVES


class TestShockParsing(unittest.TestCase):
    """Test shock string parsing."""
    
    def test_equity_shock_negative(self):
        """Test parsing equity-10% shock."""
        asset_class, direction, magnitude = parse_shock("equity-10%")
        
        self.assertEqual(asset_class, "equity")
        self.assertEqual(direction, "-")
        self.assertAlmostEqual(magnitude, 0.10, places=4)
    
    def test_rates_shock_positive(self):
        """Test parsing rates+100bp shock."""
        asset_class, direction, magnitude = parse_shock("rates+100bp")
        
        self.assertEqual(asset_class, "rates")
        self.assertEqual(direction, "+")
        self.assertAlmostEqual(magnitude, 0.01, places=4)  # 100bp = 1%
    
    def test_gold_shock_positive(self):
        """Test parsing gold+5% shock."""
        asset_class, direction, magnitude = parse_shock("gold+5%")
        
        self.assertEqual(asset_class, "gold")
        self.assertEqual(direction, "+")
        self.assertAlmostEqual(magnitude, 0.05, places=4)
    
    def test_case_insensitive(self):
        """Test case-insensitive parsing."""
        asset_class, direction, magnitude = parse_shock("EQUITY-15%")
        
        self.assertEqual(asset_class, "equity")
        self.assertEqual(direction, "-")
        self.assertAlmostEqual(magnitude, 0.15, places=4)
    
    def test_decimal_format(self):
        """Test decimal format without %."""
        asset_class, direction, magnitude = parse_shock("equity-0.12")
        
        self.assertEqual(asset_class, "equity")
        self.assertEqual(direction, "-")
        self.assertAlmostEqual(magnitude, 0.12, places=4)


class TestShockApplication(unittest.TestCase):
    """Test shock application to returns."""
    
    def setUp(self):
        """Create synthetic returns data."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        
        self.returns = pd.DataFrame({
            "SPY": np.random.randn(100) * 0.01,
            "QQQ": np.random.randn(100) * 0.015,
            "TLT": np.random.randn(100) * 0.008,
            "GLD": np.random.randn(100) * 0.012,
        }, index=dates)
        
        # Simple catalog
        self.catalog = {
            "SPY": {"asset_class": "public_equity"},
            "QQQ": {"asset_class": "public_equity"},
            "TLT": {"asset_class": "treasury_long"},
            "GLD": {"asset_class": "gold"},
        }
    
    def test_equity_shock_negative(self):
        """Test equity-10% shock reduces equity returns at t=0."""
        shocked = apply_shocks(
            self.returns,
            ["equity-10%"],
            self.catalog,
            verbose=False
        )
        
        # Check that SPY and QQQ were affected at t=0
        original_spy = self.returns.iloc[0]["SPY"]
        shocked_spy = shocked.iloc[0]["SPY"]
        
        self.assertLess(shocked_spy, original_spy)
        self.assertAlmostEqual(shocked_spy, original_spy * 0.9, places=6)
        
        # TLT and GLD should be unchanged
        self.assertAlmostEqual(
            shocked.iloc[0]["TLT"],
            self.returns.iloc[0]["TLT"],
            places=6
        )
    
    def test_rates_shock_positive(self):
        """Test rates+100bp shock reduces bond returns at t=0."""
        shocked = apply_shocks(
            self.returns,
            ["rates+100bp"],
            self.catalog,
            verbose=False
        )
        
        # TLT should be negatively affected (duration ~17yr)
        original_tlt = self.returns.iloc[0]["TLT"]
        shocked_tlt = shocked.iloc[0]["TLT"]
        
        # Expected impact: -17% for 100bp rise
        self.assertLess(shocked_tlt, original_tlt)
        
        # SPY and QQQ should be unchanged
        self.assertAlmostEqual(
            shocked.iloc[0]["SPY"],
            self.returns.iloc[0]["SPY"],
            places=6
        )
    
    def test_gold_shock_positive(self):
        """Test gold+5% shock increases gold returns at t=0."""
        shocked = apply_shocks(
            self.returns,
            ["gold+5%"],
            self.catalog,
            verbose=False
        )
        
        # GLD should be positively affected (multiplied by 1.05)
        original_gld = self.returns.iloc[0]["GLD"]
        shocked_gld = shocked.iloc[0]["GLD"]
        
        # Check that shock was applied (5% increase in magnitude)
        self.assertAlmostEqual(shocked_gld, original_gld * 1.05, places=6)
        
        # If original was negative, shocked will be more negative
        # If original was positive, shocked will be more positive
        if original_gld >= 0:
            self.assertGreater(shocked_gld, original_gld)
        else:
            self.assertLess(shocked_gld, original_gld)  # More negative
    
    def test_multiple_shocks(self):
        """Test applying multiple shocks simultaneously."""
        shocked = apply_shocks(
            self.returns,
            ["equity-10%", "gold+5%"],
            self.catalog,
            verbose=False
        )
        
        # Both shocks should be applied
        original_spy = self.returns.iloc[0]["SPY"]
        shocked_spy = shocked.iloc[0]["SPY"]
        
        original_gld = self.returns.iloc[0]["GLD"]
        shocked_gld = shocked.iloc[0]["GLD"]
        
        # Equity shock reduces returns
        self.assertLess(shocked_spy, original_spy)
        
        # Gold shock multiplies by 1.05
        self.assertAlmostEqual(shocked_gld, original_gld * 1.05, places=6)
    
    def test_no_shocks(self):
        """Test that empty shock list returns unchanged data."""
        shocked = apply_shocks(self.returns, [], self.catalog, verbose=False)
        
        pd.testing.assert_frame_equal(shocked, self.returns)


class TestHorizonMetrics(unittest.TestCase):
    """Test horizon-specific metrics computation."""
    
    def setUp(self):
        """Create synthetic returns for different horizons."""
        np.random.seed(42)
        
        # 10 years of daily returns
        dates = pd.date_range("2013-01-01", periods=252*10, freq="D")
        self.returns = pd.Series(
            np.random.randn(252*10) * 0.01 + 0.0003,  # Positive drift
            index=dates
        )
    
    def test_1y_horizon(self):
        """Test 1-year horizon metrics."""
        metrics = compute_horizon_metrics(self.returns, "1y")
        
        self.assertIn("CAGR", metrics)
        self.assertIn("Sharpe", metrics)
        self.assertIn("MaxDD", metrics)
        self.assertIn("Volatility", metrics)
    
    def test_5y_horizon(self):
        """Test 5-year horizon metrics."""
        metrics = compute_horizon_metrics(self.returns, "5y")
        
        self.assertIn("CAGR", metrics)
        self.assertIsNotNone(metrics["CAGR"])
    
    def test_10y_horizon(self):
        """Test 10-year horizon metrics."""
        metrics = compute_horizon_metrics(self.returns, "10y")
        
        self.assertIn("CAGR", metrics)
        self.assertIsNotNone(metrics["CAGR"])
    
    def test_insufficient_data(self):
        """Test that metrics work even with less data than horizon."""
        # Use only 1 year of data for 5-year horizon
        short_returns = self.returns.tail(252)
        
        metrics = compute_horizon_metrics(short_returns, "5y")
        
        # Should still return metrics using available data
        self.assertIn("CAGR", metrics)
        self.assertIsNotNone(metrics["CAGR"])


class TestCandidateGeneration(unittest.TestCase):
    """Test custom candidate generation with overrides."""
    
    @classmethod
    def setUpClass(cls):
        """Load real price data once for all tests."""
        try:
            cls.prices = get_prices(["SPY", "TLT", "GLD"], start="2018-01-01")
            cls.returns = compute_returns(cls.prices)
            
            # Load catalog
            try:
                from core.utils import load_json
                cls.catalog = load_json(str(ROOT / "config/assets_catalog.json"))
            except Exception:
                with open(ROOT / "config/assets_catalog.json") as f:
                    cls.catalog = json.load(f)
        
        except Exception as e:
            print(f"Warning: Could not load data for integration tests: {e}")
            cls.prices = None
            cls.returns = None
            cls.catalog = None
    
    def test_smoke_balanced(self):
        """Smoke test: generate balanced candidates."""
        if self.returns is None:
            self.skipTest("No price data available")
        
        objective_cfg = DEFAULT_OBJECTIVES["balanced"]
        
        candidates = generate_custom_candidates(
            self.returns,
            objective_cfg,
            self.catalog,
            n_candidates=5,
            verbose=False
        )
        
        # Should generate at least 3 candidates
        self.assertGreaterEqual(len(candidates), 3)
        
        # Each candidate should have weights and metrics
        for c in candidates:
            self.assertIn("weights", c)
            self.assertIn("metrics", c)
            self.assertIn("name", c)
            
            # Weights should sum to ~1
            self.assertAlmostEqual(sum(c["weights"].values()), 1.0, places=2)
    
    def test_custom_satellite_caps(self):
        """Test custom satellite caps."""
        if self.returns is None:
            self.skipTest("No price data available")
        
        objective_cfg = DEFAULT_OBJECTIVES["balanced"]
        
        candidates = generate_custom_candidates(
            self.returns,
            objective_cfg,
            self.catalog,
            n_candidates=5,
            custom_sat_caps=[0.15, 0.25],
            verbose=False
        )
        
        # Should generate candidates with custom satellite caps
        self.assertGreaterEqual(len(candidates), 2)
        
        # Check that sat_cap field is present
        for c in candidates:
            self.assertIn("sat_cap", c)
            self.assertIn(c["sat_cap"], [0.15, 0.25])
    
    def test_custom_optimizers(self):
        """Test custom optimizers."""
        if self.returns is None:
            self.skipTest("No price data available")
        
        objective_cfg = DEFAULT_OBJECTIVES["growth"]
        
        candidates = generate_custom_candidates(
            self.returns,
            objective_cfg,
            self.catalog,
            n_candidates=4,
            custom_optimizers=["hrp", "equal_weight"],
            verbose=False
        )
        
        # Should generate candidates with custom optimizers
        self.assertGreaterEqual(len(candidates), 2)
        
        # Check that optimizer field is present
        for c in candidates:
            self.assertIn("optimizer", c)
            self.assertIn(c["optimizer"], ["hrp", "equal_weight"])
    
    def test_constraints_enforced(self):
        """Test that objective constraints are enforced."""
        if self.returns is None:
            self.skipTest("No price data available")
        
        objective_cfg = DEFAULT_OBJECTIVES["income"]
        
        candidates = generate_custom_candidates(
            self.returns,
            objective_cfg,
            self.catalog,
            n_candidates=5,
            verbose=False
        )
        
        for c in candidates:
            # Weights should sum to ~1
            total = sum(c["weights"].values())
            self.assertAlmostEqual(total, 1.0, places=2)
            
            # All weights should be non-negative
            for w in c["weights"].values():
                self.assertGreaterEqual(w, 0.0)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests (optional, requires data)."""
    
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        try:
            cls.prices = get_prices(["SPY", "TLT"], start="2018-01-01")
            cls.returns = compute_returns(cls.prices)
            
            # Load catalog
            try:
                from core.utils import load_json
                cls.catalog = load_json(str(ROOT / "config/assets_catalog.json"))
            except Exception:
                with open(ROOT / "config/assets_catalog.json") as f:
                    cls.catalog = json.load(f)
        
        except Exception as e:
            print(f"Warning: Could not load data for integration tests: {e}")
            cls.prices = None
            cls.returns = None
            cls.catalog = None
    
    def test_full_workflow_with_shocks(self):
        """Test full workflow: load data → apply shocks → generate candidates → score."""
        if self.returns is None:
            self.skipTest("No price data available")
        
        # Apply shock
        shocked_returns = apply_shocks(
            self.returns,
            ["equity-10%"],
            self.catalog,
            verbose=False
        )
        
        # Generate candidates
        objective_cfg = DEFAULT_OBJECTIVES["balanced"]
        candidates = generate_custom_candidates(
            shocked_returns,
            objective_cfg,
            self.catalog,
            n_candidates=3,
            verbose=False
        )
        
        # Should have candidates
        self.assertGreaterEqual(len(candidates), 1)
        
        # Compute scores
        for c in candidates:
            weights_vec = pd.Series(c["weights"]).reindex(shocked_returns.columns).fillna(0.0)
            port_ret = (shocked_returns * weights_vec).sum(axis=1)
            
            metrics = compute_horizon_metrics(port_ret, "3y")
            
            sharpe = metrics.get("Sharpe", 0.0)
            maxdd = metrics.get("MaxDD", 0.0)
            score = sharpe - 0.2 * abs(maxdd)
            
            c["score"] = score
        
        # All candidates should have scores
        for c in candidates:
            self.assertIn("score", c)
            self.assertIsNotNone(c["score"])


if __name__ == "__main__":
    unittest.main()

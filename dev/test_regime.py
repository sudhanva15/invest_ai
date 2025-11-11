#!/usr/bin/env python3
"""Unit tests for core/macro/regime.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd
import numpy as np
from core.macro.regime import (
    regime_features,
    label_regimes,
    regime_performance,
    current_regime,
)


class TestRegimeFeatures(unittest.TestCase):
    """Test regime feature computation."""
    
    def setUp(self):
        """Create test macro data."""
        dates = pd.date_range("2020-01-01", periods=500, freq="B")
        
        self.macro_df = pd.DataFrame({
            "DGS10": np.random.uniform(1.5, 3.5, 500),
            "CPIAUCSL": 250 + np.cumsum(np.random.uniform(0, 0.5, 500)),  # Rising CPI
            "UNRATE": np.random.uniform(3.5, 6.5, 500),
            "FEDFUNDS": np.random.uniform(0.5, 2.5, 500),
        }, index=dates)
    
    def test_features_shape(self):
        """Test that features have expected columns."""
        features = regime_features(self.macro_df)
        
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features), 0)
        
        # Should have at least some of these columns
        expected_cols = {"dgs10_level", "dgs10_6m_chg", "cpi_yoy", "unrate_6m_chg"}
        actual_cols = set(features.columns)
        
        self.assertTrue(len(actual_cols.intersection(expected_cols)) > 0)
    
    def test_features_zscore(self):
        """Test that features are z-scored (mean ~0, std ~1)."""
        features = regime_features(self.macro_df)
        
        for col in features.columns:
            mean = features[col].mean()
            std = features[col].std()
            
            # Z-scored: mean ~0, std ~1 (with tolerance)
            self.assertAlmostEqual(mean, 0.0, delta=0.2)
            self.assertAlmostEqual(std, 1.0, delta=0.2)
    
    def test_empty_input(self):
        """Test handling of empty macro data."""
        empty_df = pd.DataFrame()
        features = regime_features(empty_df)
        
        self.assertTrue(features.empty)


class TestLabelRegimes(unittest.TestCase):
    """Test regime labeling."""
    
    def setUp(self):
        """Create test features."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        
        self.features = pd.DataFrame({
            "dgs10_6m_chg": np.random.normal(0, 1, 300),
            "cpi_yoy": np.random.normal(0, 1, 300),
            "unrate_6m_chg": np.random.normal(0, 1, 300),
        }, index=dates)
    
    def test_labels_count(self):
        """Test that labels have <= k unique values."""
        labels = label_regimes(self.features, method="rule_based", k=4)
        
        self.assertEqual(len(labels), len(self.features))
        self.assertLessEqual(labels.nunique(), 4)
    
    def test_labels_types(self):
        """Test that labels are from expected regime types."""
        labels = label_regimes(self.features, method="rule_based")
        
        expected_regimes = {"Risk-on", "Disinflation", "Tightening", "Recessionary"}
        actual_regimes = set(labels.unique())
        
        self.assertTrue(actual_regimes.issubset(expected_regimes))
    
    def test_kmeans_method(self):
        """Test KMeans labeling."""
        try:
            labels = label_regimes(self.features, method="kmeans", k=4)
            
            self.assertEqual(len(labels), len(self.features))
            self.assertLessEqual(labels.nunique(), 4)
        except ImportError:
            # sklearn not available, skip
            self.skipTest("sklearn not available")
    
    def test_empty_features(self):
        """Test handling of empty features."""
        empty = pd.DataFrame()
        labels = label_regimes(empty)
        
        self.assertTrue(labels.empty)


class TestRegimePerformance(unittest.TestCase):
    """Test regime performance calculation."""
    
    def setUp(self):
        """Create test portfolio returns and regime labels."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        
        # Two portfolios
        self.returns_by_portfolio = {
            "Portfolio A": pd.Series(np.random.normal(0.0004, 0.01, 252), index=dates),
            "Portfolio B": pd.Series(np.random.normal(0.0003, 0.008, 252), index=dates),
        }
        
        # Regime labels (cycle through 4 regimes)
        regimes = ["Risk-on", "Tightening", "Disinflation", "Recessionary"] * 63
        self.regime_labels = pd.Series(regimes[:252], index=dates)
    
    def test_performance_structure(self):
        """Test that regime_performance returns expected structure."""
        perf = regime_performance(self.returns_by_portfolio, self.regime_labels)
        
        self.assertIsInstance(perf, pd.DataFrame)
        self.assertGreater(len(perf), 0)
        
        # Should have columns: CAGR, Sharpe, N
        expected_cols = {"CAGR", "Sharpe", "N"}
        actual_cols = set(perf.columns)
        
        self.assertTrue(expected_cols.issubset(actual_cols))
    
    def test_performance_index(self):
        """Test that performance has multi-index (Portfolio, Regime)."""
        perf = regime_performance(self.returns_by_portfolio, self.regime_labels)
        
        self.assertEqual(perf.index.nlevels, 2)
        self.assertIn("Portfolio", perf.index.names)
        self.assertIn("Regime", perf.index.names)
    
    def test_empty_returns(self):
        """Test handling of empty returns."""
        empty_returns = {}
        perf = regime_performance(empty_returns, self.regime_labels)
        
        self.assertTrue(perf.empty)


class TestCurrentRegime(unittest.TestCase):
    """Test current regime detection."""
    
    def test_current_regime_detection(self):
        """Test that current_regime returns a valid regime."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=300, freq="B")
        
        features = pd.DataFrame({
            "dgs10_6m_chg": np.random.normal(0, 1, 300),
            "cpi_yoy": np.random.normal(0, 1, 300),
            "unrate_6m_chg": np.random.normal(0, 1, 300),
        }, index=dates)
        
        labels = label_regimes(features)
        
        regime = current_regime(features=features, labels=labels, lookback_days=30)
        
        # Should return one of the expected regimes
        expected_regimes = {"Risk-on", "Disinflation", "Tightening", "Recessionary", "Unknown"}
        self.assertIn(regime, expected_regimes)
    
    def test_current_regime_empty(self):
        """Test current_regime with empty data."""
        regime = current_regime(
            features=pd.DataFrame(),
            labels=pd.Series([], dtype=str)
        )
        
        self.assertEqual(regime, "Unknown")


if __name__ == "__main__":
    unittest.main()

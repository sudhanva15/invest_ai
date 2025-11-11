#!/usr/bin/env python3
"""Unit tests for core/utils/metrics.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import unittest
import pandas as pd
import numpy as np
from core.utils.metrics import annualized_metrics, beta_to_benchmark, value_at_risk, calmar_ratio


class TestAnnualizedMetrics(unittest.TestCase):
    """Test standardized annualized metrics calculations."""
    
    def setUp(self):
        """Create deterministic test data."""
        np.random.seed(42)
        
        # Create 252 trading days of returns (1 year)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        
        # Portfolio with 10% annual return, 15% volatility
        # Daily return = 0.10 / 252 ≈ 0.0004, daily vol = 0.15 / sqrt(252) ≈ 0.0094
        daily_mean = 0.10 / 252
        daily_vol = 0.15 / np.sqrt(252)
        
        self.returns = pd.Series(
            np.random.normal(daily_mean, daily_vol, 252),
            index=dates
        )
        
        # Create a benchmark (e.g., 8% return, 12% vol)
        bench_mean = 0.08 / 252
        bench_vol = 0.12 / np.sqrt(252)
        self.benchmark = pd.Series(
            np.random.normal(bench_mean, bench_vol, 252),
            index=dates
        )
    
    def test_annualized_metrics_structure(self):
        """Test that annualized_metrics returns expected keys."""
        metrics = annualized_metrics(self.returns)
        
        expected_keys = {"CAGR", "Volatility", "Sharpe", "MaxDD", "N", "Start", "End"}
        self.assertEqual(set(metrics.keys()), expected_keys)
    
    def test_cagr_calculation(self):
        """Test CAGR calculation with known input."""
        # Simple 1-year return of 10%
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series([0.10 / 252] * 252, index=dates)
        
        metrics = annualized_metrics(returns)
        
        # CAGR should be approximately 10%
        self.assertAlmostEqual(metrics["CAGR"], 0.10, delta=0.01)
    
    def test_volatility_scaling(self):
        """Test that volatility is properly annualized."""
        # Returns with known daily std dev
        daily_std = 0.01  # 1% daily
        returns = pd.Series(np.random.normal(0, daily_std, 252))
        
        metrics = annualized_metrics(returns)
        
        # Annualized vol = daily_std * sqrt(252) ≈ 0.16
        expected_vol = daily_std * np.sqrt(252)
        self.assertAlmostEqual(metrics["Volatility"], expected_vol, delta=0.03)
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        metrics = annualized_metrics(self.returns, risk_free_rate=0.02)
        
        # Sharpe = (CAGR - rf) / Volatility
        expected_sharpe = (metrics["CAGR"] - 0.02) / metrics["Volatility"]
        self.assertAlmostEqual(metrics["Sharpe"], expected_sharpe, delta=0.01)
    
    def test_max_drawdown(self):
        """Test max drawdown calculation."""
        # Create returns with known drawdown
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        returns = pd.Series([0.01, 0.02, -0.05, -0.03, 0.01, 0.02, -0.02, 0.01, 0.01, 0.01], index=dates)
        
        metrics = annualized_metrics(returns)
        
        # MaxDD should be negative
        self.assertLess(metrics["MaxDD"], 0)
        self.assertGreater(metrics["MaxDD"], -1)  # Should not be < -100%
    
    def test_empty_returns(self):
        """Test handling of empty returns."""
        empty = pd.Series([], dtype=float)
        metrics = annualized_metrics(empty)
        
        self.assertTrue(np.isnan(metrics["CAGR"]))
        self.assertEqual(metrics["N"], 0)
    
    def test_portfolio_with_weights(self):
        """Test metrics calculation with DataFrame and weights."""
        # Create multi-asset returns
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns_df = pd.DataFrame({
            "A": np.random.normal(0.10/252, 0.15/np.sqrt(252), 252),
            "B": np.random.normal(0.08/252, 0.12/np.sqrt(252), 252),
        }, index=dates)
        
        weights = {"A": 0.6, "B": 0.4}
        
        metrics = annualized_metrics(returns_df, weights=weights)
        
        # Should compute portfolio metrics
        self.assertIsInstance(metrics["CAGR"], float)
        self.assertFalse(np.isnan(metrics["CAGR"]))


class TestBeta(unittest.TestCase):
    """Test beta calculation."""
    
    def test_beta_calculation(self):
        """Test beta relative to benchmark."""
        np.random.seed(42)
        
        # Create correlated returns
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        bench = pd.Series(np.random.normal(0, 0.01, 252), index=dates)
        port = 1.2 * bench + pd.Series(np.random.normal(0, 0.005, 252), index=dates)
        
        beta = beta_to_benchmark(port, bench)
        
        # Beta should be close to 1.2
        self.assertAlmostEqual(beta, 1.2, delta=0.2)
    
    def test_beta_insufficient_data(self):
        """Test beta with insufficient overlap."""
        port = pd.Series([0.01, 0.02], index=[0, 1])
        bench = pd.Series([0.01, 0.015], index=[0, 1])
        
        beta = beta_to_benchmark(port, bench, min_overlap=252)
        
        self.assertTrue(np.isnan(beta))


class TestVaR(unittest.TestCase):
    """Test Value at Risk calculation."""
    
    def test_var_historical(self):
        """Test historical VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        
        var_95 = value_at_risk(returns, confidence=0.95, method="historical")
        
        # 95% VaR should be negative (5th percentile)
        self.assertLess(var_95, 0)
    
    def test_var_parametric(self):
        """Test parametric VaR."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0, 0.01, 1000))
        
        var_95 = value_at_risk(returns, confidence=0.95, method="parametric")
        
        # Should be negative
        self.assertLess(var_95, 0)


class TestCalmar(unittest.TestCase):
    """Test Calmar ratio calculation."""
    
    def test_calmar_ratio(self):
        """Test Calmar ratio = CAGR / abs(MaxDD)."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=252, freq="B")
        returns = pd.Series(np.random.normal(0.10/252, 0.15/np.sqrt(252), 252), index=dates)
        
        calmar = calmar_ratio(returns)
        
        # Should be positive for positive CAGR
        self.assertGreater(calmar, 0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for objective constraints and class caps enforcement."""

import unittest
import sys
from pathlib import Path

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.recommendation_engine import _apply_objective_constraints, _class_of_symbol


class TestConstraints(unittest.TestCase):
    """Test constraint enforcement logic."""
    
    def setUp(self):
        """Set up test catalog."""
        self.catalog = {
            "assets": [
                {"symbol": "SPY", "class": "equity_us"},
                {"symbol": "TLT", "class": "bonds_tsy"},
                {"symbol": "GLD", "class": "commodities"},
                {"symbol": "DBC", "class": "commodities"},
                {"symbol": "BND", "class": "bonds_ig"},
            ],
            "caps": {
                "asset_class": {
                    "commodities": 0.20,
                    "equity_us": 0.80,
                    "bonds_tsy": 0.80,
                    "bonds_ig": 0.80,
                }
            }
        }
    
    def test_class_mapping(self):
        """Test _class_of_symbol maps correctly."""
        self.assertEqual(_class_of_symbol("SPY", self.catalog), "equity_us")
        self.assertEqual(_class_of_symbol("GLD", self.catalog), "commodities")
        self.assertEqual(_class_of_symbol("TLT", self.catalog), "bonds_tsy")
        
        # Test heuristic fallback
        empty_catalog = {"assets": []}
        self.assertEqual(_class_of_symbol("DBC", empty_catalog), "commodities")
        self.assertEqual(_class_of_symbol("VTI", empty_catalog), "equity_us")
    
    def test_commodity_cap_enforcement(self):
        """Test that commodity cap (20%) is enforced."""
        symbols = ["SPY", "GLD", "DBC", "TLT"]
        
        # Start with weights violating commodity cap
        weights = {
            "SPY": 0.40,
            "GLD": 0.20,  # commodities
            "DBC": 0.15,  # commodities -> total 35%
            "TLT": 0.25,
        }
        
        # Apply constraints
        result = _apply_objective_constraints(
            weights,
            symbols=symbols,
            catalog=self.catalog,
            core_min=0.60,
            sat_max_total=0.40,
            sat_max_single=0.10
        )
        
        # Check commodity total is capped at 20%
        commodity_total = result.get("GLD", 0) + result.get("DBC", 0)
        self.assertLessEqual(commodity_total, 0.21, 
                            f"Commodity weight {commodity_total:.2%} exceeds 20% cap")
        
        # Check weights still sum to ~1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=2,
                              msg=f"Weights sum to {total:.3f}, expected 1.0")
    
    def test_empty_weights_fallback(self):
        """Test fallback to equal weight when constraints produce zero weights."""
        symbols = ["SPY", "TLT", "GLD"]
        
        # All weights zero
        weights = {"SPY": 0.0, "TLT": 0.0, "GLD": 0.0}
        
        result = _apply_objective_constraints(
            weights,
            symbols=symbols,
            catalog=self.catalog,
            core_min=0.70,
            sat_max_total=0.30,
            sat_max_single=0.10
        )
        
        # Should fallback to equal weight
        self.assertGreater(len(result), 0, "Should have non-empty result")
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=2)
    
    def test_small_universe_constraints(self):
        """Test constraints with very few symbols."""
        symbols = ["SPY", "TLT"]
        weights = {"SPY": 0.6, "TLT": 0.4}
        
        result = _apply_objective_constraints(
            weights,
            symbols=symbols,
            catalog=self.catalog,
            core_min=0.50,
            sat_max_total=0.50,
            sat_max_single=0.30
        )
        
        # Should preserve structure
        self.assertEqual(len(result), 2)
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=2)
    
    def test_multiple_class_caps(self):
        """Test enforcement of multiple asset class caps simultaneously."""
        symbols = ["SPY", "TLT", "GLD", "DBC", "BND"]
        
        # Weights with both commodities and bonds potentially over-weighted
        weights = {
            "SPY": 0.30,
            "TLT": 0.20,
            "BND": 0.20,
            "GLD": 0.15,
            "DBC": 0.15,  # commodities total = 30%
        }
        
        result = _apply_objective_constraints(
            weights,
            symbols=symbols,
            catalog=self.catalog,
            core_min=0.60,
            sat_max_total=0.40,
            sat_max_single=0.10
        )
        
        # Check commodity cap
        commodity_total = result.get("GLD", 0) + result.get("DBC", 0)
        self.assertLessEqual(commodity_total, 0.21,
                            f"Commodity weight {commodity_total:.2%} exceeds 20% cap")
        
        # Weights should sum to 1.0
        total = sum(result.values())
        self.assertAlmostEqual(total, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()

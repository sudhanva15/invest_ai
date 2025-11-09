"""Test suite for portfolio weight constraints."""

import unittest
from core.portfolio.constraints import apply_weight_constraints, validate_weights


class TestPortfolioConstraints(unittest.TestCase):
    """Test cases for portfolio weight constraints module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Configure standard test case
        self.core = ["VTI", "BND"]
        self.sats = ["AAPL", "GOOGL"]
        
        # Sample weights meeting all constraints
        self.core_only = {
            "VTI": 0.60,  # Core can be large
            "BND": 0.40   # No single position cap
        }
        
        self.with_sats = {
            "VTI": 0.45,  # Core = 75%
            "BND": 0.30,
            "AAPL": 0.15,  # Sats = 25%
            "GOOGL": 0.10
        }
    
    def test_single_position_cap(self):
        """Test capping of individual satellite positions."""
        # Arrange: Set up violating weights
        w = {
            "VTI": 0.35,    # Core can exceed cap
            "BND": 0.30,
            "AAPL": 0.20,   # Satellite should be capped
            "GOOGL": 0.15   # Satellite should be capped
        }
        
        # Act: Apply constraints
        adjusted = apply_weight_constraints(
            weights=w,
            core_symbols=self.core,
            satellite_symbols=self.sats,
            single_max=0.07  # Strict cap for satellites
        )
        
        # Assert: Only satellites are capped
        for symbol in self.sats:
            self.assertLessEqual(
                adjusted[symbol],
                0.07,
                f"Satellite position {symbol} exceeds cap"
            )
            
        # Core positions can be large
        for symbol in self.core:
            self.assertGreaterEqual(
                adjusted[symbol],
                0.07,
                f"Core position {symbol} should not be capped"
            )
        
    def test_satellite_allocation_cap(self):
        """Test satellite allocation maximum."""
        # Arrange: Set up high satellite allocation
        w = {
            "VTI": 0.30,
            "BND": 0.20,
            "AAPL": 0.30,  # Total satellite = 50%
            "GOOGL": 0.20
        }
        
        # Act: Apply constraints
        adjusted = apply_weight_constraints(
            weights=w,
            core_symbols=self.core,
            satellite_symbols=self.sats
        )
        
        # Assert: Satellite allocation within limit
        sat_sum = sum(adjusted[s] for s in self.sats)
        self.assertLessEqual(
            sat_sum,
            0.35,
            f"Satellite allocation {sat_sum:.1%} exceeds 35% cap"
        )
        
    def test_core_allocation_minimum(self):
        """Test core allocation minimum."""
        # Arrange: Set up low core allocation
        w = {
            "VTI": 0.30,
            "BND": 0.20,  # Total core = 50%
            "AAPL": 0.30,
            "GOOGL": 0.20
        }
        
        # Act: Apply constraints
        adjusted = apply_weight_constraints(
            weights=w,
            core_symbols=self.core,
            satellite_symbols=self.sats
        )
        
        # Assert: Core allocation meets minimum
        core_sum = sum(adjusted[s] for s in self.core)
        self.assertGreaterEqual(
            core_sum,
            0.65,
            f"Core allocation {core_sum:.1%} below 65% minimum"
        )
        
    def test_weight_normalization(self):
        """Test that final weights sum to 1.0."""
        # Test with both valid and invalid initial weights
        test_cases = [
            self.core_only,   # Core only
            self.with_sats,   # With satellites
            {k: v*2 for k, v in self.with_sats.items()},  # Sum > 1
            {k: v/2 for k, v in self.with_sats.items()}   # Sum < 1
        ]
        
        for weights in test_cases:
            satellites = self.sats if any(s in weights for s in self.sats) else None
            
            adjusted = apply_weight_constraints(
                weights=weights,
                core_symbols=self.core,
                satellite_symbols=satellites
            )
            
            total = sum(adjusted.values())
            self.assertAlmostEqual(
                total,
                1.0,
                places=6,
                msg=f"Weights sum to {total:.6f}, expected 1.0"
            )
        
    def test_validation(self):
        """Test the weight validation function."""
        # Core-only portfolio should validate
        violations = validate_weights(
            weights=self.core_only,
            core_symbols=self.core
        )
        self.assertEqual(
            len(violations), 0,
            f"Core-only portfolio had violations: {violations}"
        )
        
        # Portfolio with satellites should validate
        violations = validate_weights(
            weights=self.with_sats,
            core_symbols=self.core,
            satellite_symbols=self.sats
        )
        self.assertEqual(
            len(violations), 0,
            f"Valid satellite portfolio had violations: {violations}"
        )
        
        # High satellite allocation should fail
        invalid = {
            "VTI": 0.35,
            "BND": 0.25,
            "AAPL": 0.25,  # Satellites total 40% > 35%
            "GOOGL": 0.15
        }
        violations = validate_weights(
            weights=invalid,
            core_symbols=self.core,
            satellite_symbols=self.sats
        )
        self.assertGreater(
            len(violations), 0,
            "High satellite allocation should have violations"
        )
        
    def test_core_only_portfolio(self):
        """Test constraints with core-only portfolio."""
        # Act: Apply constraints to core-only portfolio
        adjusted = apply_weight_constraints(
            weights=self.core_only,
            core_symbols=self.core
        )
        
        # Assert 1: Weights sum to 1.0
        self.assertAlmostEqual(
            sum(adjusted.values()),
            1.0,
            places=6,
            msg="Core-only weights should sum to 1.0"
        )
        
        # Assert 2: No violations in validation
        violations = validate_weights(
            weights=self.core_only,
            core_symbols=self.core
        )
        self.assertEqual(
            len(violations), 0,
            f"Core-only portfolio had violations: {violations}"
        )
        
        # Assert 3: Core weights can be large
        self.assertGreater(
            adjusted["VTI"], 
            0.07,
            "Core positions should not be capped"
        )
        self.assertGreater(
            adjusted["BND"],
            0.07,
            "Core positions should not be capped"
        )


if __name__ == "__main__":
    unittest.main()
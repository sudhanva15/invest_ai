"""Unit tests for candidate distinctness and diversity metrics."""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestDistinctness(unittest.TestCase):
    """Test distinctness metrics for candidate portfolios."""
    
    def cosine_similarity(self, a: pd.Series, b: pd.Series) -> float:
        """Compute cosine similarity between two weight vectors."""
        aa = a.values
        bb = b.values
        den = float((np.linalg.norm(aa) * np.linalg.norm(bb)) or 0.0)
        if den == 0:
            return 0.0
        return float(np.dot(aa, bb) / den)
    
    def count_distinct_candidates(self, candidates: list, threshold: float = 0.995) -> int:
        """Count distinct candidates using cosine similarity."""
        if not candidates:
            return 0
        
        # Extract weight vectors
        symbols = set()
        for c in candidates:
            symbols.update(c.get("weights", {}).keys())
        symbols = sorted(symbols)
        
        weight_vectors = []
        for c in candidates:
            w = pd.Series(c.get("weights", {})).reindex(symbols).fillna(0.0)
            weight_vectors.append(w)
        
        # Count distinct using similarity threshold
        distinct = []
        for w in weight_vectors:
            if not distinct:
                distinct.append(w)
                continue
            
            sims = [self.cosine_similarity(w, d) for d in distinct]
            if max(sims) < threshold:
                distinct.append(w)
        
        return len(distinct)
    
    def test_identical_candidates(self):
        """Test that identical candidates are detected."""
        candidates = [
            {"name": "A", "weights": {"SPY": 0.5, "TLT": 0.5}},
            {"name": "B", "weights": {"SPY": 0.5, "TLT": 0.5}},
            {"name": "C", "weights": {"SPY": 0.5, "TLT": 0.5}},
        ]
        
        distinct = self.count_distinct_candidates(candidates)
        self.assertEqual(distinct, 1, "Identical candidates should count as 1 distinct")
    
    def test_clearly_distinct_candidates(self):
        """Test that clearly different candidates are counted as distinct."""
        candidates = [
            {"name": "Equity-Heavy", "weights": {"SPY": 0.8, "TLT": 0.2}},
            {"name": "Bond-Heavy", "weights": {"SPY": 0.2, "TLT": 0.8}},
            {"name": "Balanced", "weights": {"SPY": 0.5, "TLT": 0.5}},
        ]
        
        distinct = self.count_distinct_candidates(candidates)
        self.assertEqual(distinct, 3, "Clearly distinct candidates should all count")
    
    def test_near_identical_threshold(self):
        """Test that near-identical candidates (>99.5% similar) are grouped."""
        candidates = [
            {"name": "A", "weights": {"SPY": 0.500, "TLT": 0.500}},
            {"name": "B", "weights": {"SPY": 0.501, "TLT": 0.499}},  # Very similar
            {"name": "C", "weights": {"SPY": 0.502, "TLT": 0.498}},  # Very similar
        ]
        
        distinct = self.count_distinct_candidates(candidates, threshold=0.995)
        # These should be detected as nearly identical
        self.assertLessEqual(distinct, 2, 
                            "Near-identical candidates should be grouped")
    
    def test_multiasset_distinctness(self):
        """Test distinctness with more assets."""
        candidates = [
            {
                "name": "HRP",
                "weights": {"SPY": 0.40, "TLT": 0.30, "GLD": 0.20, "BND": 0.10}
            },
            {
                "name": "Max Sharpe",
                "weights": {"SPY": 0.60, "TLT": 0.20, "GLD": 0.10, "BND": 0.10}
            },
            {
                "name": "Min Var",
                "weights": {"SPY": 0.20, "TLT": 0.40, "GLD": 0.10, "BND": 0.30}
            },
            {
                "name": "Equal Weight",
                "weights": {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BND": 0.25}
            },
        ]
        
        distinct = self.count_distinct_candidates(candidates, threshold=0.995)
        self.assertGreaterEqual(distinct, 3, 
                               "Different optimization methods should produce distinct portfolios")
    
    def test_small_perturbations(self):
        """Test that small jitter creates measurably distinct candidates."""
        np.random.seed(42)
        
        base_weights = {"SPY": 0.4, "TLT": 0.3, "GLD": 0.2, "BND": 0.1}
        candidates = [{"name": "Base", "weights": base_weights}]
        
        # Add larger random perturbations (5% std to exceed 0.995 threshold)
        for i in range(5):
            perturbed = {}
            for sym, w in base_weights.items():
                perturbed[sym] = max(0, w + np.random.normal(0, 0.05))
            # Renormalize
            total = sum(perturbed.values())
            perturbed = {k: v/total for k, v in perturbed.items()}
            candidates.append({"name": f"Perturbed_{i}", "weights": perturbed})
        
        distinct = self.count_distinct_candidates(candidates, threshold=0.995)
        # With 5% jitter, should get multiple distinct candidates
        self.assertGreater(distinct, 1, 
                          "Perturbations should create some distinctness")
    
    def test_empty_candidates(self):
        """Test handling of empty candidate list."""
        distinct = self.count_distinct_candidates([])
        self.assertEqual(distinct, 0, "Empty list should return 0 distinct")
    
    def test_single_candidate(self):
        """Test single candidate case."""
        candidates = [{"name": "Only", "weights": {"SPY": 1.0}}]
        distinct = self.count_distinct_candidates(candidates)
        self.assertEqual(distinct, 1, "Single candidate should count as 1 distinct")


class TestDiversificationMetrics(unittest.TestCase):
    """Test portfolio diversification metrics."""
    
    def test_concentration_penalty(self):
        """Test that concentrated portfolios are penalized."""
        # Concentrated portfolio
        concentrated = {"SPY": 0.50, "TLT": 0.30, "GLD": 0.20}
        max_weight_conc = max(concentrated.values())
        penalty_conc = max(0.0, max_weight_conc - 0.20)
        
        # Diversified portfolio
        diversified = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "BND": 0.25}
        max_weight_div = max(diversified.values())
        penalty_div = max(0.0, max_weight_div - 0.20)
        
        self.assertGreater(penalty_conc, penalty_div,
                          "Concentrated portfolio should have higher penalty")
    
    def test_equal_weight_diversity(self):
        """Test that equal-weight has zero concentration penalty."""
        n = 5
        weights = {f"ASSET_{i}": 1.0/n for i in range(n)}
        max_weight = max(weights.values())
        penalty = max(0.0, max_weight - 0.20)
        
        self.assertEqual(penalty, 0.0,
                        "Equal-weight portfolio should have zero concentration penalty")


if __name__ == "__main__":
    unittest.main()

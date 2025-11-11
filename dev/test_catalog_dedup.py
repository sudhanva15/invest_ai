import sys
from pathlib import Path
import unittest
import pandas as pd

# Add repo root to path for imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.streamlit_app import _ensure_unique_symbol_index

class TestCatalogDedup(unittest.TestCase):
    def test_dedup_and_reindex(self):
        df = pd.DataFrame([
            {"symbol": "SPY", "name": "S&P 500 ETF", "min_tier": "retail", "min_risk_pct": 0},
            {"symbol": "QQQ", "name": "Nasdaq 100 ETF", "min_tier": "retail", "min_risk_pct": 0},
            {"symbol": "SPY", "name": "S&P 500 ETF (dup)", "min_tier": "retail", "min_risk_pct": 0},
        ])
        dedup = _ensure_unique_symbol_index(df)
        self.assertTrue("SPY" in dedup.index and "QQQ" in dedup.index)
        self.assertEqual(len(dedup.index), 2)
        w_index = pd.Index(["SPY", "QQQ"], dtype=str)
        reindexed = dedup.reindex(w_index)
        self.assertEqual(list(reindexed.index), ["SPY", "QQQ"])

if __name__ == "__main__":
    unittest.main(verbosity=2)

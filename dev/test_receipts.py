import sys
from pathlib import Path
import unittest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

class TestReceipts(unittest.TestCase):
    def test_receipts_present(self):
        try:
            from core.utils.receipts import build_receipts  # optional util
            util_available = True
        except Exception:
            util_available = False

        from core.data_ingestion import get_prices_with_provenance
        tickers = ["SPY","QQQ","TLT"]
        df, _ = get_prices_with_provenance(tickers, start="2010-01-01")

        if util_available:
            rec = build_receipts(tickers, df)
            self.assertEqual(len(rec), len(tickers))
            keys = set(rec[0].keys())
            self.assertTrue({"ticker","provider","backfill_pct","first","last","nan_rate"} <= keys)
        else:
            # Fallback: check metadata attrs exist so UI can build receipts inline
            self.assertTrue(hasattr(df, "_provider_map"))
            self.assertTrue(hasattr(df, "_backfill_pct"))
            self.assertTrue(hasattr(df, "_coverage"))

if __name__ == "__main__":
    unittest.main()
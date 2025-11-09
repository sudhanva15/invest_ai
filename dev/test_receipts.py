import unittest

class TestReceipts(unittest.TestCase):
    def test_receipts_present(self):
        try:
            from core.utils.receipts import build_receipts  # optional util
            util_available = True
        except Exception:
            util_available = False

        from core.data_sources.router_smart import get_prices_with_provenance
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
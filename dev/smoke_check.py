#!/usr/bin/env python
from core.data_ingestion import get_prices_with_provenance
from core.data_sources.fred import load_series

def main():
    # Test price data with provenance
    tickers = ["SPY", "QQQ", "TLT", "IEF", "GLD"]
    prices_df, prov = get_prices_with_provenance(tickers)
    
    print(f"\nPrice DataFrame shape: {prices_df.shape}")
    print("\nProvider Map:", prices_df.attrs.get("provider_map", {}))
    print("Backfill %:", prices_df.attrs.get("backfill_pct", {}))
    print("Coverage:", prices_df.attrs.get("coverage", {}))
    
    # Basic validation
    assert len(prices_df) > 0, "No price data loaded"
    assert prices_df.attrs.get("provider_map"), "Missing provider map"
    assert prices_df.attrs.get("backfill_pct"), "Missing backfill percentages"
    assert prices_df.attrs.get("coverage"), "Missing coverage data"
    
    # Test FRED data
    fred_series = ["DGS10", "T10Y2Y", "CPIAUCSL"]
    for series_id in fred_series:
        s = load_series(series_id)
        print(f"\n{series_id} rows:", len(s))
        assert len(s) > 0, f"No data for {series_id}"

if __name__ == "__main__":
    main()
    print("\nAll checks passed!")
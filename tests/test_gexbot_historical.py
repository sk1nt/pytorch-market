from market_ml.gexbot import load_historical_gex, get_available_tickers, GexbotError
from datetime import datetime, timedelta
import os
import pytest

def test_gexbot_tickers():
    """Test that we can fetch the list of available tickers from gexbot.com API."""
    tickers = get_available_tickers()
    assert 'stocks' in tickers, "Response should contain 'stocks' key"
    assert 'indexes' in tickers, "Response should contain 'indexes' key"
    assert 'futures' in tickers, "Response should contain 'futures' key"
    
    print("Available tickers:")
    print(f"  Stocks: {tickers['stocks']}")
    print(f"  Indexes: {tickers['indexes']}")
    print(f"  Futures: {tickers['futures']}")
    
    # Verify expected symbols
    assert 'SPX' in tickers['indexes'], "SPX should be available"
    assert 'NDX' in tickers['indexes'], "NDX should be available"
    assert 'SPY' in tickers['stocks'], "SPY should be available"
    assert 'NQ_NDX' in tickers['futures'], "NQ_NDX should be available as a future"

def test_gexbot_historical():
    """Test loading historical GEX data using the classic endpoint."""
    api_key = os.environ.get("GEXBOT_API_KEY", "VmAk9O0APpXw")
    
    # Use a recent date range for SPX
    end = datetime.now().date()
    start = end - timedelta(days=7)
    
    # Test with SPX using 1d aggregation
    try:
        df = load_historical_gex("SPX", start.isoformat(), end.isoformat(), api_key=api_key, aggregation="1d")
    except GexbotError as e:
        # Skip test gracefully if API key is invalid/missing
        msg = str(e).lower()
        if "unauthorized" in msg or "401" in msg or "api request failed" in msg:
            pytest.skip("Gexbot API key invalid or missing; skipping historical test.")
        raise
    
    assert not df.empty, "No GEX data returned for SPX"
    print(f"\nSPX GEX Data (last 7 days, 1d aggregation):")
    print(f"Columns: {list(df.columns)}")
    print(f"Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    test_gexbot_tickers()
    test_gexbot_historical()

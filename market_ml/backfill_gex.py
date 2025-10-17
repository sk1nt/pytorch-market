"""
Backfill GEX historical data by polling classic/zero endpoint over time.

Since classic/1d is not available on the current API tier, this script
polls the classic/zero endpoint at intervals to build a historical time-series.

Usage:
    # Quick backfill (every 30 seconds for testing)
    python -m market_ml.backfill_gex --samples 10 --interval 30
    
    # Long-term backfill (every 5 minutes for 2 months)
    python -m market_ml.backfill_gex --samples 17280 --interval 300
"""

import argparse
import time
from datetime import datetime
from market_ml.gexbot import load_historical_gex, probe_gexbot_key
from market_ml.config import Config
import pandas as pd
import os

def backfill_gex_data(symbols, api_key, num_samples, interval_seconds, output_dir):
    """
    Poll GEXBOT classic/zero endpoint at intervals to build historical data.
    
    Args:
        symbols: List of symbols to track
        api_key: GEXBOT API key
        num_samples: Number of snapshots to collect
        interval_seconds: Seconds between each poll
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize storage
    historical_data = {symbol: [] for symbol in symbols}
    
    print(f"Starting GEX backfill for {len(symbols)} symbols")
    print(f"Will collect {num_samples} samples at {interval_seconds}s intervals")
    print(f"Estimated duration: {(num_samples * interval_seconds) / 3600:.1f} hours")
    print(f"Output: {output_dir}")
    print("\nPress Ctrl+C to stop and save current data\n")
    
    try:
        for i in range(num_samples):
            timestamp = datetime.now()
            print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Sample {i+1}/{num_samples}")
            
            for symbol in symbols:
                try:
                    df = load_historical_gex(
                        symbol=symbol,
                        start_date='2000-01-01',  # Use wide date range to avoid filtering out current data
                        end_date='2099-12-31',
                        aggregation='zero',
                        api_key=api_key
                    )
                    if not df.empty:
                        historical_data[symbol].append(df)
                        price = df.iloc[0].get('price', 'N/A')
                        zero_gamma = df.iloc[0].get('zero_gamma', 'N/A')
                        print(f"  {symbol}: price={price}, zero_gamma={zero_gamma}")
                    else:
                        print(f"  {symbol}: No data returned")
                except Exception as e:
                    print(f"  {symbol}: Error - {e}")
            
            # Save after each round
            if (i + 1) % 10 == 0:  # Save every 10 samples
                print(f"\nSaving checkpoint at sample {i+1}...")
                save_data(historical_data, output_dir)
            
            # Wait for next interval (unless it's the last sample)
            if i < num_samples - 1:
                time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("\n\nBackfill interrupted by user")
    
    # Final save
    print("\nSaving final data...")
    save_data(historical_data, output_dir)
    print("Backfill complete!")

def save_data(historical_data, output_dir):
    """Save collected data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, dfs in historical_data.items():
        if dfs:
            df_all = pd.concat(dfs, ignore_index=False)
            # Remove duplicates based on index (timestamp)
            df_all = df_all[~df_all.index.duplicated(keep='last')]
            df_all = df_all.sort_index()
            
            output_path = os.path.join(output_dir, f"{symbol}_gex_historical.csv")
            df_all.to_csv(output_path)
            print(f"  Saved {symbol}: {len(df_all)} snapshots to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill GEX historical data')
    parser.add_argument('--symbols', nargs='+', 
                       default=['VIX', 'ES_SPX', 'NQ_NDX', 'SPY', 'QQQ', 'AAPL'],
                       help='Symbols to track')
    parser.add_argument('--samples', type=int, default=100,
                       help='Number of snapshots to collect')
    parser.add_argument('--interval', type=int, default=300,
                       help='Seconds between polls (default: 300 = 5 min)')
    parser.add_argument('--output', default='outputs/backfill',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load config and validate API key
    config = Config()
    probe = probe_gexbot_key(config.GEXBOT_API_KEY)
    if not probe.get("valid"):
        print(f"GEXBOT API key check failed: {probe.get('detail')}")
        raise SystemExit(1)
    
    print(f"API key validated: {probe.get('detail')}\n")
    
    backfill_gex_data(
        symbols=args.symbols,
        api_key=config.GEXBOT_API_KEY,
        num_samples=args.samples,
        interval_seconds=args.interval,
        output_dir=args.output
    )

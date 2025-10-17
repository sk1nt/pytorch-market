"""
Backfill GEX data during market hours only (8:30 AM - 6:00 PM EST).

Polls GEXBOT classic endpoint every minute during trading hours to build
high-resolution intraday time-series data.

Usage:
    # Run for 5 trading days
    python -m market_ml.backfill_market_hours --days 5
    
    # Run indefinitely during market hours
    python -m market_ml.backfill_market_hours --days 0
    
    # Use different aggregation (zero, one, full)
    python -m market_ml.backfill_market_hours --days 10 --aggregation full
"""

import argparse
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from market_ml.gexbot import load_historical_gex, probe_gexbot_key
from market_ml.config import Config
import pandas as pd
import os

# Market hours in EST
MARKET_OPEN = dt_time(8, 30)  # 8:30 AM
MARKET_CLOSE = dt_time(18, 0)  # 6:00 PM
EST = ZoneInfo('America/New_York')
POLL_INTERVAL = 60  # 1 minute

def is_market_hours():
    """Check if current time is during market hours (8:30 AM - 6:00 PM EST)."""
    now_est = datetime.now(EST)
    current_time = now_est.time()
    
    # Check if it's a weekday (Monday=0, Sunday=6)
    is_weekday = now_est.weekday() < 5
    
    # Check if time is between market open and close
    is_trading_time = MARKET_OPEN <= current_time <= MARKET_CLOSE
    
    return is_weekday and is_trading_time

def seconds_until_market_open():
    """Calculate seconds until next market open."""
    now_est = datetime.now(EST)
    current_time = now_est.time()
    current_weekday = now_est.weekday()
    
    # If it's a weekday and before market open today
    if current_weekday < 5 and current_time < MARKET_OPEN:
        target = now_est.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
        return int((target - now_est).total_seconds())
    
    # Calculate next Monday at market open
    days_until_monday = (7 - current_weekday) if current_weekday >= 5 else (1 if current_weekday == 4 and current_time >= MARKET_CLOSE else 1)
    if current_weekday < 5 and current_time < MARKET_CLOSE:
        days_until_monday = 0
    elif current_weekday == 4:  # Friday
        days_until_monday = 3
    elif current_weekday == 5:  # Saturday
        days_until_monday = 2
    elif current_weekday == 6:  # Sunday
        days_until_monday = 1
    
    target = now_est.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0, microsecond=0)
    target = target.replace(day=now_est.day + days_until_monday)
    
    return int((target - now_est).total_seconds())

def backfill_market_hours(symbols, api_key, num_days, aggregation, output_dir):
    """
    Poll GEXBOT during market hours to build intraday historical data.
    
    Args:
        symbols: List of symbols to track
        api_key: GEXBOT API key
        num_days: Number of trading days to collect (0 = run indefinitely)
        aggregation: Data aggregation type ('zero', 'one', 'full')
        output_dir: Directory to save data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize storage
    historical_data = {symbol: [] for symbol in symbols}
    
    # Calculate expected samples per day
    market_duration_minutes = (MARKET_CLOSE.hour * 60 + MARKET_CLOSE.minute) - \
                             (MARKET_OPEN.hour * 60 + MARKET_OPEN.minute)
    samples_per_day = market_duration_minutes  # 1 sample per minute
    total_samples = samples_per_day * num_days if num_days > 0 else float('inf')
    
    print(f"Starting market-hours GEX backfill for {len(symbols)} symbols")
    print(f"Market hours: 8:30 AM - 6:00 PM EST (Monday-Friday)")
    print(f"Polling interval: {POLL_INTERVAL}s (every minute)")
    print(f"Aggregation: {aggregation}")
    print(f"Samples per day: ~{samples_per_day}")
    if num_days > 0:
        print(f"Target: {num_days} trading days (~{int(total_samples)} samples)")
    else:
        print(f"Running indefinitely during market hours")
    print(f"Output: {output_dir}")
    print("\nPress Ctrl+C to stop and save current data\n")
    
    sample_count = 0
    
    try:
        while sample_count < total_samples:
            # Check if we're in market hours
            if not is_market_hours():
                now_est = datetime.now(EST)
                wait_seconds = seconds_until_market_open()
                wait_hours = wait_seconds / 3600
                
                next_open = now_est.replace(hour=MARKET_OPEN.hour, minute=MARKET_OPEN.minute, second=0)
                
                print(f"\n[{now_est.strftime('%Y-%m-%d %H:%M:%S %Z')}] Outside market hours")
                print(f"Waiting {wait_hours:.1f} hours until market opens...")
                print(f"Next market open: {next_open.strftime('%Y-%m-%d %H:%M:%S %Z')}")
                
                # Sleep in chunks so we can handle Ctrl+C
                for _ in range(wait_seconds // 60):
                    time.sleep(60)
                    if is_market_hours():
                        break
                
                continue
            
            # We're in market hours - collect data
            timestamp = datetime.now(EST)
            sample_count += 1
            
            print(f"[{timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}] Sample {sample_count}" + 
                  (f"/{int(total_samples)}" if num_days > 0 else ""))
            
            for symbol in symbols:
                try:
                    df = load_historical_gex(
                        symbol=symbol,
                        start_date='2000-01-01',  # Wide range to avoid filtering
                        end_date='2099-12-31',
                        aggregation=aggregation,
                        api_key=api_key
                    )
                    if not df.empty:
                        historical_data[symbol].append(df)
                        price = df.iloc[0].get('price', 'N/A')
                        zero_gamma = df.iloc[0].get('zero_gamma', 'N/A')
                        sum_gex = df.iloc[0].get('sum_gex_oi', 'N/A')
                        print(f"  {symbol}: price=${price}, zero_gamma=${zero_gamma}, sum_gex={sum_gex}")
                    else:
                        print(f"  {symbol}: No data returned")
                except Exception as e:
                    print(f"  {symbol}: Error - {e}")
            
            # Save checkpoint every 60 samples (every hour)
            if sample_count % 60 == 0:
                print(f"\n💾 Saving checkpoint at sample {sample_count}...")
                save_data(historical_data, output_dir, aggregation)
            
            # Wait until next minute
            if sample_count < total_samples:
                time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Backfill interrupted by user")
    
    # Final save
    print("\n💾 Saving final data...")
    save_data(historical_data, output_dir, aggregation)
    print(f"✅ Backfill complete! Collected {sample_count} samples")

def save_data(historical_data, output_dir, aggregation):
    """Save collected data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for symbol, dfs in historical_data.items():
        if dfs:
            df_all = pd.concat(dfs, ignore_index=False)
            # Remove duplicates based on index (timestamp)
            df_all = df_all[~df_all.index.duplicated(keep='last')]
            df_all = df_all.sort_index()
            
            filename = f"{symbol}_gex_{aggregation}_intraday.csv"
            output_path = os.path.join(output_dir, filename)
            df_all.to_csv(output_path)
            print(f"  ✅ Saved {symbol}: {len(df_all)} snapshots to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backfill GEX data during market hours (8:30 AM - 6:00 PM EST)'
    )
    parser.add_argument('--symbols', nargs='+',
                       default=['VIX', 'ES_SPX', 'NQ_NDX', 'SPY', 'QQQ', 'AAPL'],
                       help='Symbols to track')
    parser.add_argument('--days', type=int, default=5,
                       help='Number of trading days to collect (0 = run indefinitely)')
    parser.add_argument('--aggregation', default='zero',
                       choices=['zero', 'one', 'full'],
                       help='Aggregation type: zero (0DTE), one (0-1DTE), full (all DTE)')
    parser.add_argument('--output', default='outputs/intraday',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Load config and validate API key
    config = Config()
    probe = probe_gexbot_key(config.GEXBOT_API_KEY)
    if not probe.get("valid"):
        print(f"❌ GEXBOT API key check failed: {probe.get('detail')}")
        raise SystemExit(1)
    
    print(f"✅ API key validated: {probe.get('detail')}\n")
    
    backfill_market_hours(
        symbols=args.symbols,
        api_key=config.GEXBOT_API_KEY,
        num_days=args.days,
        aggregation=args.aggregation,
        output_dir=args.output
    )

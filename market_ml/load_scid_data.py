"""
Quick script to load and analyze MNQ and NQ SCID files from Sierra Chart.

Usage:
    python -m market_ml.load_scid_data /path/to/MNQ.scid
    python -m market_ml.load_scid_data /path/to/NQ.scid --tail 100
"""

import argparse
from market_ml.data import load_sierra_chart_scid
import pandas as pd

def analyze_scid_file(path, max_records=None, tail_records=None):
    """Load and display summary of SCID file."""
    print(f"Loading SCID file: {path}")
    print("=" * 70)
    
    # Load the data
    df = load_sierra_chart_scid(path, max_records=max_records)
    
    if df.empty:
        print("❌ No data loaded (file might be empty or corrupted)")
        return
    
    print(f"\n✅ Loaded {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Duration: {(df['timestamp'].max() - df['timestamp'].min())}")
    
    print("\n📊 Data Summary:")
    print("-" * 70)
    print(df.describe())
    
    print("\n📈 First 10 records:")
    print("-" * 70)
    print(df.head(10).to_string())
    
    if tail_records:
        print(f"\n📉 Last {tail_records} records:")
        print("-" * 70)
        print(df.tail(tail_records).to_string())
    
    print("\n💾 Columns available:")
    print("-" * 70)
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Additional statistics
    print("\n📊 Trading Statistics:")
    print("-" * 70)
    print(f"Total volume: {df['total_volume'].sum():,.0f}")
    print(f"Total trades: {df['num_trades'].sum():,.0f}")
    print(f"Avg volume per bar: {df['total_volume'].mean():,.0f}")
    print(f"Avg trades per bar: {df['num_trades'].mean():,.1f}")
    
    if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
        df['delta'] = df['ask_volume'] - df['bid_volume']
        print(f"Cumulative delta: {df['delta'].sum():,.0f}")
        print(f"Avg delta per bar: {df['delta'].mean():,.0f}")
    
    # Price statistics
    price_range = df['high'].max() - df['low'].min()
    print(f"\nPrice range: {df['low'].min():.2f} - {df['high'].max():.2f} (range: {price_range:.2f})")
    
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load and analyze Sierra Chart .scid files'
    )
    parser.add_argument('path', help='Path to .scid file')
    parser.add_argument('--max-records', type=int, default=None,
                       help='Maximum number of records to load')
    parser.add_argument('--tail', type=int, default=10,
                       help='Number of tail records to display')
    parser.add_argument('--save-csv', help='Save to CSV file')
    
    args = parser.parse_args()
    
    df = analyze_scid_file(args.path, max_records=args.max_records, tail_records=args.tail)
    
    if args.save_csv and df is not None:
        df.to_csv(args.save_csv, index=False)
        print(f"\n💾 Saved to {args.save_csv}")

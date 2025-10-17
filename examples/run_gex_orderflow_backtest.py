"""
Quick example script to run the GEX + Order Flow + MACD backtest.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
from market_ml.gex_orderflow_strategy import run_full_backtest


def plot_backtest_results(results_df, save_path=None):
    """
    Create visualization of backtest results.
    
    Args:
        results_df: DataFrame with backtest results
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # 1. Price & Signals
    ax1 = axes[0]
    ax1.plot(results_df['timestamp'], results_df['price'], label='Price', color='black', linewidth=0.8)
    
    # Plot zero gamma if available
    if 'zero_gamma' in results_df.columns:
        ax1.plot(results_df['timestamp'], results_df['zero_gamma'], 
                label='Zero Gamma', color='red', linestyle='--', alpha=0.7)
    
    # Mark long entries
    longs = results_df[results_df['signal'].diff() == 1]
    ax1.scatter(longs['timestamp'], longs['price'], color='green', marker='^', 
               s=100, label='Long Entry', zorder=5)
    
    # Mark short entries
    shorts = results_df[results_df['signal'].diff() == -1]
    ax1.scatter(shorts['timestamp'], shorts['price'], color='red', marker='v', 
               s=100, label='Short Entry', zorder=5)
    
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.set_title('GEX + Order Flow + MACD Strategy Backtest')
    ax1.grid(True, alpha=0.3)
    
    # 2. MACD
    ax2 = axes[1]
    ax2.plot(results_df['timestamp'], results_df['macd'], label='MACD', color='blue')
    ax2.plot(results_df['timestamp'], results_df['macd_signal'], label='Signal', color='red')
    ax2.bar(results_df['timestamp'], results_df['macd_hist'], label='Histogram', 
           color='gray', alpha=0.3)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('MACD')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative Delta
    if 'cum_delta' in results_df.columns:
        ax3 = axes[2]
        ax3.plot(results_df['timestamp'], results_df['cum_delta'], 
                color='purple', label='Cumulative Delta')
        ax3.axhline(0, color='black', linewidth=0.5)
        ax3.fill_between(results_df['timestamp'], 0, results_df['cum_delta'],
                        where=results_df['cum_delta'] >= 0, color='green', alpha=0.3)
        ax3.fill_between(results_df['timestamp'], 0, results_df['cum_delta'],
                        where=results_df['cum_delta'] < 0, color='red', alpha=0.3)
        ax3.set_ylabel('Cumulative Delta')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
    
    # 4. Equity Curve
    if 'equity' in results_df.columns:
        ax4 = axes[3]
        ax4.plot(results_df['timestamp'], results_df['equity'], 
                color='darkgreen', linewidth=2, label='Strategy Equity')
        ax4.set_ylabel('Equity ($)')
        ax4.set_xlabel('Time')
        ax4.legend(loc='upper left')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def main():
    """
    Run example backtest on NQ futures with GEX data.
    """
    # Example: Use most recent data
    scid_file = '/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid'
    
    # Check if we have GEX data yet
    import os
    gex_file = 'outputs/intraday/NQ_NDX_gex_zero_intraday.csv'
    
    if not os.path.exists(gex_file):
        print(f"GEX file not found: {gex_file}")
        print("Note: Market hours collector needs to run first to generate GEX data")
        print("Using backfill data instead...")
        gex_file = 'outputs/backfill/NQ_NDX_gex_historical.csv'
    
    if not os.path.exists(gex_file):
        print(f"No GEX data available yet. Please wait for collectors to gather data.")
        print("Check status with: ./check_status.sh")
        return
    
    # Run backtest
    print("Running backtest...")
    results, metrics = run_full_backtest(
        scid_file=scid_file,
        gex_file=gex_file,
        output_file='outputs/gex_orderflow_backtest.csv',
        initial_capital=10000,
        transaction_fee=0.0004  # $0.40 per $1000 (typical futures commission)
    )
    
    # Print results
    print("\n" + "="*70)
    print("BACKTEST PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"Initial Capital: $10,000.00")
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:+.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    if metrics['win_rate']:
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print("="*70)
    
    # Plot results
    print("\nGenerating plots...")
    plot_backtest_results(results, save_path='outputs/gex_orderflow_backtest.png')
    
    # Show sample trades
    print("\nSample Trades:")
    print("-" * 70)
    trades = results[results['signal'].diff().fillna(0) != 0].head(10)
    for idx, row in trades.iterrows():
        direction = "LONG" if row['signal'] == 1 else "SHORT" if row['signal'] == -1 else "FLAT"
        print(f"{row['timestamp']} | {direction:5s} | Price: ${row['price']:,.2f} | "
              f"Equity: ${row.get('equity', 0):,.2f}")
    
    print("\nResults saved to:")
    print("  - outputs/gex_orderflow_backtest.csv")
    print("  - outputs/gex_orderflow_backtest.png")


if __name__ == '__main__':
    main()

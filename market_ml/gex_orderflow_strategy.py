"""
GEX + Order Flow + MACD Strategy

Combines:
1. GEX data (zero gamma levels, major strikes)
2. SCID order flow (cumulative delta, volume)
3. MACD indicator (12, 26, 9 default settings)

Strategy Logic:
- LONG: MACD crossover + price above zero gamma + positive cumulative delta
- SHORT: MACD crossunder + price below zero gamma + negative cumulative delta
"""
import pandas as pd
import numpy as np
from pathlib import Path


def calculate_macd(df, price_col='close', fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator with default settings.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for price
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
    
    Returns:
        DataFrame with MACD columns added
    """
    df = df.copy()
    
    # Calculate EMAs
    ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
    
    # MACD line
    df['macd'] = ema_fast - ema_slow
    
    # Signal line
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Histogram
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Crossover signals
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) & 
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    )
    df['macd_cross_down'] = (
        (df['macd'] < df['macd_signal']) & 
        (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    )
    
    return df


def calculate_order_flow_features(df):
    """
    Calculate order flow features from SCID data.
    
    Args:
        df: DataFrame with bid_volume, ask_volume
    
    Returns:
        DataFrame with order flow features
    """
    df = df.copy()
    
    # Delta (buyer vs seller aggression)
    df['delta'] = df['ask_volume'] - df['bid_volume']
    
    # Cumulative delta
    df['cum_delta'] = df['delta'].cumsum()
    
    # Delta change rate (momentum)
    df['delta_change'] = df['delta'].diff()
    
    # Volume weighted delta
    df['total_volume_safe'] = df['total_volume'].replace(0, 1)  # Avoid division by zero
    df['delta_ratio'] = df['delta'] / df['total_volume_safe']
    
    # Rolling metrics (1-minute = 60 bars if tick data)
    # Adjust window based on your data frequency
    window = 20  # Will adjust based on actual bar frequency
    
    df['delta_ma'] = df['delta'].rolling(window).mean()
    df['cum_delta_slope'] = df['cum_delta'].diff(window)
    df['volume_ma'] = df['total_volume'].rolling(window).mean()
    
    return df


def add_gex_features(scid_df, gex_df):
    """
    Merge GEX data with SCID tick data.
    
    Args:
        scid_df: DataFrame with SCID tick data
        gex_df: DataFrame with GEX levels
    
    Returns:
        Merged DataFrame with GEX features
    """
    # Merge using nearest backward lookup (asof)
    merged = pd.merge_asof(
        scid_df.sort_values('timestamp'),
        gex_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward',
        suffixes=('', '_gex')
    )
    
    # Calculate distance from key levels
    if 'zero_gamma' in merged.columns:
        merged['distance_from_zero_gamma'] = merged['close'] - merged['zero_gamma']
        merged['above_zero_gamma'] = merged['distance_from_zero_gamma'] > 0
        merged['pct_from_zero_gamma'] = (
            merged['distance_from_zero_gamma'] / merged['zero_gamma'] * 100
        )
    
    # Major level distances
    if 'major_call_strike_1' in merged.columns:
        merged['distance_from_call_1'] = merged['close'] - merged['major_call_strike_1']
    
    if 'major_put_strike_1' in merged.columns:
        merged['distance_from_put_1'] = merged['close'] - merged['major_put_strike_1']
    
    return merged


def resample_to_1min(df):
    """
    Resample tick data to 1-minute bars for MACD calculation.
    
    Args:
        df: DataFrame with tick data (must have timestamp index)
    
    Returns:
        DataFrame resampled to 1-minute OHLCV
    """
    df = df.set_index('timestamp') if 'timestamp' in df.columns else df
    
    # Resample to 1-minute bars
    ohlc = df['close'].resample('1min').ohlc()
    volume = df['total_volume'].resample('1min').sum()
    
    # Order flow aggregation
    delta_sum = df['delta'].resample('1min').sum() if 'delta' in df.columns else None
    cum_delta = df['cum_delta'].resample('1min').last() if 'cum_delta' in df.columns else None
    
    # GEX features (last value in period)
    gex_cols = [col for col in df.columns if any(x in col for x in 
                ['zero_gamma', 'major_', 'sum_gex', 'above_zero'])]
    
    resampled = ohlc.copy()
    resampled['volume'] = volume
    
    if delta_sum is not None:
        resampled['delta'] = delta_sum
    if cum_delta is not None:
        resampled['cum_delta'] = cum_delta
    
    # Add GEX features
    for col in gex_cols:
        resampled[col] = df[col].resample('1min').last()
    
    resampled = resampled.reset_index()
    
    return resampled


def generate_signals(df):
    """
    Generate trading signals from combined features.
    
    Strategy:
    - LONG: MACD crossover + above zero gamma + positive delta trend
    - SHORT: MACD crossunder + below zero gamma + negative delta trend
    - EXIT: Opposite signal or stop loss
    
    Args:
        df: DataFrame with all features (MACD, GEX, order flow)
    
    Returns:
        DataFrame with signal column (1=long, -1=short, 0=flat)
    """
    df = df.copy()
    
    # Initialize signals
    df['signal'] = 0
    
    # Long conditions
    long_macd = df['macd_cross_up'] == True
    if 'above_zero_gamma' in df.columns:
        long_gex = df['above_zero_gamma'] == True
        short_gex = df['above_zero_gamma'] == False
    else:
        long_gex = True
        short_gex = True
    
    if 'cum_delta' in df.columns:
        long_delta = df['cum_delta'] > 0
        short_delta = df['cum_delta'] < 0
    else:
        long_delta = True
        short_delta = True
    
    # Short conditions
    short_macd = df['macd_cross_down'] == True
    
    # Combined signals
    df.loc[long_macd & long_gex & long_delta, 'signal'] = 1
    df.loc[short_macd & short_gex & short_delta, 'signal'] = -1
    
    # Forward fill signals (stay in position until opposite signal)
    df['signal'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
    
    return df


def backtest_strategy(df, initial_capital=10000, transaction_fee=0.0004, 
                     stop_loss_pct=None, take_profit_pct=None):
    """
    Backtest the GEX + Order Flow + MACD strategy.
    
    Args:
        df: DataFrame with signals
        initial_capital: Starting capital
        transaction_fee: Fee per trade (0.0004 = 0.04% = $0.40 per $1000)
        stop_loss_pct: Stop loss percentage (e.g., 0.02 = 2%)
        take_profit_pct: Take profit percentage (e.g., 0.03 = 3%)
    
    Returns:
        DataFrame with backtest results
    """
    df = df.copy()
    
    # Rename close to price for consistency
    if 'close' in df.columns and 'price' not in df.columns:
        df['price'] = df['close']
    
    # Calculate returns and equity
        df['position'] = df['signal'].shift(1).fillna(0)  # Execute next bar
        df['returns'] = df['price'].pct_change(fill_method=None)
        # Ensure first return is zero to avoid propagating NaNs into equity
        df['returns'] = df['returns'].fillna(0)
        df['strategy_returns'] = df['position'] * df['returns']
    
        # Apply transaction costs on position changes
        df['position_change'] = df['position'].diff().abs().fillna(0)
        df['transaction_cost'] = df['position_change'] * transaction_fee
        df['strategy_returns_net'] = (df['strategy_returns'] - df['transaction_cost']).fillna(0)
    
        # Calculate equity curve (start exactly at initial_capital)
        df['equity'] = initial_capital * (1 + df['strategy_returns_net']).cumprod()
    
    # Calculate PnL
    df['pnl'] = df['equity'].diff().fillna(0)
    df['net_pnl'] = df['pnl']
    
    # Add stop loss / take profit if specified
    if stop_loss_pct or take_profit_pct:
        df = apply_risk_management(df, stop_loss_pct, take_profit_pct)
    
    return df


def apply_risk_management(df, stop_loss_pct=None, take_profit_pct=None):
    """
    Apply stop loss and take profit to backtest.
    
    Args:
        df: DataFrame with positions and prices
        stop_loss_pct: Stop loss percentage (e.g., 0.02 = 2%)
        take_profit_pct: Take profit percentage (e.g., 0.03 = 3%)
    
    Returns:
        DataFrame with risk management applied
    """
    df = df.copy()
    
    # Track entry price for each position
    df['entry_price'] = np.nan
    df['position_change'] = df['signal'].diff().fillna(0) != 0
    df.loc[df['position_change'], 'entry_price'] = df.loc[df['position_change'], 'price']
    df['entry_price'] = df['entry_price'].ffill()
    
    # Calculate returns from entry
    df['return_from_entry'] = (df['price'] - df['entry_price']) / df['entry_price']
    df.loc[df['signal'] == -1, 'return_from_entry'] *= -1  # Invert for shorts
    
    # Hit stop loss or take profit
    df['hit_stop'] = False
    df['hit_target'] = False
    
    if stop_loss_pct:
        df['hit_stop'] = (df['signal'] != 0) & (df['return_from_entry'] <= -stop_loss_pct)
    
    if take_profit_pct:
        df['hit_target'] = (df['signal'] != 0) & (df['return_from_entry'] >= take_profit_pct)
    
    # Exit on stop or target
    df.loc[df['hit_stop'] | df['hit_target'], 'signal'] = 0
    
    return df


def calculate_performance_metrics(df):
    """
    Calculate strategy performance metrics.
    
    Args:
        df: DataFrame with backtest results
    
    Returns:
        dict with performance metrics
    """
    # Get final/initial equity from valid (non-NaN) rows
    if 'equity' in df.columns:
        eq_valid = df[['equity']].dropna()
        if not eq_valid.empty:
            initial_equity = float(eq_valid['equity'].iloc[0])
            final_equity = float(eq_valid['equity'].iloc[-1])
        else:
            initial_equity = 0.0
            final_equity = 0.0
    else:
        initial_equity = 0.0
        final_equity = 0.0

    # Total return
    total_return = ((final_equity / initial_equity - 1) * 100) if initial_equity > 0 else 0.0

    # Trade statistics (approximate by segments between signal changes)
    change_mask = df['signal'].fillna(0).astype(float).diff().fillna(0) != 0
    change_idx = list(df.index[change_mask])
    trade_pnls = []
    for i in range(0, len(change_idx) - 1):
        start_idx = change_idx[i]
        end_idx = change_idx[i + 1]
        # Consider segments where we were IN a position (signal after start != 0)
        start_signal = float(df.loc[start_idx, 'signal'])
        if start_signal != 0 and not np.isnan(df.loc[start_idx, 'equity']) and not np.isnan(df.loc[end_idx, 'equity']):
            trade_pnls.append(float(df.loc[end_idx, 'equity'] - df.loc[start_idx, 'equity']))
    num_trades = len(trade_pnls)
    wr = (100.0 * (np.array(trade_pnls) > 0).mean()) if num_trades > 0 else 0.0

    # Drawdown
    if 'equity' in df.columns:
        rolling_peak = df['equity'].cummax()
        drawdown = (df['equity'] - rolling_peak) / rolling_peak * 100
        max_drawdown = float(drawdown.min()) if drawdown.notna().any() else 0.0
    else:
        max_drawdown = 0.0

    # Sharpe ratio (using per-bar returns; for 1-minute bars, scale ~sqrt(252*390))
    if 'equity' in df.columns:
        bar_rets = df['equity'].pct_change().dropna()
        if bar_rets.std() and bar_rets.std() > 0:
            sharpe = float((bar_rets.mean() / bar_rets.std()) * np.sqrt(252))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0
    
    return {
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'num_trades': num_trades,
        'win_rate': wr,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe,
        'start_date': df['timestamp'].iloc[0] if 'timestamp' in df.columns else None,
        'end_date': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
    }


def run_full_backtest(scid_file, gex_file, output_file=None, 
                     initial_capital=10000, transaction_fee=0.0004):
    """
    Complete end-to-end backtest pipeline.
    
    Args:
        scid_file: Path to SCID file
        gex_file: Path to GEX CSV file
        output_file: Optional path to save results
        initial_capital: Starting capital
        transaction_fee: Transaction fee rate
    
    Returns:
        tuple: (results_df, metrics_dict)
    
    Example:
        results, metrics = run_full_backtest(
            '/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid',
            'outputs/intraday/NQ_NDX_gex_zero_intraday.csv',
            output_file='outputs/backtest_results.csv'
        )
        
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    """
    from market_ml.data import load_sierra_chart_scid
    
    print("Loading data...")
    # Load SCID data
    scid_df = load_sierra_chart_scid(scid_file)
    print(f"Loaded {len(scid_df):,} SCID records")
    
    # Load GEX data
    gex_df = pd.read_csv(gex_file, parse_dates=['timestamp'])
    print(f"Loaded {len(gex_df):,} GEX records")
    
    print("\nCalculating order flow features...")
    # Calculate order flow
    scid_df = calculate_order_flow_features(scid_df)
    
    print("Merging with GEX data...")
    # Merge with GEX
    merged_df = add_gex_features(scid_df, gex_df)
    
    print("Resampling to 1-minute bars...")
    # Resample to 1-minute for MACD
    df_1m = resample_to_1min(merged_df)

    # Add common indicators
    from market_ml.features import add_common_indicators
    print("Adding common indicators...")
    df_1m = add_common_indicators(df_1m, price_col='close', volume_col='total_volume')

    print("Calculating MACD...")
    # Calculate MACD
    df_1m = calculate_macd(df_1m, price_col='close')
    
    print("Generating signals...")
    # Generate signals
    df_1m = generate_signals(df_1m)
    
    print("Running backtest...")
    # Backtest
    results = backtest_strategy(df_1m, initial_capital, transaction_fee)
    
    print("Calculating metrics...")
    # Metrics
    metrics = calculate_performance_metrics(results)
    
    # Save if requested
    if output_file:
        results.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    
    return results, metrics


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GEX + Order Flow + MACD Strategy Backtest')
    parser.add_argument('--scid', required=True, help='Path to SCID file')
    parser.add_argument('--gex', required=True, help='Path to GEX CSV file')
    parser.add_argument('--output', help='Output CSV file for results')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--fee', type=float, default=0.0004, help='Transaction fee rate')
    
    args = parser.parse_args()
    
    results, metrics = run_full_backtest(
        args.scid,
        args.gex,
        output_file=args.output,
        initial_capital=args.capital,
        transaction_fee=args.fee
    )
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Period: {metrics['start_date']} to {metrics['end_date']}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Final Equity: ${metrics['final_equity']:,.2f}")
    print(f"Total Return: {metrics['total_return_pct']:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%" if metrics['win_rate'] else "Win Rate: N/A")
    print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print("="*60)

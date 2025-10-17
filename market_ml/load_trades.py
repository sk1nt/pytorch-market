"""
Helper functions to load and prepare trade data from various sources.
"""
import pandas as pd
import polars as pl
from pathlib import Path


def load_trades_from_csv(filepath, timestamp_col='timestamp', parse_dates=True):
    """
    Load trades from CSV file.
    
    Expected columns:
    - timestamp: Trade execution time
    - price: Execution price
    - signal: 1 (long), -1 (short), 0 (flat)
    
    Optional columns (will be calculated if missing):
    - position: Position size
    - equity: Account equity
    - gross_pnl, net_pnl, fees, etc.
    
    Args:
        filepath: Path to CSV file
        timestamp_col: Name of timestamp column
        parse_dates: Whether to parse timestamp as datetime
    
    Returns:
        pandas DataFrame with trades
    """
    if parse_dates:
        df = pd.read_csv(filepath, parse_dates=[timestamp_col])
    else:
        df = pd.read_csv(filepath)
    
    # Validate required columns
    required = ['timestamp', 'price', 'signal']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def load_trades_from_broker_csv(filepath, broker='interactive_brokers'):
    """
    Load trades from broker-specific CSV format.
    
    Supported brokers:
    - 'interactive_brokers': IB Flex Query format
    - 'tradovate': Tradovate trade history
    - 'ninja_trader': NinjaTrader trade export
    - 'sierra_chart': Sierra Chart trade list
    
    Args:
        filepath: Path to broker CSV
        broker: Broker name
    
    Returns:
        pandas DataFrame in standardized format
    """
    df = pd.read_csv(filepath)
    
    if broker == 'interactive_brokers':
        # IB Flex Query columns: DateTime, Symbol, Buy/Sell, Quantity, Price, Proceeds
        df = df.rename(columns={
            'DateTime': 'timestamp',
            'Price': 'price',
            'Quantity': 'quantity'
        })
        # Convert Buy/Sell to signal
        df['signal'] = df['Buy/Sell'].map({'BUY': 1, 'SELL': -1, 'BOT': 1, 'SLD': -1})
        
    elif broker == 'tradovate':
        # Tradovate: Time, Contract, Side, Qty, Price
        df = df.rename(columns={
            'Time': 'timestamp',
            'Price': 'price',
            'Qty': 'quantity'
        })
        df['signal'] = df['Side'].map({'Buy': 1, 'Sell': -1})
        
    elif broker == 'ninja_trader':
        # NinjaTrader: Time, Instrument, Action, Quantity, Price
        df = df.rename(columns={
            'Time': 'timestamp',
            'Price': 'price',
            'Quantity': 'quantity'
        })
        df['signal'] = df['Action'].map({'Buy': 1, 'Sell': -1})
        
    elif broker == 'sierra_chart':
        # Sierra Chart: DateTime, Type, Price, Quantity
        df = df.rename(columns={
            'DateTime': 'timestamp',
            'Price': 'price',
            'Quantity': 'quantity'
        })
        df['signal'] = df['Type'].map({'Buy': 1, 'Sell': -1, 'B': 1, 'S': -1})
    
    else:
        raise ValueError(f"Unsupported broker: {broker}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df[['timestamp', 'price', 'signal', 'quantity']]


def load_trades_from_scid(filepath, strategy_func):
    """
    Generate trades by applying a strategy to SCID tick data.
    
    Args:
        filepath: Path to .scid file
        strategy_func: Function that takes DataFrame and returns signals
                      Signature: func(df) -> Series of {1, -1, 0}
    
    Returns:
        pandas DataFrame with trades
    
    Example:
        def my_strategy(df):
            # Simple delta strategy
            df['delta'] = df['ask_volume'] - df['bid_volume']
            df['cum_delta'] = df['delta'].cumsum()
            
            # Go long when cumulative delta crosses above 1000
            # Go short when crosses below -1000
            signals = pd.Series(0, index=df.index)
            signals[df['cum_delta'] > 1000] = 1
            signals[df['cum_delta'] < -1000] = -1
            
            return signals
        
        trades = load_trades_from_scid('MNQZ25_FUT_CME.scid', my_strategy)
    """
    from market_ml.data import load_sierra_chart_scid
    
    # Load tick data
    df = load_sierra_chart_scid(filepath)
    
    # Apply strategy to generate signals
    df['signal'] = strategy_func(df)
    
    # Extract trades (only where signal changes)
    df['signal_change'] = df['signal'].diff().fillna(0) != 0
    trades = df[df['signal_change']].copy()
    
    # Use close price as execution price
    trades = trades.rename(columns={'close': 'price'})
    
    return trades[['timestamp', 'price', 'signal']]


def combine_trades_with_gex(trades_df, gex_df, merge_type='asof'):
    """
    Combine trade signals with GEX data.
    
    Args:
        trades_df: DataFrame with trades (timestamp, price, signal)
        gex_df: DataFrame with GEX data (timestamp, zero_gamma, etc.)
        merge_type: 'asof' (nearest backward), 'inner', 'left'
    
    Returns:
        DataFrame with trades + GEX features
    
    Example:
        trades = load_trades_from_csv('my_trades.csv')
        gex = pd.read_csv('outputs/intraday/NQ_NDX_gex_zero_intraday.csv', 
                         parse_dates=['timestamp'])
        
        combined = combine_trades_with_gex(trades, gex)
        # Now you have: timestamp, price, signal, zero_gamma, major_call_oi, etc.
    """
    if merge_type == 'asof':
        # Merge using nearest backward timestamp
        result = pd.merge_asof(
            trades_df.sort_values('timestamp'),
            gex_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
    else:
        result = pd.merge(trades_df, gex_df, on='timestamp', how=merge_type)
    
    return result


def validate_trades(df):
    """
    Validate trade DataFrame has required format.
    
    Args:
        df: Trade DataFrame
    
    Returns:
        dict with validation results
    """
    issues = []
    
    # Check required columns
    required = ['timestamp', 'price', 'signal']
    missing = [col for col in required if col not in df.columns]
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    # Check data types
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        issues.append("timestamp column is not datetime type")
    
    # Check signal values
    if 'signal' in df.columns:
        valid_signals = {-1, 0, 1}
        actual_signals = set(df['signal'].unique())
        invalid = actual_signals - valid_signals
        if invalid:
            issues.append(f"Invalid signal values: {invalid} (expected -1, 0, 1)")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"Columns with NaN values: {nan_cols}")
    
    # Check chronological order
    if 'timestamp' in df.columns and not df['timestamp'].is_monotonic_increasing:
        issues.append("Timestamps are not in chronological order")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'num_trades': len(df),
        'date_range': (df['timestamp'].min(), df['timestamp'].max()) if 'timestamp' in df.columns else None,
        'signal_distribution': df['signal'].value_counts().to_dict() if 'signal' in df.columns else None
    }


# Example usage
if __name__ == '__main__':
    # Example 1: Load existing trades
    print("Example 1: Load from CSV")
    print("=" * 60)
    trades = load_trades_from_csv('outputs/strategy_trades.csv')
    validation = validate_trades(trades)
    print(f"Valid: {validation['valid']}")
    print(f"Trades: {validation['num_trades']}")
    print(f"Date range: {validation['date_range']}")
    print(f"Signal distribution: {validation['signal_distribution']}")
    print()
    
    # Example 2: SCID-based strategy
    print("Example 2: Generate from SCID file")
    print("=" * 60)
    
    def simple_delta_strategy(df):
        """Example: Trade based on cumulative delta"""
        df['delta'] = df['ask_volume'] - df['bid_volume']
        df['cum_delta'] = df['delta'].cumsum()
        
        signals = pd.Series(0, index=df.index)
        signals[df['cum_delta'] > 1000] = 1
        signals[df['cum_delta'] < -1000] = -1
        
        return signals
    
    # Would load from SCID:
    # trades = load_trades_from_scid('/mnt/c/SierraChart/Data/MNQZ25_FUT_CME.scid', 
    #                                simple_delta_strategy)
    print("See function docstring for example")

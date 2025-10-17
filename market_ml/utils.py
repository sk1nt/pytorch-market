def add_liquidation_price(
    df,
    open_col='open',
    dir_col='dir_signal',
    leverage=1.0,
    maintenance_margin=0.005,
    out_col='liquidation_price',
):
    """
    Adds a liquidation_price column to a DataFrame, using long/short formulas per row.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                pl.when(pl.col(dir_col) == 1)
                  .then((pl.col(open_col) * leverage) / (leverage + 1 - maintenance_margin * leverage))
                  .when(pl.col(dir_col) == -1)
                  .then((pl.col(open_col) * leverage) / (leverage - 1 + maintenance_margin * leverage))
                  .otherwise(None)
                  .alias(out_col)
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    def calc_liq(row):
        if row[dir_col] == 1:
            return (row[open_col] * leverage) / (leverage + 1 - maintenance_margin * leverage)
        elif row[dir_col] == -1:
            return (row[open_col] * leverage) / (leverage - 1 + maintenance_margin * leverage)
        else:
            return None
    df[out_col] = df.apply(calc_liq, axis=1)
    return df
def long_liquidation_price(p, l, mmr):
    """
    Calculates the liquidation price for a long position.
    p: entry price
    l: leverage
    mmr: maintenance margin rate
    """
    return (p * l) / (l + 1 - mmr * l)

def short_liquidation_price(p, l, mmr):
    """
    Calculates the liquidation price for a short position.
    p: entry price
    l: leverage
    mmr: maintenance margin rate
    """
    return (p * l) / (l - 1 + mmr * l)
def win_rate(
    df,
    net_pnl_col='trade_net_taker_pnl',
):
    """
    Calculates win rate (percentage of trades with positive net PnL).
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            wins = (df[net_pnl_col] > 0).sum()
            total = df.height
            return wins / total if total > 0 else np.nan
    except ImportError:
        pass
    # Fallback to pandas
    wins = (df[net_pnl_col] > 0).sum()
    total = len(df)
    return wins / total if total > 0 else np.nan
def add_compounding_trades(
    df,
    capital=100,
    leverage=1.0,
    maker_fee=0.0002,
    taker_fee=0.0005,
    log_return_col='trade_log_return',
    open_col='open',
    dir_col='dir_signal',
    entry_col='entry_trade_value',
    exit_col='exit_trade_value',
    signed_qty_col='signed_trade_qty',
    gross_pnl_col='trade_gross_pnl',
    net_taker_col='trade_net_taker_pnl',
    net_maker_col='trade_net_maker_pnl',
    tx_fee_maker_col='tx_fee_maker',
    tx_fee_taker_col='tx_fee_taker',
):
    """
    Adds compounding trade sizing, leverage, transaction fees, and net PnL columns to a DataFrame.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            df = df.with_columns([
                pl.col(log_return_col).cumsum().alias('cum_trade_log_return'),
            ])
            df = df.with_columns([
                (pl.col('cum_trade_log_return').exp() * capital * leverage).shift().fill_null(capital * leverage).alias(entry_col),
                (pl.col('cum_trade_log_return').exp() * capital * leverage).alias(exit_col),
            ])
            df = df.with_columns([
                (pl.col(entry_col) / pl.col(open_col) * pl.col(dir_col)).alias(signed_qty_col),
                (pl.col(exit_col) - pl.col(entry_col)).alias(gross_pnl_col),
                (pl.col(entry_col) * maker_fee + pl.col(exit_col) * maker_fee).alias(tx_fee_maker_col),
                (pl.col(entry_col) * taker_fee + pl.col(exit_col) * taker_fee).alias(tx_fee_taker_col),
                (pl.col(gross_pnl_col) - pl.col(tx_fee_taker_col)).alias(net_taker_col),
                (pl.col(gross_pnl_col) - pl.col(tx_fee_maker_col)).alias(net_maker_col),
            ])
            return df
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df['cum_trade_log_return'] = df[log_return_col].cumsum()
    df[entry_col] = np.roll(capital * leverage * np.exp(df['cum_trade_log_return'].values), 1)
    df[entry_col][0] = capital * leverage
    df[exit_col] = capital * leverage * np.exp(df['cum_trade_log_return'])
    df[signed_qty_col] = df[entry_col] / df[open_col] * df[dir_col]
    df[gross_pnl_col] = df[exit_col] - df[entry_col]
    df[tx_fee_maker_col] = df[entry_col] * maker_fee + df[exit_col] * maker_fee
    df[tx_fee_taker_col] = df[entry_col] * taker_fee + df[exit_col] * taker_fee
    df[net_taker_col] = df[gross_pnl_col] - df[tx_fee_taker_col]
    df[net_maker_col] = df[gross_pnl_col] - df[tx_fee_maker_col]
    return df
def add_trade_net_taker_pnl(
    df,
    gross_col='trade_gross_pnl',
    fee_col='tx_fee_taker',
    out_col='trade_net_taker_pnl',
):
    """
    Adds trade_net_taker_pnl column (trade_gross_pnl - tx_fee_taker) to a DataFrame.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                (pl.col(gross_col) - pl.col(fee_col)).alias(out_col)
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[out_col] = df[gross_col] - df[fee_col]
    return df
def add_tx_fees(
    df,
    maker_fee=0.0002,
    taker_fee=0.0005,
    entry_col='entry_trade_value',
    exit_col='exit_trade_value',
    maker_col='tx_fee_maker',
    taker_col='tx_fee_taker',
):
    """
    Adds transaction fee columns (tx_fee_maker, tx_fee_taker) to a DataFrame.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                (pl.col(entry_col) * maker_fee + pl.col(exit_col) * maker_fee).alias(maker_col),
                (pl.col(entry_col) * taker_fee + pl.col(exit_col) * taker_fee).alias(taker_col),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[maker_col] = df[entry_col] * maker_fee + df[exit_col] * maker_fee
    df[taker_col] = df[entry_col] * taker_fee + df[exit_col] * taker_fee
    return df
def add_compounding_trade_sizes_polars(
    df,
    log_return_col='trade_log_return',
    open_col='open',
    dir_col='dir_signal',
    capital=100,
    entry_col='entry_trade_value',
    exit_col='exit_trade_value',
    signed_qty_col='signed_trade_qty',
    gross_pnl_col='trade_gross_pnl',
):
    """
    Adds compounding entry/exit trade value and signed trade qty columns for polars DataFrames.
    Entry value is previous equity, exit value is current equity, qty is sized accordingly.
    """
    import polars as pl
    if not isinstance(df, pl.DataFrame):
        raise TypeError("This utility is for polars DataFrames only.")
    df = df.with_columns([
        pl.col(log_return_col).cumsum().alias('cum_trade_log_return'),
    ])
    df = df.with_columns([
        (pl.col('cum_trade_log_return').exp() * capital).shift().fill_null(capital).alias(entry_col),
        (pl.col('cum_trade_log_return').exp() * capital).alias(exit_col),
    ])
    df = df.with_columns([
        (pl.col(entry_col) / pl.col(open_col) * pl.col(dir_col)).alias(signed_qty_col),
        (pl.col(exit_col) - pl.col(entry_col)).alias(gross_pnl_col),
    ])
    return df
def equity_curve_polars(capital, col_name, suffix):
    """
    Returns a polars expression for equity curve: capital + cumulative sum of col_name.
    """
    import polars as pl
    return (capital + pl.col(col_name).cum_sum()).alias(f'equity_curve_{suffix}')

def add_equity_curves(
    df,
    capital=100,
    gross_col='trade_gross_pnl',
    taker_col='trade_net_taker_pnl',
    maker_col='trade_net_maker_pnl',
    gross_curve='equity_curve_gross',
    taker_curve='equity_curve_taker',
    maker_curve='equity_curve_maker',
):
    """
    Adds equity curve columns for gross, taker, and maker PnL to a DataFrame.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                equity_curve_polars(capital, gross_col, 'gross'),
                equity_curve_polars(capital, taker_col, 'taker'),
                equity_curve_polars(capital, maker_col, 'maker'),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[gross_curve] = capital + df[gross_col].cumsum()
    df[taker_curve] = capital + df[taker_col].cumsum()
    df[maker_curve] = capital + df[maker_col].cumsum()
    return df
def add_trade_net_pnl(
    df,
    gross_col='trade_gross_pnl',
    maker_fee_col='maker_fee',
    taker_fee_col='taker_fee',
    net_taker_col='trade_net_taker_pnl',
    net_maker_col='trade_net_maker_pnl',
):
    """
    Adds trade_net_taker_pnl and trade_net_maker_pnl columns to a DataFrame.
    Net PnL = trade_gross_pnl - fee (for both maker and taker).
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                (pl.col(gross_col) - pl.col(taker_fee_col)).alias(net_taker_col),
                (pl.col(gross_col) - pl.col(maker_fee_col)).alias(net_maker_col),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[net_taker_col] = df[gross_col] - df[taker_fee_col]
    df[net_maker_col] = df[gross_col] - df[maker_fee_col]
    return df
def add_transaction_fees(
    df,
    entry_col='entry_trade_value',
    exit_col='exit_trade_value',
    maker_fee=0.0002,
    taker_fee=0.0005,
    maker_col='maker_fee',
    taker_col='taker_fee',
):
    """
    Adds maker_fee and taker_fee columns to a DataFrame based on entry/exit trade values.
    Defaults use Binance-like rates (0.02% maker, 0.05% taker).
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                (pl.col(entry_col) * taker_fee + pl.col(exit_col) * taker_fee).alias(taker_col),
                (pl.col(entry_col) * maker_fee + pl.col(exit_col) * maker_fee).alias(maker_col),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[taker_col] = df[entry_col] * taker_fee + df[exit_col] * taker_fee
    df[maker_col] = df[entry_col] * maker_fee + df[exit_col] * maker_fee
    return df
import numpy as np
import matplotlib.pyplot as plt

def add_directional_signal(df, y_hat_col='y_hat', out_col='dir_signal'):
    """
    Adds a directional signal column to a DataFrame based on the sign of y_hat.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns(
                pl.col(y_hat_col).sign().alias(out_col)
            )
    except ImportError:
        pass
    # Fallback to pandas
    df[out_col] = np.sign(df[y_hat_col])
    return df

def plot_column(df, col, title=None, xlabel='Time', ylabel=None, figsize=(10,5)):
    """
    Plot a column from a pandas or polars DataFrame.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            y = df[col].to_numpy()
        else:
            y = df[col].values
    except ImportError:
        y = df[col].values
    plt.figure(figsize=figsize)
    plt.plot(y)
    plt.title(title or col)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel or col)
    plt.grid(True, alpha=0.3)
    plt.show()

def add_constant_trade_sizing(df, open_col='open', log_return_col='trade_log_return', dir_col='dir_signal', capital=100, ratio=1.0):
    """
    Adds constant trade sizing columns to a DataFrame (entry/exit value, qty, signed qty).
    Works for both pandas and polars DataFrames.
    """
    trade_value = ratio * capital
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns([
                pl.lit(trade_value).alias('entry_trade_value'),
                (trade_value * pl.col(log_return_col).exp()).alias('exit_trade_value'),
                (trade_value / pl.col(open_col)).alias('trade_qty'),
                (pl.col('trade_qty') * pl.col(dir_col)).alias('signed_trade_qty'),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df['entry_trade_value'] = trade_value
    df['exit_trade_value'] = trade_value * np.exp(df[log_return_col])
    df['trade_qty'] = trade_value / df[open_col]
    df['signed_trade_qty'] = df['trade_qty'] * df[dir_col]
    return df

def add_compounding_trade_sizing(df, open_col='open', log_return_col='trade_log_return', dir_col='dir_signal', initial_capital=100, ratio=1.0):
    """
    Adds compounding trade sizing columns to a DataFrame (entry/exit value, qty, signed qty).
    Each trade's size is based on the compounded equity curve (grows/shrinks with returns).
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            # Compute cumulative log return for compounding
            cum_log_return = df[log_return_col].cumsum()
            equity_curve = (initial_capital * (cum_log_return.apply(np.exp) if hasattr(cum_log_return, 'apply') else np.exp(cum_log_return)))
            # Add as a Series to polars DataFrame
            df = df.with_columns([
                pl.Series('compounding_equity', equity_curve),
            ])
            return df.with_columns([
                (pl.col('compounding_equity') * ratio).alias('entry_trade_value'),
                (pl.col('entry_trade_value') * pl.col(log_return_col).exp()).alias('exit_trade_value'),
                (pl.col('entry_trade_value') / pl.col(open_col)).alias('trade_qty'),
                (pl.col('trade_qty') * pl.col(dir_col)).alias('signed_trade_qty'),
            ])
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    cum_log_return = df[log_return_col].cumsum()
    df['compounding_equity'] = initial_capital * np.exp(cum_log_return)
    df['entry_trade_value'] = df['compounding_equity'] * ratio
    df['exit_trade_value'] = df['entry_trade_value'] * np.exp(df[log_return_col])
    df['trade_qty'] = df['entry_trade_value'] / df[open_col]
    df['signed_trade_qty'] = df['trade_qty'] * df[dir_col]
    return df

def add_trade_gross_pnl(df, entry_col='entry_trade_value', exit_col='exit_trade_value', out_col='trade_gross_pnl'):
    """
    Adds a trade_gross_pnl column (exit_trade_value - entry_trade_value) to a DataFrame.
    Works for both pandas and polars DataFrames.
    """
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.with_columns(
                (pl.col(exit_col) - pl.col(entry_col)).alias(out_col)
            )
    except ImportError:
        pass
    # Fallback to pandas
    df = df.copy()
    df[out_col] = df[exit_col] - df[entry_col]
    return df

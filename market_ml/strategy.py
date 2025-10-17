"""
Trading strategy utilities for position sizing, leverage, and liquidation modeling.

Implements concepts from "Let's Build a Quant Trading Strategy - Part 2":
- Constant position sizing
- Compounding position sizing
- Leverage calculation
- Liquidation simulation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategies."""
    
    initial_capital: float = 10000.0
    sizing_method: str = "constant"  # "constant" or "compounding"
    leverage: float = 1.0
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0005  # 0.05% taker fee
    liquidation_threshold: float = 0.0  # Margin threshold for liquidation (0 = no liquidation)
    entry_liquidity: str = "maker"  # "maker" or "taker" for entry leg fees


@dataclass
class TradeResult:
    """Result of a single trade with detailed metrics."""
    
    entry_value: float
    exit_value: float
    position_size: float
    direction: int  # 1 for long, -1 for short, 0 for flat
    gross_pnl: float
    tx_fees: float
    net_pnl: float
    is_liquidated: bool = False


def generate_signals_from_predictions(
    predictions: pd.Series,
    threshold: float = 0.0
) -> pd.Series:
    """
    Convert model predictions to trading signals.
    
    Args:
        predictions: Model predictions (can be probabilities, returns, etc.)
        threshold: Threshold for generating signals
        
    Returns:
        Series of signals: 1 (long), -1 (short), or 0 (flat)
        
    Example:
        >>> # Binary classifier probabilities
        >>> signals = generate_signals_from_predictions(probs, threshold=0.5)
        >>> # Expected returns
        >>> signals = generate_signals_from_predictions(expected_returns, threshold=0.0)
    """
    signals = pd.Series(0, index=predictions.index)
    signals[predictions > threshold] = 1
    signals[predictions < -threshold] = -1
    return signals


def calculate_constant_position_size(
    capital: float,
    price: float,
    leverage: float = 1.0
) -> float:
    """
    Calculate position size using constant capital allocation.
    
    Always uses the same amount of capital per trade, regardless of P&L.
    
    Args:
        capital: Initial capital to allocate
        price: Current asset price
        leverage: Leverage multiplier
        
    Returns:
        Position size in units of the asset
    """
    leveraged_capital = capital * leverage
    return leveraged_capital / price


def calculate_compounding_position_size(
    current_equity: float,
    price: float,
    leverage: float = 1.0
) -> float:
    """
    Calculate position size using compounding (reinvesting profits).
    
    Uses current account equity, allowing profits to compound.
    
    Args:
        current_equity: Current account equity (capital + accumulated P&L)
        price: Current asset price
        leverage: Leverage multiplier
        
    Returns:
        Position size in units of the asset
    """
    leveraged_equity = current_equity * leverage
    return leveraged_equity / price


def calculate_transaction_fees(
    entry_value: float,
    exit_value: float,
    maker_fee: float = 0.0002,
    taker_fee: float = 0.0005,
    entry_liquidity: str = "maker",
) -> Tuple[float, float]:
    """
    Calculate maker and taker transaction fees.
    
    Assumes entry is a maker order and exit is a taker order.
    
    Args:
        entry_value: Dollar value at entry
        exit_value: Dollar value at exit
        maker_fee: Maker fee rate
        taker_fee: Taker fee rate
        
    Returns:
        Tuple of (total_fees_with_maker, total_fees_with_taker)
    """
    if entry_liquidity not in ("maker", "taker"):
        raise ValueError("entry_liquidity must be 'maker' or 'taker'")

    maker_entry_fee = entry_value * maker_fee
    taker_entry_fee = entry_value * taker_fee
    taker_exit_fee = exit_value * taker_fee

    # For reporting we return (fees_maker_entry_model, fees_taker_entry_model)
    fees_maker = maker_entry_fee + taker_exit_fee
    fees_taker = taker_entry_fee + taker_exit_fee

    # For equity updates, the caller will choose which of these to apply based on config
    return fees_maker, fees_taker


def check_liquidation(
    equity: float,
    initial_capital: float,
    leverage: float,
    liquidation_threshold: float = 0.0
) -> bool:
    """
    Check if position should be liquidated due to margin call.
    
    Liquidation occurs when equity falls below a threshold relative to
    the leveraged position value.
    
    Args:
        equity: Current account equity
        initial_capital: Initial capital
        leverage: Leverage multiplier
        liquidation_threshold: Threshold for liquidation (0 = 100% loss, 0.5 = 50% loss)
        
    Returns:
        True if position should be liquidated
        
    Example:
        >>> # 2x leverage, liquidate if equity drops 50%
        >>> check_liquidation(5000, 10000, leverage=2.0, liquidation_threshold=0.5)
        True
    """
    if leverage <= 1.0:
        return False  # No liquidation without leverage
    
    # Calculate margin threshold
    # With 2x leverage, you're liquidated if you lose more than 50% of capital
    max_loss_before_liquidation = initial_capital * (1.0 / leverage)
    threshold_equity = initial_capital - max_loss_before_liquidation * (1 - liquidation_threshold)
    
    return equity <= threshold_equity


def run_constant_sizing_backtest(
    prices: pd.Series,
    signals: pd.Series,
    config: PositionSizingConfig
) -> pd.DataFrame:
    """
    Run backtest with constant position sizing.
    
    Each trade uses the same initial capital amount, regardless of P&L.
    
    Args:
        prices: Price series
        signals: Trading signals (1, -1, or 0)
        config: Position sizing configuration
        
    Returns:
        DataFrame with detailed trade-by-trade results
    """
    signals = signals.shift(1).fillna(0)  # Execute on next bar
    prices = prices.loc[signals.index]
    
    results = []
    equity = config.initial_capital
    position = 0.0
    position_value = 0.0
    
    for timestamp, signal in signals.items():
        price = prices[timestamp]
        
        # Close existing position if signal changes
        if position != 0 and signal != np.sign(position):
            exit_value = abs(position) * price
            gross_pnl = (exit_value - position_value) * np.sign(position)
            
            fees_maker, fees_taker = calculate_transaction_fees(
                position_value, exit_value,
                config.maker_fee, config.taker_fee,
                entry_liquidity=config.entry_liquidity,
            )
            
            net_pnl_maker = gross_pnl - fees_maker
            net_pnl_taker = gross_pnl - fees_taker
            
            equity += net_pnl_maker  # Use maker fee for equity update
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': 0,
                'position': 0.0,
                'position_value': 0.0,
                'equity': equity,
                'gross_pnl': gross_pnl,
                'fees_maker': fees_maker,
                'fees_taker': fees_taker,
                'net_pnl_maker': net_pnl_maker,
                'net_pnl_taker': net_pnl_taker,
                'is_liquidated': False,
            })
            
            position = 0.0
            position_value = 0.0
        
        # Open new position if signal is non-zero
        if signal != 0 and position == 0:
            position_size = calculate_constant_position_size(
                config.initial_capital, price, config.leverage
            )
            position = position_size * signal
            position_value = abs(position) * price
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': signal,
                'position': position,
                'position_value': position_value,
                'equity': equity,
                'gross_pnl': 0.0,
                'fees_maker': 0.0,
                'fees_taker': 0.0,
                'net_pnl_maker': 0.0,
                'net_pnl_taker': 0.0,
                'is_liquidated': False,
            })
        elif position != 0:
            # Mark to market
            current_value = abs(position) * price
            unrealized_pnl = (current_value - position_value) * np.sign(position)
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': np.sign(position),
                'position': position,
                'position_value': position_value,
                'equity': equity + unrealized_pnl,
                'gross_pnl': unrealized_pnl,
                'fees_maker': 0.0,
                'fees_taker': 0.0,
                'net_pnl_maker': unrealized_pnl,
                'net_pnl_taker': unrealized_pnl,
                'is_liquidated': False,
            })

    # Force close any open position at the end to realize P&L and fees
    if position != 0:
        last_ts = prices.index[-1]
        last_price = prices.iloc[-1]
        exit_value = abs(position) * last_price
        gross_pnl = (exit_value - position_value) * np.sign(position)

        fees_maker, fees_taker = calculate_transaction_fees(
            position_value, exit_value,
            config.maker_fee, config.taker_fee,
            entry_liquidity=config.entry_liquidity,
        )
        net_pnl_maker = gross_pnl - fees_maker
        net_pnl_taker = gross_pnl - fees_taker
        equity += net_pnl_maker

        results.append({
            'timestamp': last_ts,
            'price': last_price,
            'signal': 0,
            'position': 0.0,
            'position_value': 0.0,
            'equity': equity,
            'gross_pnl': gross_pnl,
            'fees_maker': fees_maker,
            'fees_taker': fees_taker,
            'net_pnl_maker': net_pnl_maker,
            'net_pnl_taker': net_pnl_taker,
            'is_liquidated': False,
        })
    
    df = pd.DataFrame(results)
    if len(df) > 0:
        df.set_index('timestamp', inplace=True)
    return df


def run_compounding_sizing_backtest(
    prices: pd.Series,
    signals: pd.Series,
    config: PositionSizingConfig
) -> pd.DataFrame:
    """
    Run backtest with compounding position sizing.
    
    Each trade uses current equity, allowing profits to compound and
    losses to reduce position size.
    
    Args:
        prices: Price series
        signals: Trading signals (1, -1, or 0)
        config: Position sizing configuration
        
    Returns:
        DataFrame with detailed trade-by-trade results
    """
    signals = signals.shift(1).fillna(0)  # Execute on next bar
    prices = prices.loc[signals.index]
    
    results = []
    equity = config.initial_capital
    position = 0.0
    position_value = 0.0
    is_liquidated = False
    
    for timestamp, signal in signals.items():
        if is_liquidated:
            # Account is liquidated, stop trading
            results.append({
                'timestamp': timestamp,
                'price': prices[timestamp],
                'signal': 0,
                'position': 0.0,
                'position_value': 0.0,
                'equity': 0.0,
                'gross_pnl': 0.0,
                'fees_maker': 0.0,
                'fees_taker': 0.0,
                'net_pnl_maker': 0.0,
                'net_pnl_taker': 0.0,
                'is_liquidated': True,
            })
            continue
        
        price = prices[timestamp]
        
        # Close existing position if signal changes
        if position != 0 and signal != np.sign(position):
            exit_value = abs(position) * price
            gross_pnl = (exit_value - position_value) * np.sign(position)
            
            fees_maker, fees_taker = calculate_transaction_fees(
                position_value, exit_value,
                config.maker_fee, config.taker_fee,
                entry_liquidity=config.entry_liquidity,
            )
            
            net_pnl_maker = gross_pnl - fees_maker
            net_pnl_taker = gross_pnl - fees_taker
            
            equity += net_pnl_maker  # Use maker fee for equity update
            
            # Check for liquidation
            if check_liquidation(equity, config.initial_capital, 
                                config.leverage, config.liquidation_threshold):
                is_liquidated = True
                equity = 0.0
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': 0,
                'position': 0.0,
                'position_value': 0.0,
                'equity': equity,
                'gross_pnl': gross_pnl,
                'fees_maker': fees_maker,
                'fees_taker': fees_taker,
                'net_pnl_maker': net_pnl_maker,
                'net_pnl_taker': net_pnl_taker,
                'is_liquidated': is_liquidated,
            })
            
            position = 0.0
            position_value = 0.0
        
        # Open new position if signal is non-zero
        if signal != 0 and position == 0 and not is_liquidated:
            position_size = calculate_compounding_position_size(
                equity, price, config.leverage
            )
            position = position_size * signal
            position_value = abs(position) * price
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': signal,
                'position': position,
                'position_value': position_value,
                'equity': equity,
                'gross_pnl': 0.0,
                'fees_maker': 0.0,
                'fees_taker': 0.0,
                'net_pnl_maker': 0.0,
                'net_pnl_taker': 0.0,
                'is_liquidated': False,
            })
        elif position != 0 and not is_liquidated:
            # Mark to market
            current_value = abs(position) * price
            unrealized_pnl = (current_value - position_value) * np.sign(position)
            current_equity = equity + unrealized_pnl
            
            # Check for liquidation on unrealized loss
            if check_liquidation(current_equity, config.initial_capital,
                                config.leverage, config.liquidation_threshold):
                is_liquidated = True
                current_equity = 0.0
            
            results.append({
                'timestamp': timestamp,
                'price': price,
                'signal': np.sign(position),
                'position': position,
                'position_value': position_value,
                'equity': current_equity,
                'gross_pnl': unrealized_pnl,
                'fees_maker': 0.0,
                'fees_taker': 0.0,
                'net_pnl_maker': unrealized_pnl,
                'net_pnl_taker': unrealized_pnl,
                'is_liquidated': is_liquidated,
            })
    
    df = pd.DataFrame(results)
    # Force close any open position at the end to realize P&L and fees
    if (len(df) > 0) and (not is_liquidated) and (position != 0):
        last_ts = prices.index[-1]
        last_price = prices.iloc[-1]
        exit_value = abs(position) * last_price
        gross_pnl = (exit_value - position_value) * np.sign(position)

        fees_maker, fees_taker = calculate_transaction_fees(
            position_value, exit_value,
            config.maker_fee, config.taker_fee
        )
        net_pnl_maker = gross_pnl - fees_maker
        net_pnl_taker = gross_pnl - fees_taker
        equity = max(0.0, equity + net_pnl_maker)

        exit_row = {
            'timestamp': last_ts,
            'price': last_price,
            'signal': 0,
            'position': 0.0,
            'position_value': 0.0,
            'equity': equity,
            'gross_pnl': gross_pnl,
            'fees_maker': fees_maker,
            'fees_taker': fees_taker,
            'net_pnl_maker': net_pnl_maker,
            'net_pnl_taker': net_pnl_taker,
            'is_liquidated': is_liquidated,
        }
        df = pd.concat([df, pd.DataFrame([exit_row])], ignore_index=True)
    if len(df) > 0:
        df.set_index('timestamp', inplace=True)
    return df


def run_strategy_backtest(
    prices: pd.Series,
    signals: pd.Series,
    config: PositionSizingConfig
) -> pd.DataFrame:
    """
    Run backtest with specified sizing method.
    
    Wrapper that calls appropriate backtest based on config.
    
    Args:
        prices: Price series
        signals: Trading signals (1, -1, or 0)
        config: Position sizing configuration
        
    Returns:
        DataFrame with detailed trade-by-trade results
    """
    if config.sizing_method == "constant":
        return run_constant_sizing_backtest(prices, signals, config)
    elif config.sizing_method == "compounding":
        return run_compounding_sizing_backtest(prices, signals, config)
    else:
        raise ValueError(f"Unknown sizing method: {config.sizing_method}")


def calculate_strategy_metrics(results_df: pd.DataFrame) -> dict:
    """
    Calculate performance metrics from backtest results.
    
    Args:
        results_df: Results from run_strategy_backtest
        
    Returns:
        Dictionary of performance metrics
    """
    if len(results_df) == 0:
        return {}
    
    initial_equity = results_df['equity'].iloc[0]
    final_equity = results_df['equity'].iloc[-1]
    
    # Calculate returns
    equity_series = results_df['equity']
    returns = equity_series.pct_change().fillna(0)
    
    # Check if liquidated
    was_liquidated = results_df['is_liquidated'].any()
    
    # Duration in years
    duration_days = (results_df.index[-1] - results_df.index[0]).days
    duration_years = duration_days / 365.25 if duration_days > 0 else 1
    
    # CAGR
    if final_equity > 0 and duration_years > 0:
        cagr = (final_equity / initial_equity) ** (1 / duration_years) - 1
    else:
        cagr = -1.0
    
    # Sharpe ratio
    if returns.std() > 0:
        sharpe = np.sqrt(252) * returns.mean() / returns.std()
    else:
        sharpe = 0.0
    
    # Max drawdown
    cumulative_max = equity_series.cummax()
    drawdown = equity_series / cumulative_max - 1
    max_drawdown = drawdown.min()
    
    # Win rate (on realized trades)
    realized_pnls = results_df[results_df['net_pnl_maker'] != 0]['net_pnl_maker']
    if len(realized_pnls) > 0:
        win_rate = (realized_pnls > 0).sum() / len(realized_pnls)
        avg_win = realized_pnls[realized_pnls > 0].mean() if (realized_pnls > 0).any() else 0
        avg_loss = realized_pnls[realized_pnls < 0].mean() if (realized_pnls < 0).any() else 0
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
    
    return {
        'initial_equity': initial_equity,
        'final_equity': final_equity,
        'total_return': (final_equity / initial_equity - 1) if initial_equity > 0 else -1,
        'cagr': cagr,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'num_trades': len(realized_pnls),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'was_liquidated': was_liquidated,
    }


__all__ = [
    "PositionSizingConfig",
    "TradeResult",
    "generate_signals_from_predictions",
    "calculate_constant_position_size",
    "calculate_compounding_position_size",
    "calculate_transaction_fees",
    "check_liquidation",
    "run_constant_sizing_backtest",
    "run_compounding_sizing_backtest",
    "run_strategy_backtest",
    "calculate_strategy_metrics",
]

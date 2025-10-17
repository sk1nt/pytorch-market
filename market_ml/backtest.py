"""Simple vectorised backtesting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd



@dataclass
class BacktestResult:
    """Container holding an equity curve, summary statistics, and trade log."""
    equity_curve: pd.DataFrame
    metrics: Dict[str, float]
    trade_log: pd.DataFrame = None


def _compute_cagr(equity: pd.Series) -> float:
    duration_years = (equity.index[-1] - equity.index[0]).days / 365.25
    final_value = equity.iloc[-1]
    return final_value ** (1 / duration_years) - 1 if duration_years > 0 else np.nan


def _compute_max_drawdown(equity: pd.Series) -> float:
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1
    return drawdown.min()


def _compute_sharpe(returns: pd.Series) -> float:
    if returns.std() == 0:
        return 0.0
    return np.sqrt(252) * returns.mean() / returns.std()



def run_long_only_backtest(
    prices: pd.Series,
    signals: pd.Series,
    trading_cost: float = 0.0005,
    maker_fee: float = 0.0,
    taker_fee: float = 0.0,
    position_size: float = 1.0,
) -> BacktestResult:
    """Run a long-only backtest with advanced features: maker/taker fees, variable sizing, and trade log.

    Args:
        prices: Series of prices (aligned to signals)
        signals: Series of 0/1 (or -1/0/1 for shorting) signals
        trading_cost: Proportional round-trip commission/impact (legacy)
        maker_fee: Fee for providing liquidity (per trade, proportional)
        taker_fee: Fee for taking liquidity (per trade, proportional)
        position_size: Fraction of capital to allocate per trade (default 1.0)
    Returns:
        BacktestResult with equity curve, metrics, and trade log
    """
    prices = prices.loc[signals.index]
    returns = prices.pct_change().fillna(0.0)
    position = signals.shift(1).fillna(0.0)
    # Support variable position sizing (scalar or Series)
    if isinstance(position_size, (float, int)):
        position_size = pd.Series(position_size, index=position.index)
    position = position * position_size
    turnover = position.diff().abs().fillna(0.0)
    # Fee model: apply trading_cost (legacy), plus maker/taker fees on entry/exit
    # For simplicity, charge taker_fee on position increases, maker_fee on decreases
    entry_mask = (position.diff() > 0)
    exit_mask = (position.diff() < 0)
    fees = trading_cost * turnover + taker_fee * entry_mask.astype(float) * turnover + maker_fee * exit_mask.astype(float) * turnover
    strategy_returns = position * returns - fees
    strategy_equity = (1 + strategy_returns).cumprod()
    buy_hold_equity = (1 + returns).cumprod()

    # Trade log: record entry/exit, PnL, fees
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    entry_equity = None
    for i in range(1, len(position)):
        if not in_position and position.iloc[i-1] == 0 and position.iloc[i] != 0:
            # Entry
            in_position = True
            entry_idx = position.index[i]
            entry_price = prices.iloc[i]
            entry_equity = strategy_equity.iloc[i-1]
        elif in_position and position.iloc[i-1] != 0 and position.iloc[i] == 0:
            # Exit
            exit_idx = position.index[i]
            exit_price = prices.iloc[i]
            exit_equity = strategy_equity.iloc[i]
            trade_return = (exit_price - entry_price) / entry_price * np.sign(position.iloc[i-1])
            trade_pnl = exit_equity - entry_equity
            trades.append({
                "entry_time": entry_idx,
                "exit_time": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": int(np.sign(position.iloc[i-1])),
                "trade_return": trade_return,
                "trade_pnl": trade_pnl,
            })
            in_position = False
    trade_log = pd.DataFrame(trades)

    equity_curve = pd.DataFrame(
        {
            "strategy_equity": strategy_equity,
            "buy_hold_equity": buy_hold_equity,
            "position": position,
            "strategy_returns": strategy_returns,
            "asset_returns": returns,
            "fees": fees,
        }
    )

    metrics = {
        "cagr": _compute_cagr(strategy_equity),
        "buy_hold_cagr": _compute_cagr(buy_hold_equity),
        "sharpe": _compute_sharpe(strategy_returns),
        "max_drawdown": _compute_max_drawdown(strategy_equity),
        "turnover": turnover.mean(),
        "total_fees": fees.sum(),
        "num_trades": len(trade_log),
        "avg_trade_return": trade_log["trade_return"].mean() if not trade_log.empty else np.nan,
        "avg_trade_pnl": trade_log["trade_pnl"].mean() if not trade_log.empty else np.nan,
    }

    return BacktestResult(equity_curve=equity_curve, metrics=metrics, trade_log=trade_log)


__all__ = ["run_long_only_backtest", "BacktestResult"]

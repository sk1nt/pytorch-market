"""Simple vectorised backtesting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    """Container holding an equity curve and summary statistics."""

    equity_curve: pd.DataFrame
    metrics: Dict[str, float]


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
) -> BacktestResult:
    """Run a naive long-only backtest driven by binary signals.

    The function assumes that ``signals`` already align with ``prices``.
    Whenever the signal is ``1`` the strategy goes long on the next bar,
    otherwise it stays in cash.  The trading cost parameter represents the
    proportional round-trip commission/impact.
    """

    prices = prices.loc[signals.index]

    returns = prices.pct_change().fillna(0.0)
    position = signals.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)
    strategy_returns = position * returns - trading_cost * turnover

    strategy_equity = (1 + strategy_returns).cumprod()
    buy_hold_equity = (1 + returns).cumprod()

    equity_curve = pd.DataFrame(
        {
            "strategy_equity": strategy_equity,
            "buy_hold_equity": buy_hold_equity,
            "position": position,
            "strategy_returns": strategy_returns,
            "asset_returns": returns,
        }
    )

    metrics = {
        "cagr": _compute_cagr(strategy_equity),
        "buy_hold_cagr": _compute_cagr(buy_hold_equity),
        "sharpe": _compute_sharpe(strategy_returns),
        "max_drawdown": _compute_max_drawdown(strategy_equity),
        "turnover": turnover.mean(),
    }

    return BacktestResult(equity_curve=equity_curve, metrics=metrics)


__all__ = ["run_long_only_backtest", "BacktestResult"]

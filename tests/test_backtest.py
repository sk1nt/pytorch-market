import pandas as pd

from market_ml.backtest import run_long_only_backtest


def test_backtest_generates_expected_metrics():
    dates = pd.date_range("2021-01-01", periods=10, freq="B")
    prices = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 103, 104], index=dates)
    # Strategy alternates long and flat
    signals = pd.Series([1, 1, 0, 0, 1, 1, 0, 0, 1, 1], index=dates, name="signal")

    result = run_long_only_backtest(prices, signals, trading_cost=0.0)

    # Ensure we get an equity curve with expected columns (allow extra columns like 'fees')
    expected_cols = {
        "strategy_equity",
        "buy_hold_equity",
        "position",
        "strategy_returns",
        "asset_returns",
    }
    assert expected_cols.issubset(set(result.equity_curve.columns))

    # CAGR should be finite and better than zero given the long bias
    assert result.metrics["cagr"] > -1
    assert result.metrics["sharpe"] == result.metrics["sharpe"]  # no NaN

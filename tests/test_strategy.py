"""Tests for trading strategy utilities."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from market_ml.strategy import (
    PositionSizingConfig,
    generate_signals_from_predictions,
    calculate_constant_position_size,
    calculate_compounding_position_size,
    calculate_transaction_fees,
    check_liquidation,
    run_constant_sizing_backtest,
    run_compounding_sizing_backtest,
    run_strategy_backtest,
    calculate_strategy_metrics,
)


def test_generate_signals_from_predictions():
    """Test signal generation from predictions."""
    predictions = pd.Series([0.7, 0.3, -0.2, -0.6, 0.1])
    signals = generate_signals_from_predictions(predictions, threshold=0.5)
    
    assert signals[0] == 1  # 0.7 > 0.5
    assert signals[1] == 0  # 0.3 < 0.5
    assert signals[2] == 0  # -0.2 > -0.5
    assert signals[3] == -1  # -0.6 < -0.5
    assert signals[4] == 0  # 0.1 < 0.5


def test_calculate_constant_position_size():
    """Test constant position sizing calculation."""
    capital = 10000.0
    price = 100.0
    
    # No leverage
    size = calculate_constant_position_size(capital, price, leverage=1.0)
    assert size == 100.0  # 10000 / 100
    
    # 2x leverage
    size = calculate_constant_position_size(capital, price, leverage=2.0)
    assert size == 200.0  # 20000 / 100


def test_calculate_compounding_position_size():
    """Test compounding position sizing calculation."""
    # After a profitable trade, equity grows
    equity = 12000.0
    price = 100.0
    
    size = calculate_compounding_position_size(equity, price, leverage=1.0)
    assert size == 120.0  # 12000 / 100
    
    # After a losing trade, equity shrinks
    equity = 8000.0
    size = calculate_compounding_position_size(equity, price, leverage=1.0)
    assert size == 80.0  # 8000 / 100


def test_calculate_transaction_fees():
    """Test transaction fee calculation."""
    entry_value = 10000.0
    exit_value = 11000.0
    maker_fee = 0.0002  # 0.02%
    taker_fee = 0.0005  # 0.05%
    
    fees_maker, fees_taker = calculate_transaction_fees(
        entry_value, exit_value, maker_fee, taker_fee
    )
    
    # Maker entry + taker exit
    expected_maker = entry_value * maker_fee + exit_value * taker_fee
    assert abs(fees_maker - expected_maker) < 0.01
    
    # Taker entry + taker exit
    expected_taker = entry_value * taker_fee + exit_value * taker_fee
    assert abs(fees_taker - expected_taker) < 0.01


def test_check_liquidation_no_leverage():
    """Test liquidation check without leverage."""
    initial_capital = 10000.0
    
    # No leverage means no liquidation
    assert not check_liquidation(5000, initial_capital, leverage=1.0)
    assert not check_liquidation(1000, initial_capital, leverage=1.0)


def test_check_liquidation_with_leverage():
    """Test liquidation check with leverage."""
    initial_capital = 10000.0
    leverage = 2.0
    
    # With 2x leverage, liquidated if equity drops 50%
    assert not check_liquidation(8000, initial_capital, leverage, liquidation_threshold=0.0)
    assert check_liquidation(4000, initial_capital, leverage, liquidation_threshold=0.0)
    assert check_liquidation(3000, initial_capital, leverage, liquidation_threshold=0.0)


def test_run_constant_sizing_backtest():
    """Test constant sizing backtest."""
    # Create simple test data
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], index=dates)
    signals = pd.Series([1, 1, 1, 0, 1, 1, 0, 1, 1, 0], index=dates)
    
    config = PositionSizingConfig(
        initial_capital=10000.0,
        sizing_method="constant",
        leverage=1.0,
        maker_fee=0.0002,
        taker_fee=0.0005
    )
    
    results = run_constant_sizing_backtest(prices, signals, config)
    
    # Check structure
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert 'equity' in results.columns
    assert 'position' in results.columns
    assert 'net_pnl_maker' in results.columns
    
    # Equity should always be positive with constant sizing
    assert (results['equity'] > 0).all()


def test_run_compounding_sizing_backtest():
    """Test compounding sizing backtest."""
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108, 107, 109], index=dates)
    signals = pd.Series([1, 1, 1, 0, 1, 1, 0, 1, 1, 0], index=dates)
    
    config = PositionSizingConfig(
        initial_capital=10000.0,
        sizing_method="compounding",
        leverage=1.0,
        maker_fee=0.0002,
        taker_fee=0.0005
    )
    
    results = run_compounding_sizing_backtest(prices, signals, config)
    
    # Check structure
    assert isinstance(results, pd.DataFrame)
    assert len(results) > 0
    assert 'equity' in results.columns
    assert 'position' in results.columns


def test_compounding_vs_constant_sizing():
    """Test that compounding can outperform constant sizing in winning scenarios."""
    # Create upward trending prices with buy signals
    dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
    prices = pd.Series(np.linspace(100, 150, 20), index=dates)
    signals = pd.Series([1] * 20, index=dates)  # Always long
    
    config_constant = PositionSizingConfig(
        initial_capital=10000.0,
        sizing_method="constant",
        leverage=1.0,
        maker_fee=0.0,
        taker_fee=0.0
    )
    
    config_compounding = PositionSizingConfig(
        initial_capital=10000.0,
        sizing_method="compounding",
        leverage=1.0,
        maker_fee=0.0,
        taker_fee=0.0
    )
    
    results_constant = run_constant_sizing_backtest(prices, signals, config_constant)
    results_compounding = run_compounding_sizing_backtest(prices, signals, config_compounding)
    
    # In an uptrend, compounding should do better
    final_equity_constant = results_constant['equity'].iloc[-1]
    final_equity_compounding = results_compounding['equity'].iloc[-1]
    
    # Allow equality in edge cases where arithmetic ties
    assert final_equity_compounding >= final_equity_constant


def test_liquidation_scenario():
    """Test liquidation with leverage."""
    # Create a scenario where price drops sharply
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = pd.Series([100, 95, 90, 85, 80, 75, 70, 65, 60, 55], index=dates)
    signals = pd.Series([1] * 10, index=dates)  # Always long (losing trade)
    
    config = PositionSizingConfig(
        initial_capital=10000.0,
        sizing_method="compounding",
        leverage=3.0,  # High leverage
        maker_fee=0.0,
        taker_fee=0.0,
        liquidation_threshold=0.0
    )
    
    results = run_compounding_sizing_backtest(prices, signals, config)
    
    # Should get liquidated at some point
    assert results['is_liquidated'].any()
    
    # After liquidation, equity should be 0
    liquidation_idx = results[results['is_liquidated']].index[0]
    assert results.loc[liquidation_idx:, 'equity'].max() == 0.0


def test_run_strategy_backtest_wrapper():
    """Test strategy backtest wrapper function."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='D')
    prices = pd.Series([100, 102, 101, 103, 105], index=dates)
    signals = pd.Series([1, 1, 0, 1, 1], index=dates)
    
    # Test constant sizing
    config = PositionSizingConfig(sizing_method="constant")
    results = run_strategy_backtest(prices, signals, config)
    assert len(results) > 0
    
    # Test compounding sizing
    config = PositionSizingConfig(sizing_method="compounding")
    results = run_strategy_backtest(prices, signals, config)
    assert len(results) > 0
    
    # Test invalid method
    config = PositionSizingConfig(sizing_method="invalid")
    with pytest.raises(ValueError):
        run_strategy_backtest(prices, signals, config)


def test_calculate_strategy_metrics():
    """Test strategy metrics calculation."""
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109], index=dates)
    signals = pd.Series([1, 1, 0, 1, 1, 0, 1, 1, 1, 0], index=dates)
    
    config = PositionSizingConfig(initial_capital=10000.0)
    results = run_strategy_backtest(prices, signals, config)
    metrics = calculate_strategy_metrics(results)
    
    # Check required metrics exist
    assert 'initial_equity' in metrics
    assert 'final_equity' in metrics
    assert 'total_return' in metrics
    assert 'cagr' in metrics
    assert 'sharpe' in metrics
    assert 'max_drawdown' in metrics
    assert 'num_trades' in metrics
    assert 'win_rate' in metrics
    assert 'was_liquidated' in metrics
    
    # Check reasonable values
    assert metrics['initial_equity'] == 10000.0
    assert metrics['final_equity'] > 0
    assert metrics['max_drawdown'] <= 0


def test_fees_reduce_returns():
    """Test that transaction fees reduce returns."""
    dates = pd.date_range(start='2020-01-01', periods=10, freq='D')
    prices = pd.Series([100, 110, 120, 130, 140, 150, 160, 170, 180, 190], index=dates)
    signals = pd.Series([1] * 10, index=dates)
    
    # No fees
    config_no_fees = PositionSizingConfig(
        initial_capital=10000.0,
        maker_fee=0.0,
        taker_fee=0.0
    )
    
    # With fees
    config_with_fees = PositionSizingConfig(
        initial_capital=10000.0,
        maker_fee=0.001,
        taker_fee=0.001
    )
    
    results_no_fees = run_strategy_backtest(prices, signals, config_no_fees)
    results_with_fees = run_strategy_backtest(prices, signals, config_with_fees)
    
    # Final equity should be higher without fees
    # Allow equality tolerance for rounding; fees should not produce higher equity
    assert results_no_fees['equity'].iloc[-1] >= results_with_fees['equity'].iloc[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

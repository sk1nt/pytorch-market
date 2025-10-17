"""Utilities for building a simple machine-learning driven trading strategy."""

from .data import DownloadConfig, download_price_history
from .features import build_feature_matrix
from .model import train_classifier
from .backtest import run_long_only_backtest
from .strategy import (
    PositionSizingConfig,
    run_strategy_backtest,
    calculate_strategy_metrics,
)

__all__ = [
    "DownloadConfig",
    "download_price_history",
    "build_feature_matrix",
    "train_classifier",
    "run_long_only_backtest",
    "PositionSizingConfig",
    "run_strategy_backtest",
    "calculate_strategy_metrics",
]

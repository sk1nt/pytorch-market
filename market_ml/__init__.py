"""Utilities for building a simple machine-learning driven trading strategy."""

from .data import DownloadConfig, download_price_history
from .features import build_feature_matrix
from .model import train_classifier
from .backtest import run_long_only_backtest

__all__ = [
    "DownloadConfig",
    "download_price_history",
    "build_feature_matrix",
    "train_classifier",
    "run_long_only_backtest",
]

"""Feature engineering utilities."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pandas as pd


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Return the Relative Strength Index for the provided closing prices."""

    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    roll_up = up.rolling(window=window, min_periods=window).mean()
    roll_down = down.rolling(window=window, min_periods=window).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_feature_matrix(data: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a feature matrix ``X`` and label vector ``y``.

    Parameters
    ----------
    data:
        DataFrame as returned by :func:`market_ml.data.download_price_history`.
    ticker:
        Symbol to extract from the multi-index columns produced by yfinance.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.Series]
        ``X`` contains engineered features. ``y`` is the binary label
        representing whether the next day's return is positive (``1``) or not
        (``0``).
    """

    try:
        adj_close = data["Adj Close"][ticker].rename("adj_close")
        close = data["Close"][ticker].rename("close")
        high = data["High"][ticker].rename("high")
        low = data["Low"][ticker].rename("low")
        volume = data["Volume"][ticker].rename("volume")
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError(
            f"Unable to locate ticker '{ticker}' in the downloaded price history."
        ) from exc

    returns = adj_close.pct_change().rename("return_1d")
    log_returns = np.log(adj_close / adj_close.shift(1)).rename("log_return_1d")

    ma_windows = [5, 10, 21, 50]
    ma_features = {
        f"ma_{window}_ratio": adj_close / adj_close.rolling(window).mean()
        for window in ma_windows
    }

    volatility = (
        returns.rolling(21).std() * math.sqrt(252)
    ).rename("volatility_21d")

    bollinger_window = 20
    rolling_mean = adj_close.rolling(bollinger_window).mean()
    rolling_std = adj_close.rolling(bollinger_window).std()
    bollinger = ((adj_close - rolling_mean) / rolling_std).rename("bollinger_z")

    rsi = _compute_rsi(adj_close, window=14).rename("rsi_14")

    price_range = ((high - low) / close).rename("intraday_range")
    volume_z = ((volume - volume.rolling(20).mean()) / volume.rolling(20).std()).rename(
        "volume_zscore"
    )

    feature_df = pd.concat(
        [
            returns,
            log_returns,
            *ma_features.values(),
            volatility,
            bollinger,
            rsi,
            price_range,
            volume_z,
        ],
        axis=1,
    )

    target = (returns.shift(-1) > 0).astype(int).rename("target")

    combined = pd.concat([feature_df, target], axis=1).dropna()
    feature_df = combined.drop(columns=["target"])
    target = combined["target"]

    return feature_df, target


__all__ = ["build_feature_matrix"]

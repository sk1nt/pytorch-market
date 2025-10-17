from __future__ import annotations

import numpy as np
import pandas as pd

def add_common_indicators(df, price_col='close', volume_col='total_volume'):
    """
    Adds popular technical indicators to a DataFrame for pattern analysis.
    Indicators: RSI, Bollinger Bands, ATR, Stochastic, EMA/SMA, VWAP
    Args:
        df: DataFrame with OHLCV data
        price_col: column for price (default 'close')
        volume_col: column for volume (default 'total_volume')
    Returns:
        DataFrame with new indicator columns
    """
    df = df.copy()
    # --- RSI ---
    window_rsi = 14
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window_rsi, min_periods=window_rsi).mean()
    avg_loss = loss.rolling(window_rsi, min_periods=window_rsi).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- Bollinger Bands ---
    window_bb = 20
    sma = df[price_col].rolling(window_bb).mean()
    std = df[price_col].rolling(window_bb).std()
    df['bb_upper'] = sma + 2 * std
    df['bb_lower'] = sma - 2 * std
    df['bb_middle'] = sma

    # --- ATR ---
    window_atr = 14
    high = df.get('high', df[price_col])
    low = df.get('low', df[price_col])
    close = df[price_col]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window_atr).mean()

    # --- Stochastic Oscillator ---
    window_stoch = 14
    lowest_low = low.rolling(window_stoch).min()
    highest_high = high.rolling(window_stoch).max()
    df['stoch_k'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # --- EMA/SMA ---
    df['ema_12'] = df[price_col].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df[price_col].ewm(span=26, adjust=False).mean()
    df['sma_50'] = df[price_col].rolling(50).mean()
    df['sma_200'] = df[price_col].rolling(200).mean()

    # --- VWAP ---
    if volume_col in df.columns:
        typical_price = (df.get('high', df[price_col]) + df.get('low', df[price_col]) + df[price_col]) / 3
        df['vwap'] = (typical_price * df[volume_col]).cumsum() / (df[volume_col].cumsum() + 1e-9)

    # RSI signals
    df['rsi_signal'] = np.where(df['rsi_14'] > 70, 'overbought', np.where(df['rsi_14'] < 30, 'oversold', 'neutral'))
    # Bollinger signals
    df['bb_signal'] = np.where(df[price_col] > df['bb_upper'], 'breakout_up', np.where(df[price_col] < df['bb_lower'], 'breakout_down', 'inside'))
    # ATR signals (volatility)
    atr_thresh = df['atr_14'].rolling(100).mean() * 1.5
    df['atr_signal'] = np.where(df['atr_14'] > atr_thresh, 'high_vol', 'normal_vol')
    # Stochastic signals
    df['stoch_signal'] = np.where(df['stoch_k'] > 80, 'overbought', np.where(df['stoch_k'] < 20, 'oversold', 'neutral'))
    # EMA/SMA signals
    df['ema_signal'] = np.where(df['ema_12'] > df['ema_26'], 'bullish', 'bearish')
    df['sma_signal'] = np.where(df['sma_50'] > df['sma_200'], 'bullish', 'bearish')
    # VWAP signals
    if volume_col in df.columns:
        df['vwap_signal'] = np.where(df[price_col] > df['vwap'], 'above_vwap', 'below_vwap')
    # MACD signals
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_cross'] = np.where((df['macd_hist'] > 0) & (df['macd_hist'].shift() <= 0), 'bull_cross',
                                np.where((df['macd_hist'] < 0) & (df['macd_hist'].shift() >= 0), 'bear_cross', 'none'))
    # Sentiment tagging
    def tag_sentiment(row):
        tags = []
        if row['rsi_signal'] in ['overbought', 'oversold']:
            tags.append(f"RSI:{row['rsi_signal']}")
        if row['bb_signal'] in ['breakout_up', 'breakout_down']:
            tags.append(f"BB:{row['bb_signal']}")
        if row['stoch_signal'] in ['overbought', 'oversold']:
            tags.append(f"Stoch:{row['stoch_signal']}")
        if row['ema_signal'] == 'bullish' and row['sma_signal'] == 'bullish':
            tags.append('Trend:Bull')
        if row['ema_signal'] == 'bearish' and row['sma_signal'] == 'bearish':
            tags.append('Trend:Bear')
        if row.get('macd_cross', 'none') in ['bull_cross', 'bear_cross']:
            tags.append(f"MACD:{row['macd_cross']}")
        if row.get('atr_signal', 'normal_vol') == 'high_vol':
            tags.append('Vol:High')
        if row.get('vwap_signal', None):
            tags.append(f"VWAP:{row['vwap_signal']}")
        return '|'.join(tags)
    df['sentiment_tag'] = df.apply(tag_sentiment, axis=1)
    # Price movement correlation
    price_move = df[price_col].pct_change().abs()
    move_thresh = price_move.rolling(100).mean() * 2.5
    df['large_move'] = price_move > move_thresh
    def tag_fidelity(row):
        if row['large_move']:
            if any(x in row['sentiment_tag'] for x in ['RSI:', 'BB:', 'MACD:', 'Stoch:', 'Trend:', 'Vol:High']):
                return 'A+'  # High fidelity: clear indicator trigger
            return 'B'  # Lower fidelity: ambiguous
        return ''
    df['move_fidelity'] = df.apply(tag_fidelity, axis=1)
    # Lower percentage trades
    def tag_low_pct(row):
        if row['move_fidelity'] == 'B' and row['sentiment_tag'].count('|') <= 1:
            return 'low_pct'
        return ''
    df['low_pct_trade'] = df.apply(tag_low_pct, axis=1)

    return df
"""Feature engineering utilities."""


import math
from typing import Tuple
import math
from typing import Tuple


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Return the Relative Strength Index for the provided closing prices."""

    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Allow early values so small test datasets produce usable RSI values.
    roll_up = up.rolling(window=window, min_periods=1).mean()
    roll_down = down.rolling(window=window, min_periods=1).mean()

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi



def build_feature_matrix(data: pd.DataFrame, ticker: str, extra_features: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Construct a feature matrix ``X`` and label vector ``y``.

    Parameters
    ----------
    data:
        DataFrame as returned by :func:`market_ml.data.download_price_history`.
    ticker:
        Symbol to extract from the multi-index columns produced by yfinance.
    extra_features:
        Optional DataFrame of extra features (e.g., daily sentiment), indexed by date.

    Returns
    -------
    Tuple[pandas.DataFrame, pandas.Series]
        ``X`` contains engineered features (plus any extra features). ``y`` is the binary label
        representing whether the next day's return is positive (``1``) or not (``0``).
    """

    try:
        adj_close = data["Adj Close"][ticker].rename("adj_close")
        # If the input DataFrame had columns with mismatched indices (the
        # tests sometimes build one column with a default integer index),
        # pandas.concat will produce a union index with timestamp rows plus
        # integer rows. Drop NaNs from adj_close to recover the original
        # date-based index before aligning the other series.
        adj_close = adj_close.dropna()
        # Ensure all series align with the adjusted close index. Some callers
        # (tests) construct columns with a default integer index which causes
        # an outer join and lots of NaNs; reindex to adj_close to avoid that.
        close = data["Close"][ticker].reindex(adj_close.index).rename("close")
        high = data["High"][ticker].reindex(adj_close.index).rename("high")
        low = data["Low"][ticker].reindex(adj_close.index).rename("low")
        volume = data["Volume"][ticker].reindex(adj_close.index).rename("volume")
    except KeyError as exc:  # pragma: no cover - defensive programming
        raise KeyError(
            f"Unable to locate ticker '{ticker}' in the downloaded price history."
        ) from exc

    returns = adj_close.pct_change().rename("return_1d")
    log_returns = np.log(adj_close / adj_close.shift(1)).rename("log_return_1d")

    ma_windows = [5, 10, 21, 50]
    ma_features = {
        f"ma_{window}_ratio": (
            adj_close / adj_close.rolling(window, min_periods=1).mean()
        ).rename(f"ma_{window}_ratio")
        for window in ma_windows
    }

    # Use population std (ddof=0) so small windows don't produce NaNs; set
    # min_periods=1 to allow early values on short series used in tests.
    volatility = (
        returns.rolling(21, min_periods=1).std(ddof=0) * math.sqrt(252)
    ).rename("volatility_21d")

    bollinger_window = 20
    rolling_mean = adj_close.rolling(bollinger_window, min_periods=1).mean()
    rolling_std = adj_close.rolling(bollinger_window, min_periods=1).std(ddof=0)
    bollinger = ((adj_close - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan)
    bollinger = bollinger.fillna(0).rename("bollinger_z")

    rsi = _compute_rsi(adj_close, window=14).rename("rsi_14")

    price_range = ((high - low) / close).rename("intraday_range")
    vol_roll_mean = volume.rolling(20, min_periods=1).mean()
    vol_roll_std = volume.rolling(20, min_periods=1).std(ddof=0)
    volume_z = ((volume - vol_roll_mean) / vol_roll_std).replace([np.inf, -np.inf], np.nan)
    volume_z = volume_z.fillna(0).rename("volume_zscore")

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

    # Optionally join extra features (e.g., sentiment) on the index
    if extra_features is not None:
        # Align on index (date)
        feature_df = feature_df.join(extra_features, how="left")
        feature_df = feature_df.dropna()  # Drop rows with missing extra features
        target = target.loc[feature_df.index]
    return feature_df, target


__all__ = ["build_feature_matrix"]

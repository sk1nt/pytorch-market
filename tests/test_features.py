import pandas as pd
import numpy as np

from market_ml.features import build_feature_matrix


def _make_sample_price_frame():
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    rng = np.random.default_rng(42)
    base_price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    close = pd.Series(base_price, index=dates)
    adj_close = close * 1.001  # simple offset to avoid identical series
    high = close + rng.uniform(0.2, 1.0, len(dates))
    low = close - rng.uniform(0.2, 1.0, len(dates))
    volume = 1e6 + rng.normal(0, 5e4, len(dates))

    data = pd.concat(
        {
            "Adj Close": pd.DataFrame({"TEST": adj_close}),
            "Close": pd.DataFrame({"TEST": close}),
            "High": pd.DataFrame({"TEST": high}),
            "Low": pd.DataFrame({"TEST": low}),
            "Volume": pd.DataFrame({"TEST": volume}),
        },
        axis=1,
    )
    return data


def test_feature_matrix_shapes_and_no_nans():
    prices = _make_sample_price_frame()
    X, y = build_feature_matrix(prices, "TEST")

    # After dropping NaNs there should be at least a handful of rows remaining
    assert len(X) == len(y) and len(X) > 20

    # No NaNs should remain in the engineered features or target
    assert not X.isna().any().any()
    assert not y.isna().any()

    expected_columns = {
        "return_1d",
        "log_return_1d",
        "ma_5_ratio",
        "ma_10_ratio",
        "ma_21_ratio",
        "ma_50_ratio",
        "volatility_21d",
        "bollinger_z",
        "rsi_14",
        "intraday_range",
        "volume_zscore",
    }
    assert set(X.columns) == expected_columns

    # Target should be binary 0/1
    assert set(y.unique()) <= {0, 1}

import pandas as pd
import numpy as np

from market_ml.features import build_feature_matrix


def _make_mixed_index_frame():
    # Create a normal business-day date index
    dates = pd.date_range("2021-01-01", periods=60, freq="B")
    rng = np.random.default_rng(123)
    base_price = 100 + np.cumsum(rng.normal(0, 1, len(dates)))
    close = pd.Series(base_price, index=dates)
    adj_close = close * 1.0
    high = close + rng.uniform(0.1, 1.0, len(dates))
    low = close - rng.uniform(0.1, 1.0, len(dates))
    # volume created with a default integer index to simulate mixed-index
    volume = pd.Series(1e6 + rng.normal(0, 5e4, len(dates)))

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


def test_build_feature_matrix_handles_mixed_index():
    prices = _make_mixed_index_frame()
    X, y = build_feature_matrix(prices, "TEST")

    assert len(X) == len(y) and len(X) > 20
    assert not X.isna().any().any()
    assert not y.isna().any()

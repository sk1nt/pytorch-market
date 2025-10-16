import numpy as np
import pandas as pd

from market_ml.model import train_classifier


def test_train_classifier_returns_predictions_with_expected_index():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=120, freq="B")
    X = pd.DataFrame(
        rng.normal(size=(len(dates), 5)),
        index=dates,
        columns=[f"f{i}" for i in range(5)],
    )
    y = pd.Series(rng.integers(0, 2, size=len(dates)), index=dates, name="target")

    result = train_classifier(X, y, train_ratio=0.75, random_state=1)

    assert len(result.test_predictions) == len(y) - int(len(y) * 0.75)
    assert result.test_predictions.index.equals(y.index[int(len(y) * 0.75):])
    assert set(result.feature_importances.index) == set(X.columns)


def test_train_classifier_validates_ratio():
    X = pd.DataFrame({"a": [1, 2, 3]}, index=pd.date_range("2021-01-01", periods=3))
    y = pd.Series([0, 1, 0], index=X.index)

    for invalid_ratio in (-0.1, 0.0, 1.0, 2.0):
        try:
            train_classifier(X, y, train_ratio=invalid_ratio)
        except ValueError:
            pass
        else:  # pragma: no cover - make sure the error is raised
            raise AssertionError("train_ratio validation failed")

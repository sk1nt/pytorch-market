"""Model training helpers for the trading strategy."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


@dataclass
class ModelResult:
    """Container for the trained model and diagnostics."""

    model: RandomForestClassifier
    report: str
    feature_importances: pd.Series
    test_predictions: pd.Series


def train_classifier(
    features: pd.DataFrame,
    target: pd.Series,
    train_ratio: float = 0.7,
    random_state: int = 7,
) -> ModelResult:
    """Train a Random Forest classifier using a chronological split.

    Parameters
    ----------
    features, target:
        Feature matrix and label vector produced by
        :func:`market_ml.features.build_feature_matrix`.
    train_ratio:
        Fraction of the observations to keep in the training set.  Because we
        are working with time series we perform a chronological split instead
        of a random one to avoid look-ahead bias.
    random_state:
        Seed for the ``RandomForestClassifier`` to make experiments
        reproducible.
    """

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    n_obs = len(features)
    split_idx = int(n_obs * train_ratio)

    X_train = features.iloc[:split_idx]
    y_train = target.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_test = target.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    report = classification_report(y_test, preds)

    feature_importances = pd.Series(
        model.feature_importances_, index=features.columns, name="feature_importance"
    ).sort_values(ascending=False)

    predictions = pd.Series(preds, index=y_test.index, name="prediction")

    return ModelResult(
        model=model,
        report=report,
        feature_importances=feature_importances,
        test_predictions=predictions,
    )


__all__ = ["train_classifier", "ModelResult"]

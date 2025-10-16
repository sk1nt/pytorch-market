"""Lightweight PyTorch model helpers ported from the video's repo.

This module intentionally keeps imports guarded so importing the package
does not require `torch` to be installed. Callables that require torch
will raise ImportError with a friendly message if torch is not available.
"""

from __future__ import annotations

from typing import Optional

import numpy as _np
import pandas as _pd


def _ensure_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "This function requires PyTorch. Install it with `pip install torch` "
            "or use the sklearn trainer in `market_ml.model`."
        ) from exc

    return torch, nn, optim, DataLoader, TensorDataset


def _to_tensor(x, dtype="float32"):
    import torch

    return torch.tensor(_np.asarray(x), dtype=getattr(torch, dtype))


def _binary_preds_from_logits(logits):
    import torch

    probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    return (probs > 0.5).astype(int)


def _make_series_from_preds(preds, index):
    return _pd.Series(preds, index=index, name="prediction")


class _TorchModelBase:
    """Base mixin to hold a textual repr when torch isn't available."""

    def __repr__(self) -> str:  # pragma: no cover - tiny utility
        return f"<{self.__class__.__name__} (PyTorch model)>"


def LinearModel(input_features: int):
    """Factory returning a small linear PyTorch model class instance.

    Mirrors the upstream `LinearModel` used in the video's repo.
    """
    torch, nn, *_ = _ensure_torch()

    class _Linear(nn.Module, _TorchModelBase):
        def __init__(self, input_features: int):
            super(_Linear, self).__init__()
            self.linear = nn.Linear(input_features, 1)

        def forward(self, x):
            return self.linear(x)

    return _Linear(input_features)


def NonLinearModel(input_features: int, hidden_size: int = 64):
    """Factory returning a small non-linear PyTorch model instance.

    Mirrors the `NonLinearModel` from the video's repo.
    """
    torch, nn, *_ = _ensure_torch()

    class _NonLinear(nn.Module, _TorchModelBase):
        def __init__(self, input_features: int, hidden_size: int = 64):
            super(_NonLinear, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, x):
            return self.network(x)

    return _NonLinear(input_features, hidden_size)


def train_pytorch(
    features: _pd.DataFrame,
    target: _pd.Series,
    train_ratio: float = 0.7,
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_size: int = 64,
    batch_size: int = 64,
    random_state: Optional[int] = 7,
):
    """Train a small PyTorch binary classifier and return a ModelResult-like dict.

    This function intentionally mirrors the behaviour of `train_classifier` in
    `market_ml.model` with a chronological split. It returns a simple container
    compatible with the downstream pipeline: keys `model`, `report`,
    `feature_importances`, and `test_predictions`.

    Note: PyTorch is optional for this repository. If torch is not installed
    this function will raise ImportError with a helpful message.
    """

    torch, nn, optim, DataLoader, TensorDataset = _ensure_torch()

    # Basic validation
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    rng = _np.random.default_rng(random_state)

    X = _np.asarray(features)
    y = _np.asarray(target).astype(float).reshape(-1, 1)

    n_obs = len(X)
    split_idx = int(n_obs * train_ratio)

    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    index_test = target.index[split_idx:]

    device = torch.device("cpu")

    model = NonLinearModel(X.shape[1], hidden_size)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # DataLoader
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for _epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(X_test_t)
        preds = _binary_preds_from_logits(logits)

    # Create classification report using sklearn if available
    try:
        from sklearn.metrics import classification_report

        report = classification_report(y_test.ravel().astype(int), preds)
    except Exception:
        report = "<classification_report_unavailable>"

    # PyTorch models don't expose feature importances; return zeros placeholder
    feature_importances = _pd.Series(_np.zeros(features.shape[1]), index=features.columns, name="feature_importance")

    test_predictions = _make_series_from_preds(preds, index_test)

    # Lazy import of ModelResult dataclass to avoid circular imports at package import time
    try:
        from market_ml.model import ModelResult

        return ModelResult(model=model, report=report, feature_importances=feature_importances, test_predictions=test_predictions)
    except Exception:
        # Return a simple dict if ModelResult is not importable for some reason
        return {
            "model": model,
            "report": report,
            "feature_importances": feature_importances,
            "test_predictions": test_predictions,
        }

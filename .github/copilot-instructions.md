## Purpose

Short, actionable guide for AI coding agents working on this repository. Focus on the project-specific data shapes, architectural boundaries, developer workflows, and code conventions discovered in the codebase.

## High-level architecture

- This is a small end-to-end ML trading demo. Core pieces live in `market_ml/`:
  - `data.py`: `DownloadConfig` and `download_price_history()` wrap yfinance and return a pandas DataFrame with yfinance-style multi-index columns (e.g. `('Adj Close', 'SPY')`).
  - `features.py`: `build_feature_matrix(data, ticker)` returns (X, y). X is a DataFrame of engineered features (names like `ma_5_ratio`, `rsi_14`, `volatility_21d`) and `y` is a binary Series of next-day returns (shifted by -1).
  - `model.py`: `train_classifier(X, y, train_ratio)` performs a chronological train/test split and returns a `ModelResult` dataclass (fields: `model`, `report`, `feature_importances`, `test_predictions`).
  - `backtest.py`: `run_long_only_backtest(prices, signals, trading_cost)` expects aligned Series and shifts signals by 1 bar to simulate next-day execution. Returns `BacktestResult` dataclass.
  - `main.py`: CLI that composes the pipeline. Useful example of wiring pieces together and of how outputs are consumed.

Why this structure: the project is intentionally modular to keep download / feature engineering / model training / backtest responsibilities separate and easy to reason about.

## Data & shape conventions (important)

- Market data uses yfinance layout: a DataFrame with MultiIndex columns like `('Adj Close', '<TICKER>')`. Agents should always extract series with `data['Adj Close'][ticker]` (see `features.py` and `main.py`).
- Time index must be a pandas `DatetimeIndex`. Many utilities assume chronological order and align by index.
- The modelling split is chronological (no random shuffling). Use `.iloc[:split_idx]` / `.iloc[split_idx:]` as `train_classifier` does.
- Signals are binary (0/1). `run_long_only_backtest` does `position = signals.shift(1)` to enforce execution on next bar and applies `trading_cost` against turnover (`position.diff().abs()`).

## Coding patterns & assumptions

- Many APIs return small dataclasses (`ModelResult`, `BacktestResult`) rather than raw tuples/dicts. Preserve that style for new helpers.
- Feature names are deterministic and exposed as column names; tests assert exact sets (see `tests/test_features.py`). When adding features, update tests accordingly.
- Randomness: `train_classifier` accepts `random_state` and uses deterministic classifier params. When introducing stochastic components, expose a `random_state` parameter.

## Developer workflows

- Install dependencies and run the pipeline (example from README):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --ticker SPY --start 2010-01-01 --train-ratio 0.7
```

- Run tests (pytest is used): `pytest -q` (project `tests/` uses `conftest.py` to ensure project root is importable).
- Plotting is optional (matplotlib only used if `--plot-path` provided).

## Tests & quick examples to reference

- `tests/test_features.py` builds a synthetic multi-column price DataFrame and expects `build_feature_matrix` to:
  - return features without NaNs and exact column names: `{'return_1d','log_return_1d','ma_5_ratio',...,'volume_zscore'}`
  - return binary target values {0,1}
- `tests/test_model.py` asserts `train_classifier` returns `test_predictions` indexed to the test portion of the original index and raises on invalid `train_ratio` values.
- `tests/test_backtest.py` checks `run_long_only_backtest` returns an equity curve DataFrame with columns `{'strategy_equity','buy_hold_equity','position','strategy_returns','asset_returns'}` and sensible metrics.

## Integration points & external deps

- External: `yfinance` for data, `pandas`, `numpy`, `scikit-learn` and optional `matplotlib` for plotting. Keep yfinance usage limited to `market_ml.data.download_price_history`.
- When working with price series, preserve the index alignment and prefer using the `Adj Close` column for returns and backtests.

## When making changes

- Keep splits chronological. Tests rely on deterministic indexing. If you add a new CLI option, mirror `main.py`'s argument parsing style.
- If adding features, update `tests/test_features.py` expected column set and ensure no NaNs after `dropna()`.
- Prefer returning dataclasses for multi-field results rather than loose tuples/dicts.

## If something is unclear

Point to the following files as primary examples that show expected shapes and wiring: `market_ml/data.py`, `market_ml/features.py`, `market_ml/model.py`, `market_ml/backtest.py`, and `main.py`.

---
If you'd like tweaks (more examples, stricter linting rules, or a longer-form agent guide), tell me which area to expand.

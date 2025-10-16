# Video: Let's Build a Quant Trading Strategy — Part 2 (pgUr-LzBpTo)

Source: https://www.youtube.com/watch?v=pgUr-LzBpTo

Chapters / Steps (timestamps approximate)

- 00:00 — Introduction
- 01:40 — Key Strategy Questions
- 03:44 — Machine Learning Model Recap
- 20:24 — Entry / Exit Signal
- 33:44 — Trade Sizing
- 35:14 — Constant Trade Sizing
- 59:23 — Compounding Trade Sizing
- 1:17:18 — Leverage
- 1:28:30 — Modelling Liquidation

Description summary

This video takes the ML model from Part 1 and develops a trading strategy using
the model's predictions. It covers generating entry/exit signals from model
predictions, multiple trade sizing approaches (constant vs compounding), the
use of leverage, and modelling liquidation risk when using margin.

Repository mapping (this project)

- Data download: `market_ml/data.py` (yfinance wrapper)
- Features / labels: `market_ml/features.py` (feature matrix builder)
- Model training (sklearn RF): `market_ml/model.py` — chronological split
- Optional PyTorch model added: `market_ml/torch_model.py` (ported from video repo)
- Backtesting and equity metrics: `market_ml/backtest.py`
- CLI wiring: `main.py` (now supports `--backend {sklearn,pytorch}`)

If you want, I can:

- Implement entry/exit signal generation shown in the video as a helper in `market_ml/strategy.py` and add tests.
- Add trade sizing helpers (constant / compounding) and wire them into `backtest.py` or a new `strategy.py`.
- Model liquidation simulation (per-video approach) and example notebooks.

Notes

- The video's GitHub repo is linked from the video's description: https://github.com/memlabs-research/build-a-quant-trading-strategy — I used this to port the PyTorch model classes.
- This file is a concise reference; say which of the above you'd like implemented next and I'll proceed.

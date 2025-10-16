# Quant trading strategy demo

This repository contains a small end-to-end project inspired by the
["Let's build a quant trading strategy" Reddit post](https://www.reddit.com/r/algotrading/comments/1o6e3ov/lets_build_a_quant_trading_strategy_part_1_ml/).
It downloads daily market data for a single exchange traded fund (ETF),
engineers a handful of technical features, trains a machine learning
classifier, and finally runs a simple vectorised backtest.

## Project structure

```
market_ml/
    __init__.py          # Public API surface for the helper package
    data.py              # Historical download utilities powered by yfinance
    features.py          # Feature engineering and label generation
    model.py             # Random Forest training helper
    backtest.py          # Long-only backtest implementation
main.py                  # Command line interface that orchestrates the pipeline
```

## Getting started

1. (Optional) Create and activate a virtual environment to keep dependencies isolated:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell use:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If you already have `pandas`, `scikit-learn` and `yfinance` available you
   can skip this step.

3. Run the full pipeline:

   ```bash
   python main.py --ticker SPY --start 2010-01-01 --train-ratio 0.7 \
       --equity-csv outputs/equity.csv --plot-path outputs/equity.png
   ```

   The script prints a classification report for the test period and summary
   metrics from the backtest.  The optional `--equity-csv` and `--plot-path`
   arguments allow you to persist the equity curve to disk.

### Backtesting assumptions

The backtest is intentionally minimal and assumes trades are executed at the
next day's close.  A small transaction cost is deducted whenever the strategy
changes state (from invested to cash or vice versa).  This is sufficient to
reproduce the educational content from the original tutorial but should not be
mistaken for a production-ready trading system.

## Extending the project

Some ideas for further experimentation:

- Experiment with different models (Gradient Boosting, XGBoost, neural networks).
- Add alternative data sources such as macro-economic indicators.
- Incorporate risk management overlays (volatility targeting, stop losses).
- Deploy the trained model behind an API for paper trading.

The modular layout of the codebase is intended to make these extensions
straightforward.

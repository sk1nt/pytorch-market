"""Command line entry-point that stitches together the full pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Optional

import pandas as pd

from market_ml import (
    DownloadConfig,
    build_feature_matrix,
    download_price_history,
    run_long_only_backtest,
    train_classifier,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Machine learning driven ETF trading strategy demo",
    )
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to analyse (default: SPY)")
    parser.add_argument(
        "--start",
        default="2010-01-01",
        help="Start date for the historical download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="Optional end date for the download (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of observations used for training (chronological split)",
    )
    parser.add_argument(
        "--backend",
        choices=["sklearn", "pytorch"],
        default="sklearn",
        help="Choose training backend: 'sklearn' (default) or 'pytorch'",
    )
    parser.add_argument(
        "--equity-csv",
        type=Path,
        default=None,
        help="If provided, save the equity curve from the backtest to this CSV file.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="If provided, store a PNG chart of the strategy vs. buy & hold curves.",
    )
    return parser.parse_args()


def _parse_date(date_str: Optional[str]) -> Optional[dt.date]:
    if date_str in (None, "", "None"):
        return None
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def main() -> None:
    args = _parse_args()

    start = _parse_date(args.start)
    end = _parse_date(args.end)

    if start is None:
        raise SystemExit("A valid start date is required.")

    config = DownloadConfig(tickers=[args.ticker], start=start, end=end)
    print(f"Downloading price history for {args.ticker}...")
    raw_data = download_price_history(config)

    print("Building feature matrix...")
    X, y = build_feature_matrix(raw_data, args.ticker)

    print("Training model...")
    if args.backend == "pytorch":
        try:
            from market_ml.torch_model import train_pytorch

            model_result = train_pytorch(
                X,
                y,
                train_ratio=args.train_ratio,
                epochs=20,
            )
        except ImportError as exc:
            raise SystemExit(
                "PyTorch backend requested but PyTorch is not installed. "
                "Install torch or use the default sklearn backend."
            ) from exc
    else:
        model_result = train_classifier(X, y, train_ratio=args.train_ratio)

    print("Classification report on the test set:\n")
    print(model_result.report)

    test_prices = raw_data["Adj Close"][args.ticker].loc[model_result.test_predictions.index]

    print("Running backtest...")
    backtest = run_long_only_backtest(test_prices, model_result.test_predictions)

    metrics = pd.Series(backtest.metrics)
    print("Backtest metrics:")
    print(metrics.to_string(float_format="{:.2%}".format))

    if args.equity_csv:
        args.equity_csv.parent.mkdir(parents=True, exist_ok=True)
        backtest.equity_curve.to_csv(args.equity_csv, index=True)
        print(f"Saved equity curve to {args.equity_csv}.")

    if args.plot_path:
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - plotting optional
            print("matplotlib is not installed; skipping plot generation.")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            backtest.equity_curve[["strategy_equity", "buy_hold_equity"]].plot(ax=ax)
            ax.set_title(f"{args.ticker} ML Strategy vs. Buy & Hold")
            ax.set_ylabel("Equity Curve (start = 1.0)")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            args.plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.plot_path)
            plt.close(fig)
            print(f"Saved comparison plot to {args.plot_path}.")

    print("Top 10 feature importances:")
    print(model_result.feature_importances.head(10).to_string(float_format="{:.2f}".format))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

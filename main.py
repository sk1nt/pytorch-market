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
from market_ml.data import download_binance_trades, download_binance_trades_range
from market_ml.gexbot import load_historical_gex, stream_gex_realtime, load_gex_snapshot


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
    # Add gexbot.com integration options
    parser.add_argument(
        "--gex-historical",
        action="store_true", 
        help="Load historical GEX data from gexbot.com",
    )
    parser.add_argument(
        "--gex-stream",
        action="store_true",
        help="Stream real-time GEX data from gexbot.com",
    )
    parser.add_argument(
        "--gex-snapshot",
        action="store_true",
        help="Load current GEX term structure snapshot",
    )
    parser.add_argument(
        "--gex-key",
        help="Optional gexbot.com API key",
    )
    parser.add_argument(
        "--gex-expiry",
        help="Optional expiration date for GEX data (YYYY-MM-DD)",
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
    parser.add_argument(
        "--download-btcusdt-trades",
        action="store_true",
        help="Download recent BTCUSDT trades from Binance.",
    )
    parser.add_argument(
        "--download-btcusdt-trades-6mo",
        action="store_true",
        help="Download last 6 months of BTCUSDT trades from Binance US and save to CSV.",
    )
    parser.add_argument(
        "--load-scid",
        type=str,
        default=None,
        help="Path to Sierra Chart .scid file to load and print summary."
    )
    parser.add_argument(
        "--stream-scid",
        type=str,
        default=None,
        help="Path to Sierra Chart .scid file to stream in real time."
    )
    parser.add_argument(
        "--load-sierra-trades",
        type=str,
        default=None,
        help="Path to Sierra Chart trade CSV/TXT file or directory to load."
    )
    parser.add_argument(
        "--load-sierra-depth",
        type=str,
        default=None,
        help="Path to Sierra Chart market depth CSV/TXT file or directory to load."
    )
    parser.add_argument(
        "--load-sierra-depth-bin",
        type=str,
        default=None,
        help="Path to Sierra Chart .depth binary file to load."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="If provided, save loaded data to this CSV file."
    )
    return parser.parse_args()


def _parse_date(date_str: Optional[str]) -> Optional[dt.date]:
    if date_str in (None, "", "None"):
        return None
    return dt.datetime.strptime(date_str, "%Y-%m-%d").date()


def main() -> None:
    args = _parse_args()
    from market_ml.data import (
        load_sierra_chart_scid, stream_sierra_chart_scid,
        load_sierra_chart_trades, load_sierra_chart_depth
    )

    if args.load_scid:
        print(f"Loading Sierra Chart .scid file: {args.load_scid}")
        df = load_sierra_chart_scid(args.load_scid)
        print(df.head())
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved to {args.output_csv}")
        return
    if args.stream_scid:
        print(f"Streaming Sierra Chart .scid file: {args.stream_scid}")
        for rec in stream_sierra_chart_scid(args.stream_scid):
            print(rec)
        return
    if args.load_sierra_trades:
        print(f"Loading Sierra Chart trades: {args.load_sierra_trades}")
        df = load_sierra_chart_trades(args.load_sierra_trades)
        print(df.head())
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved to {args.output_csv}")
        return
    if args.load_sierra_depth:
        print(f"Loading Sierra Chart market depth: {args.load_sierra_depth}")
        df = load_sierra_chart_depth(args.load_sierra_depth)
        print(df.head())
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
            print(f"Saved to {args.output_csv}")
        return

    # Handle gexbot.com data requests
    if args.gex_historical:
        print(f"Loading historical GEX data for {args.ticker}...")
        df = load_historical_gex(args.ticker, args.start, args.end or dt.date.today().isoformat(), args.gex_key)
        print(df.head())
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
        return

    if args.gex_stream:
        print(f"Streaming real-time GEX data for {args.ticker}...")
        for data in stream_gex_realtime([args.ticker], args.gex_key):
            print(data)
        return

    if args.gex_snapshot:
        print(f"Loading GEX snapshot for {args.ticker}...")
        df = load_gex_snapshot(args.ticker, args.gex_expiry, args.gex_key)
        print(df.head())
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
        return

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

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
from market_ml.polygon import load_aggregates, stream_trades, load_options_chain
from market_ml.strategy import (
    PositionSizingConfig,
    run_strategy_backtest,
    calculate_strategy_metrics,
)
from market_ml.social import load_demo_tweets, aggregate_daily_sentiment


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
    # Add Polygon.io options
    parser.add_argument(
        "--polygon-data",
        choices=["bars", "trades", "options"],
        help="Load data from Polygon.io: bars (OHLCV), trades (streaming), or options chain",
    )
    parser.add_argument(
        "--polygon-timespan",
        choices=["minute", "hour", "day", "week", "month"],
        default="day",
        help="Timespan for Polygon.io aggregates",
    )
    parser.add_argument(
        "--polygon-multiplier",
        type=int,
        default=1,
        help="Multiplier for timespan (e.g. 5 for 5-minute bars)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of observations used for training (chronological split)",
    )
    # Strategy options (Part 2 features)
    parser.add_argument(
        "--use-strategy",
        action="store_true",
        help="Use advanced strategy backtest with position sizing, leverage, and fees",
    )
    parser.add_argument(
        "--sizing-method",
        choices=["constant", "compounding"],
        default="constant",
        help="Position sizing method for strategy backtest",
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Leverage multiplier for strategy backtest",
    )
    parser.add_argument(
        "--maker-fee",
        type=float,
        default=0.0002,
        help="Maker fee rate (e.g., 0.0002 = 2 bps)",
    )
    parser.add_argument(
        "--taker-fee",
        type=float,
        default=0.0005,
        help="Taker fee rate (e.g., 0.0005 = 5 bps)",
    )
    parser.add_argument(
        "--liquidation-threshold",
        type=float,
        default=0.0,
        help="Liquidation threshold (0.0 disables liquidation checks)",
    )
    parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Enable shorting by mapping 0 predictions to -1 signals",
    )
    parser.add_argument(
        "--entry-liquidity",
        choices=["maker", "taker"],
        default="maker",
        help="Liquidity type for entry leg fees (maker or taker)",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000.0,
        help="Initial capital for strategy backtest",
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
    # Add social media options
    parser.add_argument(
        "--social-demo",
        action="store_true",
        help="Load demo social media data (tweets) for sentiment analysis",
    )
    parser.add_argument(
        "--social-username",
        type=str,
        default="realDonaldTrump",
        help="Twitter/X username to load tweets from (default: realDonaldTrump)",
    )
    parser.add_argument(
        "--social-days",
        type=int,
        default=30,
        help="Number of days of historical social data to load (default: 30)",
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

    # Handle social media requests
    if args.social_demo:
        print(f"Loading demo tweets from @{args.social_username}...")
        tweets = load_demo_tweets(username=args.social_username, days_back=args.social_days)
        print(f"\nLoaded {len(tweets)} tweets")
        print("\nSample tweets:")
        print(tweets[["username", "text", "retweets", "likes"]].head(5))
        
        print("\n" + "="*80)
        print("Aggregating daily sentiment...")
        daily = aggregate_daily_sentiment(tweets)
        print(daily[daily["tweet_count"] > 0])  # Only show days with tweets
        
        print("\n" + "="*80)
        print("Summary statistics:")
        print(f"Average sentiment: {daily['sentiment_score'].mean():.3f}")
        print(f"Total engagement: {daily['engagement'].sum():,.0f}")
        print(f"Most active day: {daily['tweet_count'].idxmax()}")
        print(f"Highest sentiment day: {daily['sentiment_score'].idxmax()}")
        
        if args.output_csv:
            daily.to_csv(args.output_csv)
            print(f"\nSaved daily sentiment data to {args.output_csv}")
        return

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

    # Handle Polygon.io data requests
    if args.polygon_data:
        if args.polygon_data == "bars":
            print(f"Loading {args.polygon_timespan} bars for {args.ticker} from Polygon.io...")
            df = load_aggregates(
                args.ticker,
                args.start,
                args.end or datetime.now().strftime("%Y-%m-%d"),
                multiplier=args.polygon_multiplier,
                timespan=args.polygon_timespan
            )
            print(df.head())
            if args.output_csv:
                df.to_csv(args.output_csv)
            return
            
        elif args.polygon_data == "trades":
            print(f"Streaming trades for {args.ticker} from Polygon.io...")
            for trade in stream_trades(args.ticker):
                print(trade)
            return
            
        elif args.polygon_data == "options":
            print(f"Loading options chain for {args.ticker} from Polygon.io...")
            df = load_options_chain(args.ticker)
            print(df.head())
            if args.output_csv:
                df.to_csv(args.output_csv)
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

    if args.use_strategy:
        print("Running advanced strategy backtest...")
        # Map predictions (0/1) to signals (0/1); shorting not enabled by default
        signals = model_result.test_predictions.astype(int)
        if args.allow_shorts:
            # Map 0 predictions to -1 to allow shorting
            signals = signals.replace(0, -1)
        cfg = PositionSizingConfig(
            initial_capital=args.initial_capital,
            sizing_method=args.sizing_method,
            leverage=args.leverage,
            maker_fee=args.maker_fee,
            taker_fee=args.taker_fee,
            liquidation_threshold=args.liquidation_threshold,
            entry_liquidity=args.entry_liquidity,
        )
        results = run_strategy_backtest(test_prices, signals, cfg)
        metrics = calculate_strategy_metrics(results)
        print("Strategy metrics:")
        print(pd.Series(metrics).to_string(float_format="{:.4f}".format))

        if args.equity_csv:
            args.equity_csv.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(args.equity_csv, index=True)
            print(f"Saved strategy trade log to {args.equity_csv}.")
        # Optional plotting for strategy results
        if args.plot_path:
            try:
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib is not installed; skipping plot generation.")
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                results['equity'].plot(ax=ax, label='Strategy Equity')
                ax.set_title(f"{args.ticker} Strategy Equity ({args.sizing_method}, lev={args.leverage})")
                ax.set_ylabel("Equity")
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                args.plot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(args.plot_path)
                plt.close(fig)
                print(f"Saved strategy equity plot to {args.plot_path}.")
    else:
        print("Running backtest...")
        backtest = run_long_only_backtest(test_prices, model_result.test_predictions)

        metrics = pd.Series(backtest.metrics)
        print("Backtest metrics:")
        print(metrics.to_string(float_format="{:.2%}".format))

        if args.equity_csv:
            args.equity_csv.parent.mkdir(parents=True, exist_ok=True)
            backtest.equity_curve.to_csv(args.equity_csv, index=True)
            print(f"Saved equity curve to {args.equity_csv}.")

    # Only plot legacy backtest comparison here; strategy plotting is handled in the strategy branch above
    if args.plot_path and not args.use_strategy:
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

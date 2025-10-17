# Using Social Sentiment as ML Features

You can include daily social sentiment features in your ML model to improve predictions.

## Python Example

```python
from market_ml.data import download_price_history
from market_ml.social import load_demo_tweets, aggregate_daily_sentiment
from market_ml.features import build_feature_matrix

# Download price data
data = download_price_history(...)

# Load and aggregate sentiment data
tweets = load_demo_tweets(username="realDonaldTrump", days_back=90)
sentiment = aggregate_daily_sentiment(tweets)

# Align sentiment index to match price data (if needed)
sentiment = sentiment.reindex(data.index, method='ffill')

# Build feature matrix with sentiment features
X, y = build_feature_matrix(data, "SPY", extra_features=sentiment)

# Now X includes all sentiment columns as features for ML training
print(X.columns)
```

## CLI Integration

To use sentiment features from the command line, add a flag (e.g., `--use-sentiment`) and wire the pipeline to join daily sentiment features before training:

```bash
python main.py --ticker SPY --start 2015-01-01 --use-sentiment --social-username realDonaldTrump --social-days 90
```

In `main.py`, join the sentiment DataFrame to your price data and pass it to `build_feature_matrix`:

```python
if args.use_sentiment:
   from market_ml.social import load_demo_tweets, aggregate_daily_sentiment
   tweets = load_demo_tweets(username=args.social_username, days_back=args.social_days)
   sentiment = aggregate_daily_sentiment(tweets)
   sentiment = sentiment.reindex(raw_data.index, method='ffill')
   X, y = build_feature_matrix(raw_data, args.ticker, extra_features=sentiment)
else:
   X, y = build_feature_matrix(raw_data, args.ticker)
```

All columns from the daily sentiment DataFrame (e.g., `sentiment_score`, `volume_weighted_sentiment`, `engagement`, etc.) will be included as features.
# Polygon.io and gexbot.com Usage Examples

This project supports advanced market data and options analytics via Polygon.io and GEX (gamma exposure) analytics via gexbot.com.

## Polygon.io: Options Chain and 0DTE Analysis

You can use the CLI to fetch options chain data and analyze 0DTE (zero days to expiry) contracts:

```bash
python main.py --polygon-data options --ticker SPY --output-csv outputs/spy_options.csv
```

Or, in Python:

```python
from market_ml.polygon import load_options_chain
df = load_options_chain('SPY', expiration_date='2025-10-16')
print(df.head())
```

To find the highest open interest or volume 0DTE option for a symbol:

```python
df = load_options_chain('QQQ', expiration_date='2025-10-16')
oi_option = df.loc[df['open_interest'].idxmax()]
vol_option = df.loc[df['volume'].idxmax()]
print('Highest OI:', oi_option)
print('Highest Volume:', vol_option)
```

## gexbot.com: GEX (Gamma Exposure) Data

You can retrieve GEX data for a symbol using the CLI:

```bash
export GEXBOT_API_KEY=your_key_here
python main.py --gex-historical --ticker SPX --start 2025-10-01 --output-csv outputs/spx_gex.csv
```

Or, in Python:

```python
from market_ml.gexbot import load_historical_gex
df = load_historical_gex('SPX', '2025-10-01', '2025-10-16', api_key='your_key_here', aggregation='1d')
print(df.head())
```

The GEX data includes columns like `gex`, `call_gex`, `put_gex`, and `price` for each date.

## Notes
- For Polygon.io, set your API key in the environment as `POLYGON_API_KEY` if required.
- For gexbot.com, set your API key as `GEXBOT_API_KEY` or pass it directly to the Python API.
- See `tests/test_polygon_0dte.py` and `tests/test_gexbot_historical.py` for more usage patterns and validation.
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
    polygon.py           # Polygon.io integration for market data and options
    gexbot.py            # gexbot.com integration for GEX (gamma exposure) data
    social.py            # Social media sentiment analysis (Twitter/X)
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

## Advanced strategy backtest (Part 2)

You can run an advanced strategy backtest with position sizing, leverage, and fees.

Example (constant sizing, no leverage):

```bash
python main.py --ticker SPY --start 2015-01-01 \
   --use-strategy --sizing-method constant --leverage 1.0 \
   --maker-fee 0.0002 --taker-fee 0.0005 \
   --equity-csv outputs/strategy_trades.csv
```

Example (compounding sizing, 2x leverage, liquidation check at 0%):

```bash
python main.py --ticker SPY --start 2015-01-01 \
   --use-strategy --sizing-method compounding --leverage 2.0 \
   --liquidation-threshold 0.0 \
   --entry-liquidity taker \
   --initial-capital 20000 \
   --equity-csv outputs/strategy_trades.csv
```

Notes:
- Predictions are mapped to signals {0,1} (long-or-flat). Enable shorting by feeding negative signals.
- Maker/taker fees are applied on entry/exit when positions close.
- Compounding uses current equity for sizing; constant sizing uses initial capital.
    - To enable shorting from classifier outputs, pass `--allow-shorts` to map 0 predictions to -1 (short) signals.
- Control fee model using `--entry-liquidity {maker,taker}`; exit leg uses taker by default.
- Set starting cash with `--initial-capital`.
- Use `--plot-path` alongside `--use-strategy` to save a strategy equity chart.

## Extending the project

Some ideas for further experimentation:

- Experiment with different models (Gradient Boosting, XGBoost, neural networks).
- Add alternative data sources such as macro-economic indicators.
- Incorporate risk management overlays (volatility targeting, stop losses).
- Deploy the trained model behind an API for paper trading.

The modular layout of the codebase is intended to make these extensions
straightforward.

## Social Media Sentiment Analysis

The project now includes social media sentiment analysis to help identify market-moving events and sentiment from influential accounts.

### Demo Usage

Load demo tweets and analyze sentiment:

```bash
python main.py --social-demo --social-username realDonaldTrump --social-days 30
```

Save daily sentiment data to CSV:

```bash
python main.py --social-demo --social-username realDonaldTrump --social-days 30 --output-csv sentiment.csv
```

### Features

- **Demo data loader**: Generates realistic tweet data for testing and development
- **Sentiment extraction**: Maps tweets to numeric sentiment scores (-1 to +1)
- **Daily aggregation**: Aggregates tweet-level data to daily sentiment metrics
- **Volume weighting**: Weights sentiment by engagement (retweets + likes)
- **Market keywords**: Extracts market-relevant keywords from tweets

### Output Format

The social media module returns a DataFrame with:
- `tweet_count`: Number of tweets per day
- `sentiment_score`: Average sentiment score
- `engagement`: Total engagement (retweets + likes)
- `volume_weighted_sentiment`: Sentiment weighted by engagement

### Python API

```python
from market_ml.social import load_demo_tweets, aggregate_daily_sentiment

# Load demo tweets
tweets = load_demo_tweets(username="realDonaldTrump", days_back=30)

# Aggregate to daily sentiment
daily = aggregate_daily_sentiment(tweets)
print(daily[["tweet_count", "sentiment_score", "engagement"]].head())
```

### Integration with Real APIs

The current implementation uses demo data. To integrate with real Twitter/X API or other platforms:

1. Obtain API credentials from X/Twitter Developer Portal
2. Modify `load_tweets()` in `market_ml/social.py` to use real API calls
3. Consider using libraries like `tweepy` (official API) or `snscrape` (public scraping)

Example real API integration:

```python
import tweepy

# Initialize API client
client = tweepy.Client(bearer_token="YOUR_BEARER_TOKEN")

# Fetch tweets
tweets = client.get_users_tweets(user_id, max_results=100)
```

### Combining with Market Data

Social sentiment can be aligned with market data using the pandas index:

```python
from market_ml.data import download_price_history
from market_ml.social import load_demo_tweets, aggregate_daily_sentiment

# Load market data
prices = download_price_history(...)

# Load and aggregate social data
tweets = load_demo_tweets(...)
sentiment = aggregate_daily_sentiment(tweets)

# Align both by date
combined = prices.join(sentiment, how="inner")
```

See `examples/sentiment_market_analysis.py` for a complete example that:
- Combines social sentiment with market returns
- Analyzes correlation between sentiment and next-day returns
- Groups returns by sentiment buckets (Bearish/Neutral/Bullish)
- Identifies most influential tweets

Run the example:
```bash
python examples/sentiment_market_analysis.py
```


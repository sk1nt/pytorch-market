"""
Example: Combining social media sentiment with market data for analysis.

This script demonstrates how to:
1. Load social media sentiment data
2. Load market price data
3. Align them by date
4. Analyze correlation between sentiment and market returns
"""

import pandas as pd
from market_ml.social import load_demo_tweets, aggregate_daily_sentiment
from market_ml.data import download_price_history, DownloadConfig
import datetime as dt


def combine_sentiment_and_market_data(ticker: str = "SPY", days_back: int = 30):
    """
    Combine social sentiment with market data and analyze correlations.
    
    Args:
        ticker: Market ticker to analyze
        days_back: Number of days of historical data
        
    Returns:
        DataFrame with both sentiment and market data aligned by date
    """
    print(f"Loading demo social media data...")
    tweets = load_demo_tweets(username="realDonaldTrump", days_back=days_back)
    daily_sentiment = aggregate_daily_sentiment(tweets)
    
    print(f"Loading market data for {ticker}...")
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=days_back)
    
    config = DownloadConfig(
        tickers=[ticker],
        start=start_date,
        end=end_date
    )
    market_data = download_price_history(config)
    
    # Extract prices for the ticker
    prices = market_data['Adj Close'][ticker]
    
    # Calculate daily returns
    returns = prices.pct_change()
    
    # Combine sentiment and market data
    combined = pd.DataFrame({
        'price': prices,
        'return': returns,
        'tweet_count': daily_sentiment['tweet_count'],
        'sentiment': daily_sentiment['sentiment_score'],
        'volume_weighted_sentiment': daily_sentiment['volume_weighted_sentiment'],
        'engagement': daily_sentiment['engagement'],
    })
    
    # Forward fill missing sentiment data (days without tweets)
    combined['sentiment_ffill'] = combined['sentiment'].ffill()
    combined['vw_sentiment_ffill'] = combined['volume_weighted_sentiment'].ffill()
    
    return combined


def analyze_sentiment_market_relationship(df: pd.DataFrame):
    """
    Analyze relationship between sentiment and market returns.
    
    Args:
        df: DataFrame with sentiment and return columns
    """
    print("\n" + "="*80)
    print("SENTIMENT-MARKET ANALYSIS")
    print("="*80)
    
    # Filter to days with tweets
    with_tweets = df[df['tweet_count'] > 0].copy()
    
    print(f"\nDays with tweets: {len(with_tweets)}")
    print(f"Days without tweets: {len(df) - len(with_tweets)}")
    
    if len(with_tweets) > 1:
        print("\nCorrelation between sentiment and next-day returns:")
        # Shift returns forward to see if sentiment predicts next day
        with_tweets['next_day_return'] = with_tweets['return'].shift(-1)
        
        corr_raw = with_tweets['sentiment'].corr(with_tweets['next_day_return'])
        corr_vw = with_tweets['volume_weighted_sentiment'].corr(with_tweets['next_day_return'])
        
        print(f"  Raw sentiment correlation: {corr_raw:.4f}")
        print(f"  Volume-weighted sentiment correlation: {corr_vw:.4f}")
        
        print("\nAverage next-day return by sentiment:")
        with_tweets['sentiment_bucket'] = pd.cut(
            with_tweets['sentiment'],
            bins=[-1.1, -0.3, 0.3, 1.1],
            labels=['Bearish', 'Neutral', 'Bullish']
        )
        
        sentiment_returns = with_tweets.groupby('sentiment_bucket', observed=True)['next_day_return'].agg(['mean', 'count'])
        print(sentiment_returns)
        
        print("\nTop 3 most bullish tweets and subsequent returns:")
        top_bullish = with_tweets.nlargest(3, 'sentiment')[['sentiment', 'engagement', 'next_day_return']]
        print(top_bullish)
        
        print("\nTop 3 most bearish tweets and subsequent returns:")
        top_bearish = with_tweets.nsmallest(3, 'sentiment')[['sentiment', 'engagement', 'next_day_return']]
        print(top_bearish)


if __name__ == "__main__":
    # Combine data
    combined_df = combine_sentiment_and_market_data(ticker="SPY", days_back=30)
    
    print("\nCombined dataset preview:")
    print(combined_df[combined_df['tweet_count'] > 0].head())
    
    # Analyze relationships
    analyze_sentiment_market_relationship(combined_df)
    
    # Save to CSV for further analysis
    output_file = "sentiment_market_combined.csv"
    combined_df.to_csv(output_file)
    print(f"\nSaved combined data to {output_file}")

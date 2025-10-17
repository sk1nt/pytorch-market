"""
Social media data loading for market sentiment analysis.

Supports loading tweets and posts from influential accounts on X/Twitter and other platforms.
Provides demo data for testing and development.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SocialConfig:
    """Configuration for social media data loading."""
    platform: str = "twitter"  # twitter, truth_social, etc.
    username: Optional[str] = None
    keywords: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    max_results: int = 100


def load_demo_tweets(username: str = "realDonaldTrump", days_back: int = 30) -> pd.DataFrame:
    """
    Load demo tweet data for testing and development.
    
    In production, this would connect to X/Twitter API or use snscrape.
    For now, returns synthetic demo data with realistic structure.
    
    Args:
        username: Twitter handle (without @)
        days_back: Number of days of historical data to generate
        
    Returns:
        DataFrame indexed by timestamp with columns:
        - username: Twitter handle
        - text: Tweet content
        - retweets: Number of retweets
        - likes: Number of likes
        - replies: Number of replies
        - sentiment_keywords: List of market-relevant keywords detected
    """
    # Generate demo data with market-relevant tweets
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Demo tweets with market impact themes
    demo_tweets = [
        {
            "text": "Stock market looking very strong! Record highs coming. #MAGA",
            "sentiment": "bullish",
            "retweets": 15000,
            "likes": 85000,
            "replies": 3200,
            "keywords": ["stock market", "strong", "record highs"]
        },
        {
            "text": "The Fed should cut rates NOW! Inflation is under control.",
            "sentiment": "dovish",
            "retweets": 22000,
            "likes": 120000,
            "replies": 8500,
            "keywords": ["Fed", "cut rates", "inflation"]
        },
        {
            "text": "China trade deal looking very promising. Great for American workers!",
            "sentiment": "bullish",
            "retweets": 18000,
            "likes": 95000,
            "replies": 4100,
            "keywords": ["China", "trade deal", "American workers"]
        },
        {
            "text": "Big Tech needs regulation. Too much power concentrated in a few companies.",
            "sentiment": "bearish_tech",
            "retweets": 28000,
            "likes": 150000,
            "replies": 12000,
            "keywords": ["Big Tech", "regulation", "power"]
        },
        {
            "text": "Cryptocurrency is a SCAM! Protect your investments in REAL assets.",
            "sentiment": "bearish_crypto",
            "retweets": 35000,
            "likes": 180000,
            "replies": 25000,
            "keywords": ["cryptocurrency", "scam", "investments"]
        },
        {
            "text": "Energy independence is crucial. Drill baby drill! Oil and gas stocks to the moon!",
            "sentiment": "bullish_energy",
            "retweets": 20000,
            "likes": 110000,
            "replies": 5500,
            "keywords": ["energy independence", "oil", "gas", "stocks"]
        },
        {
            "text": "Manufacturing jobs returning to America. Economy booming like never before!",
            "sentiment": "bullish",
            "retweets": 25000,
            "likes": 140000,
            "replies": 6200,
            "keywords": ["manufacturing", "jobs", "economy", "booming"]
        },
        {
            "text": "Interest rates are TOO HIGH. Hurting homeowners and businesses. Fed wake up!",
            "sentiment": "dovish",
            "retweets": 19000,
            "likes": 105000,
            "replies": 7800,
            "keywords": ["interest rates", "high", "homeowners", "businesses", "Fed"]
        },
    ]
    
    # Generate timestamps spread across the date range
    timestamps = pd.date_range(start=start_date, end=end_date, periods=len(demo_tweets))
    
    # Build DataFrame
    records = []
    for i, (ts, tweet_data) in enumerate(zip(timestamps, demo_tweets)):
        records.append({
            "timestamp": ts,
            "username": username,
            "text": tweet_data["text"],
            "retweets": tweet_data["retweets"] + (i * 100),  # Add some variance
            "likes": tweet_data["likes"] + (i * 500),
            "replies": tweet_data["replies"] + (i * 50),
            "sentiment_label": tweet_data["sentiment"],
            "sentiment_keywords": ", ".join(tweet_data["keywords"]),
        })
    
    df = pd.DataFrame(records)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    
    return df


def load_tweets(config: SocialConfig) -> pd.DataFrame:
    """
    Load tweets from X/Twitter API or scraping service.
    
    Currently returns demo data. In production, would integrate with:
    - Official X/Twitter API (requires credentials)
    - snscrape for public scraping
    - Third-party aggregation services
    
    Args:
        config: SocialConfig with platform, username, keywords, date range
        
    Returns:
        DataFrame indexed by timestamp with tweet data
    """
    if config.username:
        # For demo, return synthetic data
        days_back = 30
        if config.start_date and config.end_date:
            days_back = (config.end_date - config.start_date).days
        return load_demo_tweets(username=config.username, days_back=days_back)
    else:
        # Could support keyword search here
        return pd.DataFrame()


def extract_market_sentiment(df: pd.DataFrame) -> pd.Series:
    """
    Extract market sentiment score from tweets.
    
    Simple implementation: maps sentiment labels to numeric scores.
    More sophisticated version would use NLP/transformers.
    
    Args:
        df: DataFrame with sentiment_label column
        
    Returns:
        Series with sentiment scores (-1 to +1)
    """
    sentiment_map = {
        "bullish": 1.0,
        "bullish_energy": 0.8,
        "neutral": 0.0,
        "dovish": 0.3,  # Rate cuts generally positive for stocks
        "bearish_tech": -0.5,
        "bearish_crypto": -0.8,
        "bearish": -1.0,
    }
    
    return df["sentiment_label"].map(sentiment_map).fillna(0.0)


def aggregate_daily_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate tweet-level data to daily sentiment scores.
    
    Useful for aligning with daily market data.
    
    Args:
        df: DataFrame with timestamp index and sentiment data
        
    Returns:
        DataFrame with daily aggregated metrics:
        - tweet_count: Number of tweets per day
        - avg_sentiment: Average sentiment score
        - engagement: Total engagement (retweets + likes)
        - volume_weighted_sentiment: Sentiment weighted by engagement
    """
    df = df.copy()
    df["sentiment_score"] = extract_market_sentiment(df)
    df["engagement"] = df["retweets"] + df["likes"]
    
    daily = df.resample("D").agg({
        "text": "count",
        "sentiment_score": "mean",
        "engagement": "sum",
        "retweets": "sum",
        "likes": "sum",
        "replies": "sum",
    })
    
    daily.rename(columns={"text": "tweet_count"}, inplace=True)
    
    # Calculate volume-weighted sentiment
    df["date"] = df.index.date
    daily_weighted = df.groupby("date").apply(
        lambda x: (x["sentiment_score"] * x["engagement"]).sum() / x["engagement"].sum()
        if x["engagement"].sum() > 0 else 0,
        include_groups=False
    )
    
    # Align with daily index
    daily["volume_weighted_sentiment"] = 0.0
    for date, value in daily_weighted.items():
        daily.loc[pd.Timestamp(date), "volume_weighted_sentiment"] = value
    
    return daily


if __name__ == "__main__":
    # Demo usage
    print("Loading demo tweets from @realDonaldTrump...")
    tweets = load_demo_tweets(username="realDonaldTrump", days_back=30)
    print(f"\nLoaded {len(tweets)} tweets")
    print("\nSample tweets:")
    print(tweets[["username", "text", "retweets", "likes"]].head(3))
    
    print("\n" + "="*80)
    print("Daily sentiment aggregation:")
    daily = aggregate_daily_sentiment(tweets)
    print(daily.head(10))
    
    print("\n" + "="*80)
    print("Summary statistics:")
    print(f"Average sentiment: {daily['sentiment_score'].mean():.3f}")
    print(f"Total engagement: {daily['engagement'].sum():,.0f}")
    print(f"Most active day: {daily['tweet_count'].idxmax()}")
    print(f"Highest sentiment day: {daily['sentiment_score'].idxmax()}")

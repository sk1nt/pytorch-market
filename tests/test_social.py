"""Tests for social media data loading."""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from market_ml.social import (
    load_demo_tweets,
    load_tweets,
    SocialConfig,
    extract_market_sentiment,
    aggregate_daily_sentiment,
)


def test_load_demo_tweets():
    """Test loading demo tweet data."""
    tweets = load_demo_tweets(username="realDonaldTrump", days_back=30)
    
    # Check structure
    assert isinstance(tweets, pd.DataFrame)
    assert isinstance(tweets.index, pd.DatetimeIndex)
    
    # Check columns
    expected_cols = {"username", "text", "retweets", "likes", "replies", 
                     "sentiment_label", "sentiment_keywords"}
    assert set(tweets.columns) == expected_cols
    
    # Check data
    assert len(tweets) > 0
    assert tweets["username"].iloc[0] == "realDonaldTrump"
    assert tweets["retweets"].min() >= 0
    assert tweets["likes"].min() >= 0
    
    # Check chronological order
    assert tweets.index.is_monotonic_increasing


def test_load_tweets_with_config():
    """Test loading tweets via config."""
    config = SocialConfig(
        platform="twitter",
        username="realDonaldTrump",
        start_date=datetime(2025, 9, 1),
        end_date=datetime(2025, 10, 1),
    )
    
    tweets = load_tweets(config)
    assert isinstance(tweets, pd.DataFrame)
    assert len(tweets) > 0


def test_extract_market_sentiment():
    """Test sentiment extraction."""
    tweets = load_demo_tweets(username="test", days_back=7)
    sentiment = extract_market_sentiment(tweets)
    
    assert isinstance(sentiment, pd.Series)
    assert len(sentiment) == len(tweets)
    
    # Check sentiment range
    assert sentiment.min() >= -1.0
    assert sentiment.max() <= 1.0


def test_aggregate_daily_sentiment():
    """Test daily sentiment aggregation."""
    tweets = load_demo_tweets(username="test", days_back=30)
    daily = aggregate_daily_sentiment(tweets)
    
    # Check structure
    assert isinstance(daily, pd.DataFrame)
    assert isinstance(daily.index, pd.DatetimeIndex)
    
    # Check columns
    expected_cols = {"tweet_count", "sentiment_score", "engagement", 
                     "retweets", "likes", "replies", "volume_weighted_sentiment"}
    assert set(daily.columns) == expected_cols
    
    # Check aggregation
    assert daily["tweet_count"].sum() == len(tweets)
    assert daily["engagement"].sum() > 0
    
    # Check sentiment bounds
    assert daily["sentiment_score"].min() >= -1.0
    assert daily["sentiment_score"].max() <= 1.0


def test_sentiment_labels():
    """Test sentiment label mappings."""
    # Create test data with known sentiments
    test_data = pd.DataFrame({
        "sentiment_label": ["bullish", "bearish", "neutral", "dovish"],
        "text": ["test"] * 4,
    })
    
    sentiment = extract_market_sentiment(test_data)
    
    assert sentiment.iloc[0] == 1.0  # bullish
    assert sentiment.iloc[1] == -1.0  # bearish
    assert sentiment.iloc[2] == 0.0  # neutral
    assert sentiment.iloc[3] == 0.3  # dovish


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

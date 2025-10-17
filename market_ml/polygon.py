"""
Polygon.io data integration module. Provides functions for loading market data
including trades, quotes, aggregates, and options data from polygon.io.

API documentation: https://polygon.io/docs
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Generator, List, Optional, Union

import pandas as pd
import requests
import websocket
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# API Constants
BASE_URL = "https://api.polygon.io"
WS_URL = "wss://socket.polygon.io/stocks"
API_KEY = "Cn3WdnnY_DY771lpA_R0XD9lKzS4mmtY"

class PolygonError(Exception):
    """Base exception for Polygon API errors."""
    pass

def _make_request(endpoint: str, params: dict = None) -> dict:
    """Make an HTTP request to the Polygon API."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    url = f"{BASE_URL}/{endpoint}"
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise PolygonError(f"API request failed: {str(e)}") from e

def load_aggregates(
    symbol: str,
    start_date: str,
    end_date: str,
    multiplier: int = 1,
    timespan: str = "day",
    adjusted: bool = True
) -> pd.DataFrame:
    """
    Load aggregated price data (OHLCV bars) for a given symbol and date range.
    
    Args:
        symbol: Stock/crypto/forex symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        multiplier: Size of the timespan multiplier (e.g. 1, 5, 15)
        timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
        adjusted: Whether to return adjusted data
    
    Returns:
        DataFrame with columns:
        - timestamp: datetime index
        - open: Opening price
        - high: High price
        - low: Low price
        - close: Close price
        - volume: Trading volume
        - vwap: Volume weighted average price
        - transactions: Number of transactions
    """
    params = {
        "adjusted": str(adjusted).lower(),
        "sort": "asc",
        "limit": 50000
    }
    
    endpoint = f"v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
    
    try:
        data = _make_request(endpoint, params=params)
        if data["resultsCount"] == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(data["results"])
        if not df.empty:
            # Map Polygon field names to standard OHLCV names
            df = df.rename(columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
                "vw": "vwap",
                "n": "transactions"
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp").sort_index()
        return df
    except Exception as e:
        logger.error(f"Failed to load aggregate data: {str(e)}")
        raise PolygonError(f"Aggregate data load failed: {str(e)}") from e

def stream_trades(
    symbols: Union[str, List[str]],
    callback = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream real-time trades from Polygon's websocket API.
    
    Args:
        symbols: Symbol or list of symbols to stream
        callback: Optional callback function for websocket events
    
    Yields:
        Dict with fields:
        - timestamp: datetime
        - symbol: Stock symbol
        - price: Trade price
        - size: Trade size
        - exchange: Exchange ID
        - conditions: Trade condition codes
        Additional fields depending on the subscription level.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
        
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if data[0]["ev"] == "T":  # Trade events
                trade = data[0]
                formatted = {
                    "timestamp": pd.to_datetime(trade["t"], unit="ms"),
                    "symbol": trade["sym"],
                    "price": float(trade["p"]),
                    "size": int(trade["s"]),
                    "exchange": trade["x"],
                    "conditions": trade.get("c", [])
                }
                if callback:
                    callback(formatted)
                else:
                    yield formatted
        except Exception as e:
            logger.error(f"Error processing websocket message: {str(e)}")

    def on_error(ws, error):
        logger.error(f"Websocket error: {str(error)}")
        
    def on_close(ws, close_status_code, close_msg):
        logger.info("Websocket connection closed")
        
    def on_open(ws):
        auth_msg = {"action": "auth", "params": API_KEY}
        ws.send(json.dumps(auth_msg))
        
        subscribe_msg = {
            "action": "subscribe",
            "params": [f"T.{sym}" for sym in symbols]
        }
        ws.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to symbols: {symbols}")

    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(
        WS_URL,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    
    ws.run_forever()

def load_options_chain(
    underlying: str,
    expiration_date: Optional[str] = None,
    strike_price: Optional[float] = None,
    right: Optional[str] = None,
    limit: int = 100
) -> pd.DataFrame:
    """
    Load current options chain data for an underlying symbol.
    
    Args:
        underlying: Underlying stock symbol
        expiration_date: Optional expiration date filter (YYYY-MM-DD)
        strike_price: Optional strike price filter
        right: Optional contract right (call, put)
        limit: Max number of results to return
    
    Returns:
        DataFrame with options contract data:
        - symbol: Option contract symbol
        - expiry: Expiration date
        - strike: Strike price
        - right: Call/Put
        - last_price: Last trade price
        - bid: Bid price
        - ask: Ask price
        - volume: Trading volume
        - open_interest: Open interest
        - implied_volatility: IV if available
    """
    params = {
        "underlying_ticker": underlying,
        "limit": limit
    }
    
    if expiration_date:
        params["expiration_date"] = expiration_date
    if strike_price:
        params["strike_price"] = strike_price
    if right:
        params["right"] = right.lower()
        
    try:
        data = _make_request("v3/reference/options/contracts", params=params)
        if not data.get("results"):
            return pd.DataFrame()

        df = pd.json_normalize(data["results"])
        if df.empty:
            return df

        rename_map = {
            "expiration_date": "expiry",
            "day.open_interest": "open_interest",
            "day.volume": "volume",
            "last_trade.price": "last_price",
            "last_quote.bid": "bid",
            "last_quote.ask": "ask",
            "last_quote.midpoint": "midpoint",
            "Greeks.implied_volatility": "implied_volatility",
            "greeks.implied_volatility": "implied_volatility",
        }
        available_renames = {
            original: new_name
            for original, new_name in rename_map.items()
            if original in df.columns
        }
        df = df.rename(columns=available_renames)

        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"])
        elif "expiration_date" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiration_date"])

        numeric_defaults = {
            "open_interest": 0.0,
            "volume": 0.0,
            "last_price": 0.0,
            "bid": 0.0,
            "ask": 0.0,
            "midpoint": 0.0,
            "implied_volatility": pd.NA,
        }
        for column, default in numeric_defaults.items():
            if column not in df.columns:
                df[column] = default
            df[column] = pd.to_numeric(df[column], errors="coerce")
            # Only fillna for columns where default is a float (not pd.NA)
            if column != "implied_volatility":
                df[column] = df[column].fillna(0.0)

        df = df.sort_values(["expiry", "strike_price"]).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load options chain: {str(e)}")
        raise PolygonError(f"Options chain load failed: {str(e)}") from e

def get_market_status() -> Dict[str, Any]:
    """Get current market status and trading hours."""
    try:
        return _make_request("v1/marketstatus/now")
    except Exception as e:
        logger.error(f"Failed to get market status: {str(e)}")
        raise PolygonError(f"Market status check failed: {str(e)}") from e

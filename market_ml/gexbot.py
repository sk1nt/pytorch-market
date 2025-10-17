"""
Gexbot.com data integration module. Provides functions for loading historical and real-time
data from gexbot.com, including GEX (gamma exposure), CHARM (option decay), and other metrics.

API documentation: https://www.gexbot.com/apidocs

NOTE: The gexbot.com API structure is not fully documented publicly. 
The implementations below are placeholders based on typical GEX API patterns.
To use this module, you need to:
1. Contact gexbot.com support for API documentation
2. Update the endpoints and parameters below to match the actual API
3. Verify authentication method (Bearer token vs other)
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Optional, Union, Any

import pandas as pd
import requests
import websocket

logger = logging.getLogger(__name__)

# API Constants
BASE_URL = "https://api.gexbot.com"
WS_URL = "wss://stream.gexbot.com/ws"

# Known working endpoint
def get_available_tickers() -> Dict[str, List[str]]:
    """Get list of available ticker symbols from gexbot.com API.
    
    Returns:
        Dict with keys: 'stocks', 'indexes', 'futures'
        Each containing a list of available symbols.
    """
    try:
        response = requests.get(f"{BASE_URL}/tickers", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get available tickers: {str(e)}")
        raise GexbotError(f"Tickers request failed: {str(e)}") from e

class GexbotError(Exception):
    """Base exception for Gexbot API errors."""
    pass

def _make_request(url: str, api_key: str = None) -> dict:
    """Make an HTTP request to the Gexbot API.
    
    Args:
        url: Full URL with query parameters
        api_key: API key (passed as query parameter, not header)
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise GexbotError(f"API request failed: {str(e)}") from e

def load_historical_gex(
    symbol: str, 
    start_date: str, 
    end_date: str, 
    api_key: str,
    aggregation: str = "1d"
) -> pd.DataFrame:
    """
    Load historical GEX data for a given symbol and date range.
    
    API Format: https://api.gexbot.com/{TICKER}/classic/{AGGREGATION_PERIOD}?key={YOUR_API_KEY}
    
    Args:
        symbol: Stock/index symbol (e.g. 'SPX', 'SPY', 'NQ_NDX' for futures)
        start_date: Start date in YYYY-MM-DD format (not used by API, returns all available)
        end_date: End date in YYYY-MM-DD format (not used by API, returns all available)
        api_key: Required gexbot.com API key
        aggregation: Data aggregation period ('1m','5m','15m','30m','1h','4h','1d')
    
    Returns:
        DataFrame with columns:
        - timestamp: datetime index
        - price: Current price
        - gex: Total gamma exposure
        - call_gex: Call gamma exposure 
        - put_gex: Put gamma exposure
        - Additional Greeks depending on API tier
    """
    if not api_key:
        raise GexbotError("API key is required for gexbot.com")
    
    url = f"{BASE_URL}/{symbol}/classic/{aggregation}?key={api_key}"
    
    try:
        data = _make_request(url, api_key=api_key)
        
        # The API returns data in a specific format - adjust based on actual response
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame([data])
            
        if df.empty:
            return df
            
        # Convert timestamp field to datetime if present
        timestamp_fields = ['timestamp', 'time', 'date', 'datetime', 't']
        for ts_field in timestamp_fields:
            if ts_field in df.columns:
                df['timestamp'] = pd.to_datetime(df[ts_field], unit='ms', errors='coerce')
                if df['timestamp'].isna().all():
                    df['timestamp'] = pd.to_datetime(df[ts_field], errors='coerce')
                df = df.set_index('timestamp').sort_index()
                break
        
        # Filter by date range if we have a datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df.loc[start_dt:end_dt]
            
        return df
    except Exception as e:
        logger.error(f"Failed to load historical GEX data: {str(e)}")
        raise GexbotError(f"Historical data load failed: {str(e)}") from e

def stream_gex_realtime(
    symbols: Union[str, List[str]],
    callback = None,
    api_key: Optional[str] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream real-time GEX and options data from gexbot.com websocket API.
    
    Args:
        symbols: Symbol or list of symbols to stream
        callback: Optional callback function for websocket events
        api_key: Optional gexbot.com API key
    
    Yields:
        Dict with fields:
        - timestamp: datetime 
        - symbol: Stock/index symbol
        - price: Current price
        - gex: Total gamma exposure
        - call_gex: Call gamma exposure
        - put_gex: Put gamma exposure
        - dealer_delta: Dealer net delta exposure
        - dealer_gamma: Dealer gamma exposure
        - charm: Total charm (delta decay)
        - vanna: Price-vol correlation sensitivity
        Additional fields with premium API access.
    """
    if isinstance(symbols, str):
        symbols = [symbols]
        
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if callback:
                callback(data)
            else:
                # Convert timestamp and yield the data
                data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
                yield data
        except Exception as e:
            logger.error(f"Error processing websocket message: {str(e)}")

    def on_error(ws, error):
        logger.error(f"Websocket error: {str(error)}")
        
    def on_close(ws, close_status_code, close_msg):
        logger.info("Websocket connection closed")
        
    def on_open(ws):
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": symbols,
            "id": 1
        }
        if api_key:
            subscribe_msg["api_key"] = api_key
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

def load_gex_snapshot(
    symbol: str,
    expiry: Optional[str] = None,
    api_key: Optional[str] = None,
    greek_fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load current GEX term structure snapshot for a symbol.
    
    Args:
        symbol: Stock/index symbol
        expiry: Optional specific expiration date in YYYY-MM-DD format
        api_key: Optional gexbot.com API key
        greek_fields: Optional list of additional Greeks to include
            Available fields: ['delta','gamma','vega','theta','rho',
                             'charm','vanna','volga']
    
    Returns:
        DataFrame with GEX data across strikes/expiries:
        - expiry: Option expiration date 
        - strike: Strike price
        - gex: Gamma exposure at strike
        - volume: Trading volume
        - open_interest: Open interest
        - iv: Implied volatility
        - delta: Option delta (if requested)
        - gamma: Option gamma (if requested)
        Additional Greeks included if specified.
    """
    params = {"symbol": symbol}
    if expiry:
        params["expiry"] = expiry
    if greek_fields:
        params["greeks"] = ",".join(greek_fields)
        
    try:
        data = _make_request("v1/snapshot", params=params, api_key=api_key)
        df = pd.DataFrame(data["data"])
        if not df.empty:
            df["expiry"] = pd.to_datetime(df["expiry"])
            df = df.sort_values(["expiry", "strike"]).reset_index(drop=True)
        return df
    except Exception as e:
        logger.error(f"Failed to load GEX snapshot: {str(e)}")
        raise GexbotError(f"Snapshot load failed: {str(e)}") from e

def get_gex_status() -> Dict[str, Any]:
    """Get current API and data status."""
    try:
        return _make_request("v1/status")
    except Exception as e:
        logger.error(f"Failed to get API status: {str(e)}")
        raise GexbotError(f"Status check failed: {str(e)}") from e
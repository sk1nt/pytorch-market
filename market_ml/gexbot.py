# --- Gamma Level Utilities ---
def extract_gamma_levels(df_or_dict):
    """
    Given a DataFrame (historical) or dict (streaming), extract major positive/negative gamma levels and zero gamma.
    Returns dict with keys: 'major_pos', 'major_neg', 'zero_gamma', 'net_gamma', 'max_gamma_change'.
    """
    import numpy as np
    if isinstance(df_or_dict, pd.DataFrame):
        df = df_or_dict
        # Case 1: legacy data with 'gex' column
        if 'gex' in df.columns:
            major_pos = df['gex'].nlargest(2)
            major_neg = df['gex'].nsmallest(2)
            zero_gamma_idx = (df['gex'].abs()).idxmin()
            net_gamma = df['gex'].sum()
            max_gamma_change = df['gex'].diff().abs().max()
            return {
                'major_pos': major_pos,
                'major_neg': major_neg,
                'zero_gamma': zero_gamma_idx,
                'net_gamma': net_gamma,
                'max_gamma_change': max_gamma_change,
            }
        # Case 2: classic schema with 'zero_gamma' and aggregates
        out = {}
        if 'zero_gamma' in df.columns:
            out['zero_gamma'] = df['zero_gamma'].dropna().iloc[-1] if df['zero_gamma'].notna().any() else None
        # Prefer OI aggregates if present
        if 'sum_gex_oi' in df.columns and pd.api.types.is_numeric_dtype(df['sum_gex_oi']):
            out['net_gamma'] = df['sum_gex_oi'].sum()
        elif 'sum_gex_vol' in df.columns and pd.api.types.is_numeric_dtype(df['sum_gex_vol']):
            out['net_gamma'] = df['sum_gex_vol'].sum()
        # Expose majors by call/put (prefer OI over VOL)
        def _latest_nonnull(col):
            return df[col].dropna().iloc[-1] if col in df.columns and df[col].notna().any() else None
        out['major_call_oi'] = _latest_nonnull('major_call_oi')
        out['major_put_oi'] = _latest_nonnull('major_put_oi')
        out['major_call_vol'] = _latest_nonnull('major_call_vol')
        out['major_put_vol'] = _latest_nonnull('major_put_vol')
        # Back-compat placeholders
        out['major_pos'] = out['major_call_oi'] if out['major_call_oi'] is not None else out['major_call_vol']
        out['major_neg'] = out['major_put_oi'] if out['major_put_oi'] is not None else out['major_put_vol']
        out['max_gamma_change'] = None
        return out
    elif isinstance(df_or_dict, dict):
        # For streaming, handle both legacy and classic-style payloads
        out = {}
        if 'gex' in df_or_dict:
            gex = df_or_dict.get('gex', 0)
            call_gex = df_or_dict.get('call_gex', 0)
            put_gex = df_or_dict.get('put_gex', 0)
            out.update({
                'major_pos': call_gex,
                'major_neg': put_gex,
                'zero_gamma': 0 if abs(call_gex + put_gex) < 1e-6 else None,
                'net_gamma': gex,
                'max_gamma_change': None,
            })
        else:
            # classic (both zero and major levels)
            out['zero_gamma'] = df_or_dict.get('zero_gamma')
            # Use sum_gex_oi/net_gex_oi if available as proxy for net
            out['net_gamma'] = df_or_dict.get('sum_gex_oi') or df_or_dict.get('net_gex_oi') or df_or_dict.get('sum_gex_vol') or df_or_dict.get('net_gex_vol')
            # Major levels: prefer mpos_*/mneg_* (major levels schema) or major_pos_*/major_neg_* (zero schema)
            out['major_call_oi'] = df_or_dict.get('mpos_oi') or df_or_dict.get('major_pos_oi')
            out['major_call_vol'] = df_or_dict.get('mpos_vol') or df_or_dict.get('major_pos_vol')
            out['major_put_oi'] = df_or_dict.get('mneg_oi') or df_or_dict.get('major_neg_oi')
            out['major_put_vol'] = df_or_dict.get('mneg_vol') or df_or_dict.get('major_neg_vol')
            # Back-compat
            out['major_pos'] = out['major_call_oi'] if out['major_call_oi'] is not None else out['major_call_vol']
            out['major_neg'] = out['major_put_oi'] if out['major_put_oi'] is not None else out['major_put_vol']
            out['max_gamma_change'] = None
        return out
    return {}

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

def probe_gexbot_key(api_key: str) -> Dict[str, Any]:
    """Quickly validate the provided API key by attempting a lightweight request.

    Returns a dict: {"valid": bool, "detail": str}
    """
    if not api_key:
        return {"valid": False, "detail": "Missing API key"}
    try:
        # Attempt classic/zero which is accessible on free tier
        _make_request("SPX/classic/zero", api_key=api_key)
        return {"valid": True, "detail": "API key valid (classic/zero accessible)"}
    except GexbotError as e:
        msg = str(e)
        if "invalid api key" in msg.lower():
            return {"valid": False, "detail": "Invalid API Key"}
        return {"valid": False, "detail": msg}

def _make_request(path_or_url: str, api_key: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> dict:
    """Make an HTTP request to the Gexbot API.

    Accepts either a full URL or a path relative to BASE_URL. If an api_key is provided
    and not already present in the query string, it will be injected as the 'key' parameter.

    Args:
        path_or_url: Full URL or relative path (e.g., 'v1/status' or 'SPX/classic/1d')
        api_key: API key (passed as query parameter)
        params: Optional query string parameters to include
    """
    try:
        # Build URL
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            url = path_or_url
        else:
            url = f"{BASE_URL}/{path_or_url.lstrip('/')}"

        # Build params and inject key if needed
        params = dict(params or {})
        if api_key and "key=" not in url and "key" not in params:
            params["key"] = api_key

        response = requests.get(url, params=params or None, timeout=10)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as he:
            body = response.text[:1000] if response is not None else ""
            status = response.status_code if response is not None else "?"
            raise GexbotError(f"API request failed [{status}] for {url}: {he}\nBody: {body}") from he

        return response.json()
    except requests.exceptions.RequestException as e:
        raise GexbotError(f"API request failed: {str(e)}") from e

def _parse_epoch_timestamp(ts: Union[int, float, str]) -> Optional[pd.Timestamp]:
    """Parse epoch timestamp expressed in seconds or milliseconds.

    Returns pandas Timestamp or None if unparseable.
    """
    try:
        # Normalize to int
        if isinstance(ts, str):
            ts = float(ts)
        tsf = float(ts)
        # Heuristic: ms if very large
        if tsf > 1e12:
            return pd.to_datetime(tsf, unit='ms', errors='coerce')
        return pd.to_datetime(tsf, unit='s', errors='coerce')
    except Exception:
        return None

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

    # Support special classic variants like 'zero'
    classic_suffix = aggregation if aggregation else "1d"
    path = f"{symbol}/classic/{classic_suffix}"

    try:
        data = _make_request(path, api_key=api_key)
        
        # The API may return a single object or a list of objects per timestamp
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            df = pd.DataFrame(data['data'])
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame()
            
        if df.empty:
            return df
            
        # Convert timestamp
        if 'timestamp' in df.columns:
            parsed = df['timestamp'].apply(_parse_epoch_timestamp)
            if parsed.notna().any():
                df['timestamp'] = parsed
                df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
        else:
            # Try alternative fields
            for ts_field in ['time', 'date', 'datetime', 't']:
                if ts_field in df.columns:
                    tser = pd.to_datetime(df[ts_field], errors='coerce')
                    if tser.notna().any():
                        df['timestamp'] = tser
                        df = df.dropna(subset=['timestamp']).set_index('timestamp').sort_index()
                        break
        
        # Filter by date range if we have a datetime index
        if isinstance(df.index, pd.DatetimeIndex):
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df.loc[start_dt:end_dt]

        # Map fields from classic schema to friendly names
        if 'spot' in df.columns and 'price' not in df.columns:
            df = df.rename(columns={'spot': 'price'})
        
        # Handle both classic/zero schema (major_pos_*/major_neg_*) and major levels schema (mpos_*/mneg_*)
        # Map to standardized names
        rename_map = {
            'mpos_oi': 'major_call_oi',
            'mpos_vol': 'major_call_vol',
            'mneg_oi': 'major_put_oi',
            'mneg_vol': 'major_put_vol',
            'net_gex_oi': 'sum_gex_oi',
            'net_gex_vol': 'sum_gex_vol',
            'major_pos_oi': 'major_call_oi',
            'major_pos_vol': 'major_call_vol',
            'major_neg_oi': 'major_put_oi',
            'major_neg_vol': 'major_put_vol',
        }
        # Apply renames where source exists and target doesn't
        for src, tgt in rename_map.items():
            if src in df.columns and tgt not in df.columns:
                df = df.rename(columns={src: tgt})
        
        # Ensure numeric types where possible
        for c in ['price','zero_gamma','sum_gex_oi','sum_gex_vol',
                  'major_call_oi','major_call_vol','major_put_oi','major_put_vol']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        # Arrays (strikes, max_priors) remain as object columns
            
        return df
    except Exception as e:
        logger.error(f"Failed to load historical GEX data: {str(e)}")
        raise GexbotError(f"Historical data load failed: {str(e)}") from e

def load_max_change(
    symbol: str,
    api_key: str
) -> pd.DataFrame:
    """
    Load max change data for a symbol from classic/max endpoint.
    
    API returns arrays for different time windows: current, 1min, 5min, 10min, 15min, 30min.
    Each array contains [timestamp?, value?] - structure TBD based on actual response.
    
    Args:
        symbol: Stock/index symbol (e.g. 'SPX', 'SPY')
        api_key: Required gexbot.com API key
    
    Returns:
        DataFrame with columns for each time window (current, one, five, ten, fifteen, thirty)
        and timestamp index if timestamps are embedded in arrays.
    """
    if not api_key:
        raise GexbotError("API key is required for gexbot.com")

    path = f"{symbol}/classic/max"

    try:
        data = _make_request(path, api_key=api_key)
        
        # Parse the response - it should have timestamp, ticker, and arrays
        if not isinstance(data, dict):
            raise GexbotError(f"Unexpected max change response type: {type(data)}")
        
        # Extract timestamp and ticker
        ts = data.get('timestamp')
        ticker = data.get('ticker')
        
        # Build a record with flattened arrays
        # Arrays might be simple lists of numbers or nested [ts, val] pairs
        record = {'timestamp': _parse_epoch_timestamp(ts) if ts else None, 'ticker': ticker}
        
        for window in ['current', 'one', 'five', 'ten', 'fifteen', 'thirty']:
            arr = data.get(window, [])
            if isinstance(arr, list) and len(arr) > 0:
                # If nested arrays, store as-is for now
                record[f'max_change_{window}'] = arr
            else:
                record[f'max_change_{window}'] = None
        
        df = pd.DataFrame([record])
        if df['timestamp'].notna().any():
            df = df.set_index('timestamp')
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to load max change data: {str(e)}")
        raise GexbotError(f"Max change load failed: {str(e)}") from e

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
            # Convert timestamp
            if 'timestamp' in data:
                data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
            # Extract gamma levels and print actionable signals
            gamma_info = extract_gamma_levels(data)
            # Example: print actionable logic (replace with your trade logic)
            if gamma_info:
                net_gamma = gamma_info.get('net_gamma')
                major_pos = gamma_info.get('major_pos')
                major_neg = gamma_info.get('major_neg')
                zero_gamma = gamma_info.get('zero_gamma')
                try:
                    net_gamma_str = f"{float(net_gamma):.2e}" if net_gamma is not None else "n/a"
                except Exception:
                    net_gamma_str = "n/a"
                # Include call/put majors if available
                print(f"[GEXBOT] Net gamma: {net_gamma_str} | Major+ {major_pos} | Major- {major_neg} | ZeroG: {zero_gamma}")
                # Example logic: print trade zone
                price = data.get('price', None)
                if price is not None:
                    if isinstance(major_pos, (pd.Series, list)):
                        pos_level = major_pos.iloc[0] if hasattr(major_pos, 'iloc') else major_pos[0]
                        neg_level = major_neg.iloc[0] if hasattr(major_neg, 'iloc') else major_neg[0]
                    else:
                        pos_level = major_pos
                        neg_level = major_neg
                    if pos_level is not None and price >= pos_level:
                        print("[GEXBOT] Price at/above major positive gamma: consider selling or taking profits.")
                    elif neg_level is not None and price <= neg_level:
                        print("[GEXBOT] Price at/below major negative gamma: consider buying or covering shorts.")
                    elif zero_gamma is not None and abs(price - zero_gamma) < 1e-3:
                        print("[GEXBOT] Price near zero gamma: caution, indecision zone.")
            if callback:
                callback(data)
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